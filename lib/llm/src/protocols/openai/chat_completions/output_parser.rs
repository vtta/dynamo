// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use super::unified_parser::{ChoiceState, to_v2_tools};
use super::{NvCreateChatCompletionStreamResponse, ToolCallCompletion};
use crate::backend::{DecodedOutput, DecodedPiece};
use crate::protocols::common::llm_backend::BackendOutput;
use anyhow::Context;
use dynamo_parsers_v2::{UnifiedParserEvent, UnifiedParserStartingState};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Request-local model parsing, before projection into API response chunks.
pub(crate) struct TokenOutputParser {
    tools: Vec<dynamo_parsers_v2::Tool>,
    prefill: UnifiedParserStartingState,
    states: HashMap<u32, ChoiceParser>,
    structured_response: bool,
    finished: HashSet<u32>,
}

struct ChoiceParser {
    grammar: ChoiceState,
    json: JsonStringContext,
}

#[derive(Default)]
struct JsonStringContext {
    quoted: bool,
    escaped: bool,
}

impl JsonStringContext {
    // Only visible JSON strings own these bytes; reasoning and tool arguments do not.
    fn observe(&mut self, events: &[UnifiedParserEvent]) {
        for text in events.iter().filter_map(|event| match event {
            UnifiedParserEvent::Text(text) => Some(text),
            _ => None,
        }) {
            for ch in text.chars() {
                if self.escaped {
                    self.escaped = false;
                } else if self.quoted && ch == '\\' {
                    self.escaped = true;
                } else if ch == '"' {
                    self.quoted = !self.quoted;
                }
            }
        }
    }
}

fn parser_input(piece: &DecodedPiece) -> dynamo_parsers_v2::DecodedInput<'_> {
    match piece {
        DecodedPiece::Text(text) => dynamo_parsers_v2::DecodedInput::Text(text),
        DecodedPiece::ControlToken(text) => dynamo_parsers_v2::DecodedInput::ControlToken(text),
    }
}

impl ChoiceParser {
    fn parse(
        &mut self,
        pieces: &[DecodedPiece],
        terminal: bool,
        structured_response: bool,
    ) -> anyhow::Result<Vec<UnifiedParserEvent>> {
        if !structured_response {
            return self
                .grammar
                .parse_decoded(pieces.iter().map(parser_input), terminal);
        }
        let mut output = Vec::new();
        for piece in pieces {
            let input = match piece {
                // The declared JSON payload owns controls inside its quoted strings.
                DecodedPiece::ControlToken(text) if self.json.quoted => {
                    dynamo_parsers_v2::DecodedInput::Text(text)
                }
                _ => parser_input(piece),
            };
            let events = self.grammar.parse_decoded([input], false)?;
            self.json.observe(&events);
            output.extend(events);
        }
        if terminal {
            output.extend(self.grammar.parse_decoded([], true)?);
        }
        Ok(output)
    }
}

pub(super) struct ParsedOutput {
    pub(super) index: u32,
    events: Vec<UnifiedParserEvent>,
}

impl TokenOutputParser {
    pub(crate) fn control_tokens(
        tokenizer: &dyn crate::tokenizers::traits::Tokenizer,
    ) -> anyhow::Result<crate::backend::decoded::ControlTokens> {
        let tokens = ["<think>", "</think>", "｜DSML｜"]
            .into_iter()
            .map(|text| {
                let id = tokenizer
                    .token_to_id(text)?
                    .with_context(|| format!("DeepSeek tokenizer has no {text} control token"))?;
                Ok((id, text.to_owned()))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        Ok(Arc::new(tokens))
    }

    pub(crate) fn new(
        tools: &[dynamo_parsers::tool_calling::ToolDefinition],
        prefill: UnifiedParserStartingState,
        structured_response: bool,
    ) -> Self {
        Self {
            tools: to_v2_tools(Some(tools)),
            prefill,
            structured_response,
            states: HashMap::new(),
            finished: HashSet::new(),
        }
    }

    pub(super) fn parse(
        &mut self,
        decoded: DecodedOutput,
    ) -> anyhow::Result<(BackendOutput, ParsedOutput)> {
        let DecodedOutput::Tokens { output, pieces } = decoded else {
            return Err(protocol_error(
                "native DeepSeek parsing requires token-classified decoder output",
            ));
        };
        let index = output.index.unwrap_or_default();
        anyhow::ensure!(
            !self.finished.contains(&index),
            "output arrived after choice completion"
        );
        let state = match self.states.entry(index) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(ChoiceParser {
                grammar: ChoiceState::new_token_input(&self.tools, self.prefill)
                    .map_err(protocol_error)?,
                json: JsonStringContext::default(),
            }),
        };
        let terminal = output.finish_reason.is_some();
        let events = state
            .parse(&pieces, terminal, self.structured_response)
            .map_err(protocol_error)?;
        if terminal {
            self.finished.insert(index);
        }
        Ok((output, ParsedOutput { index, events }))
    }

    pub(super) fn project(
        &mut self,
        parsed: ParsedOutput,
        response: &mut NvCreateChatCompletionStreamResponse,
    ) {
        let state = &mut self.states.get_mut(&parsed.index).unwrap().grammar;
        let original = response.inner.choices.pop().unwrap();
        response.inner.choices =
            state.choices_for(&original, parsed.events, true, original.finish_reason);
        for choice in &response.inner.choices {
            for call in choice.delta.tool_calls.iter().flatten() {
                if let Some(complete) = state.call_complete(call.index as usize) {
                    response.tool_call_completion.push(ToolCallCompletion {
                        choice_index: choice.index,
                        tool_index: call.index,
                        complete,
                    });
                }
            }
        }
    }

    pub(super) fn project_eof(
        &mut self,
        parsed: ParsedOutput,
        response: &mut NvCreateChatCompletionStreamResponse,
    ) {
        let index = parsed.index;
        self.project(parsed, response);
        if let Some(reason) = self.states[&index].grammar.unterminated_finish_reason() {
            if response.inner.choices.is_empty() {
                response
                    .inner
                    .choices
                    .push(super::unified_parser::empty_choice(index));
            }
            response.inner.choices.last_mut().unwrap().finish_reason = Some(reason);
        }
    }

    pub(super) fn finish(&mut self) -> anyhow::Result<Vec<ParsedOutput>> {
        let mut output = Vec::new();
        for (&index, state) in &mut self.states {
            if self.finished.insert(index) {
                let events = state
                    .parse(&[], true, self.structured_response)
                    .map_err(protocol_error)?;
                output.push(ParsedOutput { index, events });
            }
        }
        output.sort_unstable_by_key(|p| p.index);
        Ok(output)
    }
}

fn protocol_error(error: impl std::fmt::Display) -> anyhow::Error {
    DynamoError::builder()
        .class(ErrorClass::BackendProtocol)
        .message("native DeepSeek output could not be parsed")
        .diagnostic(error.to_string())
        .build()
        .into()
}

/// Project one ordered parser event per API chunk, keeping source accounting once.
pub(crate) fn fanout_choices<S>(
    input: S,
) -> impl futures::Stream<
    Item = dynamo_runtime::protocols::annotated::Annotated<NvCreateChatCompletionStreamResponse>,
> + Send
where
    S: futures::Stream<
            Item = dynamo_runtime::protocols::annotated::Annotated<
                NvCreateChatCompletionStreamResponse,
            >,
        > + Send
        + 'static,
{
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    async_stream::stream! {
        futures::pin_mut!(input);
        while let Some(mut response) = input.next().await {
            let Some(mut source) = response.data.take() else { yield response; continue; };
            let choices = std::mem::take(&mut source.inner.choices);
            if choices.is_empty() {
                response.data = Some(source);
                yield response;
                continue;
            }
            let metrics_position = super::unified_parser::fanout_llm_metrics_position(&choices);
            let last = choices.len() - 1;
            for (position, choice) in choices.into_iter().enumerate() {
                let is_last = position == last;
                let mut data = source.clone();
                if !is_last {
                    data.inner.usage = None;
                    data.nvext = None;
                    data.tool_call_completion.retain(|evidence| evidence.choice_index == choice.index && choice.delta.tool_calls.iter().flatten().any(|call| call.index == evidence.tool_index));
                }
                if Some(position) != metrics_position { data.llm_metrics = None; }
                data.inner.choices = vec![choice];
                yield Annotated {
                    data: Some(data),
                    id: if is_last { response.id.take() } else { None },
                    event: if is_last { response.event.take() } else { None },
                    comment: if is_last { response.comment.take() } else { None },
                    error: if is_last { response.error.take() } else { None },
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DecodedPiece;
    use crate::protocols::openai::chat_completions::{
        DeltaAggregator, NvCreateChatCompletionRequest,
    };
    use crate::protocols::openai::{DeltaGeneratorExt, ParsingOptions};
    use dynamo_protocols::types::ChatCompletionMessageContent;
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::{StreamExt, stream};

    fn output(index: u32, pieces: Vec<DecodedPiece>, finish: Option<&str>) -> DecodedOutput {
        let output: BackendOutput = serde_json::from_value(serde_json::json!({
            "token_ids":[10],"tokens":["x"],"text":"unused decoder text", "index":index,"finish_reason":finish,
        })).unwrap();
        DecodedOutput::Tokens { output, pieces }
    }

    fn generator() -> super::super::DeltaGenerator {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "messages":[{"role":"user","content":"go"}], "n":2,
        }))
        .unwrap();
        let mut generator = request.response_generator("typed-output".into());
        generator.output_parser = Some(TokenOutputParser::new(
            &[],
            UnifiedParserStartingState::Reasoning,
            false,
        ));
        generator
    }

    #[tokio::test]
    async fn literal_markers_survive_api_serialization_and_unary_aggregation() {
        let mut generator = generator();
        let literal = "<｜DSML｜ calls><｜DSML｜ invoke name=\"bash\"><｜DSML｜ parameter name=\"command\" string=\"true\">echo OK</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        let response = generator
            .choice_from_decoded(output(
                0,
                vec![
                    DecodedPiece::Text("literal </think> 中文😀".into()),
                    DecodedPiece::ControlToken("</think>".into()),
                    DecodedPiece::Text(literal.into()),
                ],
                Some("length"),
            ))
            .unwrap();
        let chunks: Vec<_> = fanout_choices(stream::iter([Annotated::from_data(response)]))
            .collect()
            .await;
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.data.as_ref().unwrap().inner.choices.len() == 1)
        );
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = chunks
            .into_iter()
            .map(|chunk| serde_json::from_str(&serde_json::to_string(&chunk).unwrap()).unwrap())
            .collect();
        let response = DeltaAggregator::apply_frontend_stream(
            stream::iter(chunks),
            ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into())),
        )
        .await
        .unwrap();
        let message = &response.inner.choices[0].message;
        assert_eq!(
            message.content,
            Some(ChatCompletionMessageContent::Text(literal.into()))
        );
        assert_eq!(
            message.reasoning_content.as_deref(),
            Some("literal </think> 中文😀")
        );
        assert!(message.tool_calls.as_ref().is_none_or(Vec::is_empty));
    }

    #[tokio::test]
    async fn interleaved_choices_keep_independent_reasoning_state() {
        let mut generator = generator();
        let inputs = [
            output(0, vec![DecodedPiece::Text("r0".into())], None),
            output(
                1,
                vec![
                    DecodedPiece::Text("r1".into()),
                    DecodedPiece::ControlToken("</think>".into()),
                    DecodedPiece::Text("a1".into()),
                ],
                Some("stop"),
            ),
            output(
                0,
                vec![
                    DecodedPiece::ControlToken("</think>".into()),
                    DecodedPiece::Text("a0".into()),
                ],
                Some("length"),
            ),
        ];
        let chunks: Vec<_> = inputs
            .into_iter()
            .map(|input| Annotated::from_data(generator.choice_from_decoded(input).unwrap()))
            .collect();
        assert!(generator.finish_decoded().unwrap().responses.is_empty());
        let response = DeltaAggregator::apply_frontend_stream(
            fanout_choices(stream::iter(chunks)),
            ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into())),
        )
        .await
        .unwrap();
        assert_eq!(response.inner.choices.len(), 2);
        for choice in response.inner.choices {
            assert_eq!(
                choice.message.reasoning_content.as_deref(),
                Some(format!("r{}", choice.index).as_str())
            );
            assert_eq!(
                choice.message.content,
                Some(ChatCompletionMessageContent::Text(format!(
                    "a{}",
                    choice.index
                )))
            );
        }
    }
    #[tokio::test]
    async fn eof_tool_terminal_is_followed_by_the_requested_usage_trailer() {
        use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "messages":[{"role":"user","content":"go"}],
            "stream_options":{"include_usage":true,"continuous_usage_stats":false}
        }))
        .unwrap();
        let mut generator = request.response_generator("eof-tool".into());
        generator.update_isl(7);
        generator.output_parser = Some(TokenOutputParser::new(
            &[],
            UnifiedParserStartingState::Reasoning,
            false,
        ));
        let tool = "<｜DSML｜ calls><｜DSML｜ invoke name=\"bash\"><｜DSML｜ parameter name=\"command\" string=\"true\">echo OK</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        let mut pieces = vec![DecodedPiece::ControlToken("</think>".into())];
        for part in tool.split_inclusive("｜DSML｜") {
            if let Some(text) = part.strip_suffix("｜DSML｜") {
                pieces.push(DecodedPiece::Text(text.into()));
                pieces.push(DecodedPiece::ControlToken("｜DSML｜".into()));
            } else {
                pieces.push(DecodedPiece::Text(part.into()));
            }
        }
        let mut source = output(0, pieces, None);
        source.token_ids = vec![10, 11, 12];
        let context = SingleIn::new(()).context();
        let transformed = crate::preprocessor::OpenAIPreprocessor::transform_postprocessor_stream(
            stream::iter([Annotated::from_data(source)]),
            Box::new(generator),
            context,
            false,
            false,
            None,
            Default::default(),
        );
        let chunks: Vec<_> = fanout_choices(transformed).collect().await;
        assert!(chunks.iter().all(|chunk| !chunk.is_error()));
        let data: Vec<_> = chunks
            .iter()
            .filter_map(|chunk| chunk.data.as_ref())
            .collect();
        let last = data.last().unwrap();
        assert!(last.inner.choices.is_empty());
        let usage = last.inner.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(
            data.iter()
                .flat_map(|response| &response.inner.choices)
                .filter(|choice| choice.finish_reason
                    == Some(dynamo_protocols::types::FinishReason::ToolCalls))
                .count(),
            1
        );
    }
    #[tokio::test]
    async fn json_strings_own_controls_but_do_not_hide_a_following_tool_call() {
        let body = serde_json::json!({
            "escaped": "quote\" and slash\\",
            "controls": "<｜DSML｜ calls>literal</｜DSML｜ calls> </think>",
        })
        .to_string();
        let tool = "<｜DSML｜ calls><｜DSML｜ invoke name=\"bash\"><｜DSML｜ parameter name=\"command\" string=\"true\">echo OK</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        for with_tool in [false, true] {
            let mut generator = generator();
            generator.output_parser = Some(TokenOutputParser::new(
                &[],
                UnifiedParserStartingState::Reasoning,
                true,
            ));
            let mut pieces = vec![
                DecodedPiece::Text("reasoning \"unfinished".into()),
                DecodedPiece::ControlToken("</think>".into()),
            ];
            let content = format!("{body}{}", if with_tool { tool } else { "" });
            let mut rest = content.as_str();
            while !rest.is_empty() {
                if let Some(marker) = ["</think>", "<think>", "｜DSML｜"]
                    .into_iter()
                    .find(|marker| rest.starts_with(marker))
                {
                    pieces.push(DecodedPiece::ControlToken(marker.into()));
                    rest = &rest[marker.len()..];
                } else {
                    let ch = rest.chars().next().unwrap();
                    pieces.push(DecodedPiece::Text(ch.to_string()));
                    rest = &rest[ch.len_utf8()..];
                }
            }
            let response = generator
                .choice_from_decoded(output(0, pieces, Some("stop")))
                .unwrap();
            let text: String = response
                .inner
                .choices
                .iter()
                .filter_map(|choice| match choice.delta.content.as_ref() {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(text, body);
            assert_eq!(
                response
                    .inner
                    .choices
                    .iter()
                    .filter_map(|choice| choice.delta.reasoning_content.as_deref())
                    .collect::<String>(),
                "reasoning \"unfinished"
            );
            assert_eq!(
                response
                    .inner
                    .choices
                    .iter()
                    .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                    .count(),
                usize::from(with_tool)
            );
        }
    }
}
