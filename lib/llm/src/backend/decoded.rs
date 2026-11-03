// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, DerefMut};

use crate::protocols::common::llm_backend::BackendOutput;
use anyhow::Context;
use std::sync::Arc;

pub(crate) type ControlTokens = Arc<HashMap<u32, String>>;

/// Incremental decoder output before model syntax is parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedPiece {
    Text(String),
    ControlToken(String),
}

/// Frontend-only stream between decoding and response generation.
/// It is deliberately not a wire type: raw workers still return LLMEngineOutput.
#[derive(Debug, Clone)]
pub enum DecodedOutput {
    Text(BackendOutput),
    Tokens {
        output: BackendOutput,
        pieces: Vec<DecodedPiece>,
    },
}

impl DecodedOutput {
    pub(crate) fn new(output: BackendOutput, pieces: Option<Vec<DecodedPiece>>) -> Self {
        match pieces {
            Some(pieces) => Self::Tokens { output, pieces },
            None => Self::Text(output),
        }
    }

    pub fn into_output(self) -> BackendOutput {
        match self {
            Self::Text(output) | Self::Tokens { output, .. } => output,
        }
    }
}

impl From<BackendOutput> for DecodedOutput {
    fn from(output: BackendOutput) -> Self {
        Self::Text(output)
    }
}

impl Deref for DecodedOutput {
    type Target = BackendOutput;
    fn deref(&self) -> &BackendOutput {
        match self {
            Self::Text(output) | Self::Tokens { output, .. } => output,
        }
    }
}

impl DerefMut for DecodedOutput {
    fn deref_mut(&mut self) -> &mut BackendOutput {
        match self {
            Self::Text(output) | Self::Tokens { output, .. } => output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingPiece {
    text: String,
    consumed: usize,
    control: bool,
}

/// Keeps token classification aligned with the decoder's released stop-filtered text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedPieces {
    controls: ControlTokens,
    pending: VecDeque<PendingPiece>,
}

impl DecodedPieces {
    pub(crate) fn new(controls: ControlTokens) -> Self {
        Self {
            controls,
            pending: VecDeque::new(),
        }
    }

    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(last) = self.pending.back_mut().filter(|p| !p.control) {
            if last.consumed > 0 {
                last.text.drain(..last.consumed);
                last.consumed = 0;
            }
            last.text.push_str(text);
        } else {
            self.pending.push_back(PendingPiece {
                text: text.to_owned(),
                consumed: 0,
                control: false,
            });
        }
    }

    pub(crate) fn record(&mut self, id: Option<u32>, text: Option<&str>) -> anyhow::Result<()> {
        let Some(text) = text else {
            return Ok(());
        };
        if let Some(marker) = id.and_then(|id| self.controls.get(&id)).cloned() {
            let prefix = text
                .strip_suffix(&marker)
                .context("control token spelling changed during incremental decoding")?;
            self.text(prefix);
            self.pending.push_back(PendingPiece {
                text: marker,
                consumed: 0,
                control: true,
            });
        } else {
            self.text(text);
        }
        Ok(())
    }

    pub(crate) fn release(
        &mut self,
        text: &str,
        finished: bool,
    ) -> anyhow::Result<Vec<DecodedPiece>> {
        let mut rest = text;
        let mut output = Vec::new();
        while !rest.is_empty() {
            let piece = self
                .pending
                .front_mut()
                .context("released text has no incremental decoder source")?;
            let pending = &piece.text[piece.consumed..];
            let n = pending.len().min(rest.len());
            anyhow::ensure!(
                pending.as_bytes()[..n] == rest.as_bytes()[..n]
                    && pending.is_char_boundary(n)
                    && rest.is_char_boundary(n),
                "decoded token and released text disagree"
            );
            if !piece.control {
                push_text(&mut output, &rest[..n]);
            }
            piece.consumed += n;
            rest = &rest[n..];
            if piece.consumed == piece.text.len() {
                let piece = self.pending.pop_front().unwrap();
                if piece.control {
                    output.push(DecodedPiece::ControlToken(piece.text));
                }
            }
        }
        if finished {
            // A stop may release only a control token's prefix; it is literal text.
            if let Some(piece) = self.pending.front().filter(|p| p.control && p.consumed > 0) {
                push_text(&mut output, &piece.text[..piece.consumed]);
            }
            self.pending.clear();
        }
        Ok(output)
    }
}

fn push_text(output: &mut Vec<DecodedPiece>, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(DecodedPiece::Text(last)) = output.last_mut() {
        last.push_str(text);
    } else {
        output.push(DecodedPiece::Text(text.to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decoder() -> DecodedPieces {
        DecodedPieces {
            controls: Arc::new(HashMap::from([
                (1, "</think>".into()),
                (2, "｜DSML｜".into()),
            ])),
            pending: VecDeque::new(),
        }
    }
    #[test]
    fn origin_not_spelling_and_unicode_backlog_define_control() {
        let mut d = decoder();
        d.record(Some(3), Some("literal </think> 中文")).unwrap();
        d.record(Some(1), Some("�</think>")).unwrap();
        d.record(Some(2), Some("｜DSML｜")).unwrap();
        assert_eq!(
            d.release("literal </think> 中文�</think>｜DSML｜", true)
                .unwrap(),
            vec![
                DecodedPiece::Text("literal </think> 中文�".into()),
                DecodedPiece::ControlToken("</think>".into()),
                DecodedPiece::ControlToken("｜DSML｜".into())
            ]
        );
    }
    #[test]
    fn stop_trimmed_control_is_never_reintroduced() {
        let mut d = decoder();
        d.record(Some(1), Some("</think>")).unwrap();
        assert_eq!(
            d.release("</thi", true).unwrap(),
            vec![DecodedPiece::Text("</thi".into())]
        );
        let mut d = decoder();
        d.record(Some(1), Some("</think>")).unwrap();
        assert!(d.release("", true).unwrap().is_empty());
    }
    #[test]
    fn withheld_control_survives_a_checkpoint_clone() {
        let mut d = decoder();
        d.record(Some(1), Some("</think>")).unwrap();
        assert!(d.release("</thi", false).unwrap().is_empty());
        assert_eq!(
            d.clone().release("nk>", true).unwrap(),
            vec![DecodedPiece::ControlToken("</think>".into())]
        );
    }
    #[test]
    fn a_repeated_stop_prefix_does_not_retain_emitted_text() {
        let mut d = decoder();
        d.record(None, Some("a\n")).unwrap();
        d.release("a", false).unwrap();
        for _ in 0..1000 {
            d.record(None, Some("a\n")).unwrap();
            d.release("\na", false).unwrap();
            assert_eq!(d.pending.len(), 1);
            assert!(d.pending[0].text.len() <= 3);
        }
    }

    #[test]
    fn checkpoint_keeps_token_kinds_only_in_process() {
        use crate::protocols::common::llm_backend::DecoderCheckpoint;
        let mut checkpoint = DecoderCheckpoint::new("held".into(), vec![3]);
        checkpoint.decoded_pieces = Some(decoder());
        assert!(checkpoint.clone().decoded_pieces.is_some());
        let wire = serde_json::to_string(&checkpoint).unwrap();
        assert_eq!(wire, "\"held\"");
        assert!(
            serde_json::from_str::<DecoderCheckpoint>(&wire)
                .unwrap()
                .decoded_pieces
                .is_none()
        );
    }
}
