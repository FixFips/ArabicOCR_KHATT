# src/model.py
"""Shared CRNN-CTC model and decoding functions for Arabic handwriting OCR."""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    CRNN with multi-scale vertical encoding for Arabic script.

    Arabic letters (ba/ta/tha/nun/ya) share the same base stroke and differ
    only by dot position (above vs below baseline).  Preserving 3 vertical
    zones through adaptive pooling lets the RNN see WHERE dots sit, not just
    WHETHER they exist.

    Architecture:
        CNN (7 conv layers, full BatchNorm, Dropout2d after pools)
        -> adaptive_avg_pool2d to (3, W')  [3 vertical zones]
        -> flatten to (B, 1536, W')
        -> 2-layer BiLSTM(384)
        -> Dropout -> FC -> (T, B, num_classes)
    """

    def __init__(self, num_classes: int, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        # --- CNN feature extractor (full BatchNorm for dot-feature parity) ---
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Block 4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout2d(0.1),
            # Block 5
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Block 6
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 7
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # --- RNN sequence modeller ---
        # Input: 512 channels × 3 vertical zones = 1536 features per timestep
        self.rnn = nn.LSTM(
            512 * 3, 384,
            bidirectional=True, num_layers=2,
            batch_first=True, dropout=0.2,
        )
        # --- Optional transformer encoder layer (arch v3) ---
        # Adds global context on top of the BiLSTM's local sequential pass.
        # One layer keeps parameter count modest (~2.5M extra on top of ~15M).
        # BiLSTM output already carries positional information, so no explicit PE.
        if use_attention:
            self.attn = nn.TransformerEncoderLayer(
                d_model=768, nhead=4, dim_feedforward=1024,
                dropout=0.2, activation="gelu",
                batch_first=True, norm_first=True,
            )
        else:
            self.attn = None
        self.fc_drop = nn.Dropout(0.3)
        self.fc = nn.Linear(384 * 2, num_classes)  # bidirectional → 768

    def forward(self, x):                           # [B, 1, H, W]
        x = self.cnn(x)                             # [B, 512, H', W']
        B, C, H, W = x.shape
        # Preserve 3 vertical zones (above-baseline / baseline / below-baseline)
        x = F.adaptive_avg_pool2d(x, (3, W))        # [B, 512, 3, W']
        x = x.view(B, C * 3, W).permute(0, 2, 1)   # [B, W', 1536]
        x, _ = self.rnn(x)                          # [B, W', 768]
        if self.attn is not None:
            x = self.attn(x)                        # [B, W', 768] — global context
        x = self.fc_drop(x)
        x = self.fc(x)                              # [B, W', num_classes]
        return x.permute(1, 0, 2)                   # [T=W', B, C]


# --------------- Text / ID helpers ---------------

def text_to_ids(text: str, char2id: dict, unk_id: int) -> list[int]:
    return [char2id.get(ch, unk_id) for ch in text]


def ids_to_text(ids, id2char: dict) -> str:
    out = []
    for i in ids:
        ch = id2char.get(int(i), "")
        if not ch:
            continue
        if ch.startswith("<") and ch.endswith(">"):
            continue
        out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


# --------------- CTC decoding ---------------

def ctc_greedy_decode(logits, id2char: dict) -> list[str]:
    """Standard greedy CTC decode: argmax → collapse repeats → remove blanks."""
    pred = logits.argmax(-1).detach().cpu().numpy()  # [T, B]
    T, B = pred.shape
    texts = []
    for b in range(B):
        seq, last = [], -1
        for t in range(T):
            p = int(pred[t, b])
            if p != last and p != 0:  # 0 = CTC blank
                seq.append(p)
            last = p
        texts.append(ids_to_text(seq, id2char))
    return texts


def ctc_beam_decode(
    logits,
    id2char: dict,
    beam_width: int = 10,
    bigram_lm=None,
    lm_weight: float = 0.3,
) -> list[str]:
    """
    Prefix beam search CTC decoder with optional Arabic bigram LM.

    Arabic character sequences are highly constrained by morphology —
    after certain letter combinations, only specific letters can follow.
    The bigram LM encodes these constraints to resolve dot-group ambiguity
    (e.g. ba vs ta vs tha).
    """

    log_probs = logits.log_softmax(-1).detach().cpu()  # [T, B, C]
    T, B, C = log_probs.shape
    texts = []

    for b in range(B):
        # beams: dict mapping prefix (tuple of ids) -> (log_prob_blank, log_prob_nonblank)
        beams = {(): (0.0, float("-inf"))}

        for t in range(T):
            new_beams: dict[tuple, list[float]] = {}

            # Only consider top-k characters per timestep for speed
            top_k = min(beam_width * 3, C)
            vals, idxs = log_probs[t, b].topk(top_k)

            for prefix, (pb, pnb) in beams.items():
                p_prefix = _log_add(pb, pnb)

                for ki in range(top_k):
                    c = int(idxs[ki])
                    lp = float(vals[ki])

                    if c == 0:  # blank
                        key = prefix
                        ob, onb = new_beams.get(key, (float("-inf"), float("-inf")))
                        new_beams[key] = (_log_add(ob, p_prefix + lp), onb)
                    elif prefix and c == prefix[-1]:
                        # same char → extend only from blank path
                        key = prefix + (c,)
                        ob, onb = new_beams.get(key, (float("-inf"), float("-inf")))
                        new_beams[key] = (ob, _log_add(onb, pb + lp))
                        # also keep prefix unchanged via non-blank path
                        key2 = prefix
                        ob2, onb2 = new_beams.get(key2, (float("-inf"), float("-inf")))
                        new_beams[key2] = (ob2, _log_add(onb2, pnb + lp))
                    else:
                        key = prefix + (c,)
                        lm_bonus = 0.0
                        if bigram_lm is not None and prefix:
                            lm_bonus = lm_weight * bigram_lm.get(
                                (prefix[-1], c), bigram_lm.get(("_default",), -5.0)
                            )
                        ob, onb = new_beams.get(key, (float("-inf"), float("-inf")))
                        new_beams[key] = (ob, _log_add(onb, p_prefix + lp + lm_bonus))

            # Prune to beam_width
            scored = [(k, _log_add(v[0], v[1])) for k, v in new_beams.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            beams = {}
            for k, _ in scored[:beam_width]:
                beams[k] = new_beams[k]

        # Best beam
        best_prefix = max(beams, key=lambda k: _log_add(beams[k][0], beams[k][1]))
        texts.append(ids_to_text(list(best_prefix), id2char))

    return texts


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    mx = max(a, b)

    return mx + math.log1p(math.exp(-abs(a - b)))


def build_bigram_lm(texts: list[str], char2id: dict) -> dict:
    """
    Build a character-level bigram LM from training labels.

    Returns dict mapping (prev_char_id, next_char_id) -> log_probability.
    """

    counts: dict[tuple, int] = {}
    total_per_prev: dict[int, int] = {}

    for text in texts:
        ids = [char2id.get(ch, char2id.get("<unk>", 1)) for ch in text]
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
            total_per_prev[ids[i]] = total_per_prev.get(ids[i], 0) + 1

    # Convert to log probabilities with add-1 smoothing
    vocab_size = len(char2id)
    lm = {}
    for (prev_id, next_id), count in counts.items():
        total = total_per_prev[prev_id]
        lm[(prev_id, next_id)] = math.log((count + 1) / (total + vocab_size))

    # Default for unseen bigrams
    lm[("_default",)] = math.log(1.0 / vocab_size)
    return lm
