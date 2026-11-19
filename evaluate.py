import sys
import argparse
import json
import math

sys.dont_write_bytecode = True

import torch
from torch.utils.data import DataLoader

from dataset import HindiWordDataset, ctc_collate
from model import CRNN
from tokenizer import CTCTextEncoder


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def greedy_decode(log_probs, encoder, input_lengths=None):
    pred_ids = log_probs.argmax(2).permute(1, 0)
    if input_lengths is None:
        return [encoder.decode(seq.tolist()) for seq in pred_ids]
    return [
        encoder.decode(seq[: int(length)].tolist())
        for seq, length in zip(pred_ids, input_lengths)
    ]


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_search_single(log_probs_2d, encoder, beam_width: int, top_k: int = 40):
    """
    CTC beam search for a single sample.

    Args:
        log_probs_2d: (T, C) log-probability tensor (already trimmed to input_length)
        encoder:      CTCTextEncoder
        beam_width:   number of hypotheses to keep at each step
        top_k:        only extend with the top-k tokens per timestep for speed

    Returns:
        decoded string
    """
    NEG_INF = float("-inf")
    blank = encoder.blank_id
    T = log_probs_2d.shape[0]

    # beam maps prefix (tuple of token ids) -> [log_p_blank, log_p_non_blank]
    beam = {(): [0.0, NEG_INF]}

    for t in range(T):
        lp = log_probs_2d[t]  # (C,)

        # Only consider the top-k most probable tokens for speed.
        topk = min(top_k, len(encoder))
        topk_vals, topk_ids = lp.topk(topk)
        top_ids = topk_ids.tolist()
        # Always include blank even if not in top-k
        if blank not in top_ids:
            top_ids.append(blank)

        new_beam: dict = {}

        for prefix, (log_pb, log_pnb) in beam.items():
            log_p_total = _log_add(log_pb, log_pnb)

            # Extend with blank — prefix stays the same
            new_log_pb = log_p_total + lp[blank].item()
            if prefix not in new_beam:
                new_beam[prefix] = [NEG_INF, NEG_INF]
            new_beam[prefix][0] = _log_add(new_beam[prefix][0], new_log_pb)

            # Extend with each non-blank token in top-k
            for c in top_ids:
                if c == blank:
                    continue
                lp_c = lp[c].item()
                new_prefix = prefix + (c,)

                if prefix and prefix[-1] == c:
                    # Repeated token: only the blank-ending path can extend
                    # without merging (CTC collapse rule)
                    new_log_pnb = log_pb + lp_c
                else:
                    new_log_pnb = log_p_total + lp_c

                if new_prefix not in new_beam:
                    new_beam[new_prefix] = [NEG_INF, NEG_INF]
                new_beam[new_prefix][1] = _log_add(
                    new_beam[new_prefix][1], new_log_pnb
                )

        # Prune to beam_width by total log-probability
        beam = dict(
            sorted(
                new_beam.items(),
                key=lambda x: _log_add(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    if not beam:
        return ""
    best = max(beam.items(), key=lambda x: _log_add(x[1][0], x[1][1]))
    # The prefix is already the collapsed output (no blanks, no CTC duplicates).
    # encoder.decode() would wrongly remove consecutive identical tokens like
    # "लल" in "हल्ला", so we join tokens directly instead.
    return "".join(encoder.id_to_token[i] for i in best[0])


def beam_decode(log_probs, encoder, beam_width: int, input_lengths=None, top_k: int = 40):
    """Batch wrapper around ctc_beam_search_single."""
    # log_probs: (T, B, C)
    lp_batchfirst = log_probs.permute(1, 0, 2)  # (B, T, C)
    results = []
    for i, seq in enumerate(lp_batchfirst):
        length = int(input_lengths[i]) if input_lengths is not None else seq.shape[0]
        results.append(
            ctc_beam_search_single(seq[:length], encoder, beam_width, top_k)
        )
    return results


# ---------------------------------------------------------------------------
# Lexicon snapping
# ---------------------------------------------------------------------------

def _token_edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token lists."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (a[i - 1] != b[j - 1]),
            )
        prev = curr
    return prev[n]


def build_lexicon(vocab_path: str, encoder: CTCTextEncoder):
    """
    Load the word-level vocab and return a list of (word, token_list) pairs.
    These are the only valid output words for lexicon-constrained decoding.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # vocab.json may contain either character tokens or whole words depending
    # on how prepare_data.py was run. We want the word list from the manifests.
    # If entries are single tokens, there's no useful lexicon; return None.
    words = [w for w in raw if len(encoder.split_text(w)) > 1]
    if not words:
        return None  # vocab is character-level; snapping not applicable

    return [(w, encoder.split_text(w)) for w in words]


def snap_to_lexicon(prediction: str, lexicon, encoder: CTCTextEncoder, max_dist: int = 2):
    """
    Replace prediction with the closest lexicon word if within max_dist token edits.
    Returns the (possibly snapped) word and the edit distance.
    """
    if lexicon is None:
        return prediction, 0

    pred_tokens = encoder.split_text(prediction) if prediction else []
    best_word, best_dist = prediction, max_dist + 1

    for word, word_tokens in lexicon:
        d = _token_edit_distance(pred_tokens, word_tokens)
        if d < best_dist:
            best_dist = d
            best_word = word
        if d == 0:
            break  # exact match

    return (best_word if best_dist <= max_dist else prediction), best_dist


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_crnn.pt")
    p.add_argument("--test-manifest", default="data/manifests/test.tsv")
    p.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="CTC beam width. 1 = greedy (fast), 10–20 = beam search (more accurate).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Only extend beam with top-k tokens per timestep (speed/accuracy trade-off).",
    )
    p.add_argument(
        "--snap-lexicon",
        action="store_true",
        help="Snap each prediction to the nearest word in vocab.json by edit distance.",
    )
    p.add_argument("--vocab-path", default="data/manifests/vocab.json")
    p.add_argument("--max-snap-dist", type=int, default=2,
        help="Maximum token edit distance allowed for lexicon snapping.")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder = CTCTextEncoder.load(ckpt["vocab_path"])

    test_ds = HindiWordDataset(args.test_manifest, encoder)
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=ctc_collate,
    )

    model = CRNN(len(encoder)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    lexicon = None
    if args.snap_lexicon:
        lexicon = build_lexicon(args.vocab_path, encoder)
        if lexicon is None:
            print("WARNING: vocab.json appears character-level; lexicon snapping disabled.")
        else:
            print(f"lexicon loaded: {len(lexicon)} words", flush=True)

    use_beam = args.beam_width > 1
    print(
        f"decoding: {'beam search (width=' + str(args.beam_width) + ')' if use_beam else 'greedy'} | "
        f"lexicon snap: {'on (max_dist=' + str(args.max_snap_dist) + ')' if args.snap_lexicon and lexicon else 'off'}",
        flush=True,
    )

    correct = 0
    snapped = 0
    total = 0

    with torch.no_grad():
        for images, _, input_lengths, _, texts in test_dl:
            images = images.to(device)
            log_probs = model(images).cpu()

            if use_beam:
                preds = beam_decode(log_probs, encoder, args.beam_width,
                                    input_lengths, args.top_k)
            else:
                preds = greedy_decode(log_probs, encoder, input_lengths)

            if args.snap_lexicon and lexicon:
                snapped_preds = []
                for p in preds:
                    snapped_p, dist = snap_to_lexicon(p, lexicon, encoder, args.max_snap_dist)
                    if snapped_p != p:
                        snapped += 1
                    snapped_preds.append(snapped_p)
                preds = snapped_preds

            correct += sum(p == t for p, t in zip(preds, texts))
            total += len(texts)

    print(f"test exact match: {correct / max(total, 1):.4f}  ({correct}/{total})")
    if args.snap_lexicon and lexicon:
        print(f"predictions snapped to lexicon: {snapped}/{total} ({100*snapped/max(total,1):.1f}%)")


if __name__ == "__main__":
    main()