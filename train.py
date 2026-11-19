import argparse
from pathlib import Path
import sys
from collections import Counter

sys.dont_write_bytecode = True

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HindiWordDataset, ctc_collate
from model import CRNN
from tokenizer import CTCTextEncoder


def greedy_decode(log_probs, encoder, input_lengths=None):
    pred_ids = log_probs.argmax(2).permute(1, 0)
    if input_lengths is None:
        return [encoder.decode(seq.tolist()) for seq in pred_ids]

    return [
        encoder.decode(seq[: int(length)].tolist())
        for seq, length in zip(pred_ids, input_lengths)
    ]


def evaluate_metrics(model, loader, encoder, device, show_samples=False):
    model.eval()
    exact_correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    shown = False
    pred_counter = Counter()

    with torch.no_grad():
        for images, _, input_lengths, _, texts in loader:
            images = images.to(device)
            preds = greedy_decode(model(images).cpu(), encoder, input_lengths)
            pred_counter.update(preds)

            if show_samples and not shown:
                print("\nSample predictions:", flush=True)
                for pred, truth in list(zip(preds, texts))[:10]:
                    print(f"pred='{pred}' | truth='{truth}'", flush=True)
                shown = True

            exact_correct += sum(pred == truth for pred, truth in zip(preds, texts))
            total += len(texts)

            for pred, truth in zip(preds, texts):
                char_correct += sum(p == t for p, t in zip(pred, truth))
                char_total += max(len(pred), len(truth))

    exact_match = exact_correct / max(total, 1)
    char_acc = char_correct / max(char_total, 1)
    return exact_match, char_acc, pred_counter


def compute_ctc_loss(criterion, log_probs, targets, input_lengths, target_lengths):
    if log_probs.device.type == "mps":
        return criterion(
            log_probs.cpu(),
            targets.cpu(),
            input_lengths.cpu(),
            target_lengths.cpu(),
        )
    return criterion(log_probs, targets, input_lengths, target_lengths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/manifests/train.tsv")
    parser.add_argument("--val-manifest", default="data/manifests/val.tsv")
    parser.add_argument("--vocab-path", default="data/manifests/vocab.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-w", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0,
        help="Cap validation set size per epoch (0 = all). Keeps eval fast on large datasets.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1,
        help="Run validation every N epochs instead of every epoch. "
             "Use --eval-every 5 to cut ~20%% of wall time on large datasets.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--disable-scheduler", action="store_true")
    parser.add_argument("--scheduler-patience", type=int, default=10,
        help="Epochs of no val improvement before LR is halved (default 10).")
    parser.add_argument("--min-scheduler-epoch", type=int, default=15,
        help="Do not reduce LR before this epoch.")
    parser.add_argument("--blank-bias", type=float, default=-2.0,
        help="Initial bias for the CTC blank logit. Negative suppresses blank at init.")
    parser.add_argument("--no-augment", action="store_true",
        help="Disable training-time data augmentation (on by default).")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
        help="Path to a checkpoint to resume training from (e.g. checkpoints/last_crnn.pt).")
    parser.add_argument("--overfit-samples", type=int, default=0,
        help="Use the first N train samples for both train and val as a debug run.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")


    encoder = CTCTextEncoder.load(args.vocab_path)
    if len(encoder) <= 1:
        raise RuntimeError(
            f"{args.vocab_path} has no non-blank tokens. Run prepare_data.py "
            "and make sure it reports non-zero split and vocab sizes."
        )

    use_augment = not args.no_augment and args.overfit_samples == 0
    train_ds = HindiWordDataset(args.train_manifest, encoder, max_w=args.max_w, augment=use_augment)
    val_ds = HindiWordDataset(args.val_manifest, encoder, max_w=args.max_w, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Train or validation manifest is empty. Run prepare_data.py and "
            "make sure it reports non-zero train/val counts."
        )

    if args.train_samples > 0:
        n = min(args.train_samples, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))

    if args.val_samples > 0:
        n = min(args.val_samples, len(val_ds))
        val_ds = Subset(val_ds, list(range(n)))

    if args.overfit_samples > 0:
        n = min(args.overfit_samples, len(train_ds))
        indices = list(range(n))
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        train_ds = Subset(base_ds, indices)
        val_ds = Subset(base_ds, indices)

    checkpoint_dir = Path(args.checkpoint_dir or "checkpoints")
    if args.checkpoint_dir is None and args.overfit_samples > 0:
        checkpoint_dir = checkpoint_dir / "overfit_debug"

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
    )

    model = CRNN(len(encoder)).to(device)

    with torch.no_grad():
        model.rnn[1].fc.bias.zero_()
        model.rnn[1].fc.bias[0] = args.blank_bias

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if not args.disable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=1e-6,
        )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_exact = ckpt.get("val_exact_match", -1.0)
        best_char = ckpt.get("val_char_acc", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch} "
              f"(val_exact={best_exact:.4f} val_char={best_char:.4f})", flush=True)
    else:
        with torch.no_grad():
            model.rnn[1].fc.bias.zero_()
            model.rnn[1].fc.bias[0] = args.blank_bias

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        best_exact = -1.0
        best_char = -1.0
    log_every = max(1, args.log_every)
    eval_every = max(1, args.eval_every)

    print(f"device: {device}", flush=True)
    if device.type == "mps":
        print("using CPU fallback for CTCLoss on MPS", flush=True)
    print(f"train samples: {len(train_ds)}", flush=True)
    print(f"val samples: {len(val_ds)}", flush=True)
    print(f"train batches per epoch: {len(train_dl)}", flush=True)
    print(f"learning rate: {args.lr}", flush=True)
    print(f"blank bias: {args.blank_bias}", flush=True)
    print(f"augmentation: {'off' if args.no_augment else 'on'}", flush=True)
    print(f"eval every: {eval_every} epoch(s)", flush=True)
    if not args.disable_scheduler:
        print(f"scheduler: patience={args.scheduler_patience}, "
              f"min_epoch={args.min_scheduler_epoch}", flush=True)
    if args.overfit_samples > 0:
        print(f"starting overfit debug run on {len(train_ds)} samples...", flush=True)

    import time

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_display = epoch + 1
        print(f"\nEpoch {epoch_display}/{start_epoch + args.epochs} started", flush=True)
        epoch_start = time.time()

        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets, input_lengths, target_lengths, _) in enumerate(train_dl, start=1):
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)

            loss = compute_ctc_loss(criterion, log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_every == 0 or batch_idx == len(train_dl):
                avg_so_far = total_loss / batch_idx
                print(
                    f"epoch={epoch_display} batch={batch_idx}/{len(train_dl)} "
                    f"batch_loss={loss.item():.4f} avg_loss={avg_so_far:.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(len(train_dl), 1)
        train_secs = time.time() - epoch_start

        run_val = (epoch_display % eval_every == 0) or (epoch_display == start_epoch + args.epochs)
        if run_val:
            val_start = time.time()
            val_acc, val_char_acc, pred_counter = evaluate_metrics(
                model, val_dl, encoder, device, show_samples=True
            )
            val_secs = time.time() - val_start
            if scheduler is not None and epoch_display >= args.min_scheduler_epoch:
                scheduler.step(val_char_acc)
        else:
            val_acc, val_char_acc, pred_counter = best_exact, best_char, None
            val_secs = 0.0

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch_display}/{start_epoch + args.epochs} completed | "
            f"train_loss={avg_loss:.4f} | val_exact_match={val_acc:.4f} | "
            f"val_char_acc={val_char_acc:.4f} | lr={current_lr:.6f} | "
            f"train={train_secs:.0f}s val={val_secs:.0f}s",
            flush=True,
        )

        if run_val and pred_counter is not None:
            print(f"top_val_predictions={pred_counter.most_common(5)}", flush=True)

        if run_val and (val_acc, val_char_acc) > (best_exact, best_char):
            best_exact = val_acc
            best_char = val_char_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_path": args.vocab_path,
                    "epoch": epoch_display,
                    "val_exact_match": val_acc,
                    "val_char_acc": val_char_acc,
                },
                checkpoint_dir / "best_crnn.pt",
            )
            print(
                "saved new best checkpoint with "
                f"val_exact_match={best_exact:.4f} val_char_acc={best_char:.4f}",
                flush=True,
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_path": args.vocab_path,
                "epoch": epoch_display,
                "val_exact_match": val_acc,
                "val_char_acc": val_char_acc,
            },
            checkpoint_dir / "last_crnn.pt",
        )

    print(
        f"\ntraining finished | best_val_exact_match={best_exact:.4f} "
        f"| best_val_char_acc={best_char:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()