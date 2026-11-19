from pathlib import Path
import json
import unicodedata

from tokenizer import CTCTextEncoder
DATASET_ROOT = Path("data/raw/IIIT-HW-Hindi_v1")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def parse_split_file(split_file: Path):
    samples = []

    if not split_file.exists():
        return samples

    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rel_path, label = line.split(maxsplit=1)
            image_path = DATASET_ROOT / rel_path

            if not image_path.exists():
                print(f"Missing file: {image_path}")
                continue

            samples.append((str(image_path.resolve()), normalize_text(label)))

    return samples


def load_vocab_words(vocab_path: Path):
    with vocab_path.open("r", encoding="utf-8") as f:
        return [normalize_text(line) for line in f if line.strip()]


def parse_page_annotations(split: str, vocab_words):
    samples = []
    split_root = DATASET_ROOT / "HindiSeg" / split

    if not split_root.exists():
        return samples

    for annotation_path in sorted(split_root.glob("*/*/*.txt")):
        with annotation_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                vocab_id = int(line)
                if vocab_id < 1 or vocab_id > len(vocab_words):
                    raise ValueError(
                        f"Invalid vocab id {vocab_id} in {annotation_path}:{line_number}"
                    )

                image_path = annotation_path.with_name(f"{line_number}.jpg")
                if not image_path.exists():
                    print(f"Missing file: {image_path}")
                    continue

                samples.append((str(image_path.resolve()), vocab_words[vocab_id - 1]))

    return samples


def load_split(split: str, vocab_words):
    samples = parse_split_file(DATASET_ROOT / f"{split}.txt")
    if samples:
        return samples

    print(f"{split}.txt is empty; rebuilding from HindiSeg/{split} annotations")
    return parse_page_annotations(split, vocab_words)


def write_manifest(samples, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for image_path, text in samples:
            f.write(f"{image_path}\t{text}\n")


def build_vocab(*sample_lists):
    tokens = set()
    for samples in sample_lists:
        for _, text in samples:
            tokens.update(CTCTextEncoder.split_text(text))
    return sorted(tokens)


def main():
    vocab_words = load_vocab_words(DATASET_ROOT / "hindi_vocab.txt")
    train_samples = load_split("train", vocab_words)
    val_samples = load_split("val", vocab_words)
    test_samples = load_split("test", vocab_words)

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(
            "One or more splits are empty. Check data/raw/IIIT-HW-Hindi_v1 "
            "and the HindiSeg annotation folders before training."
        )

    write_manifest(train_samples, Path("data/manifests/train.tsv"))
    write_manifest(val_samples, Path("data/manifests/val.tsv"))
    write_manifest(test_samples, Path("data/manifests/test.tsv"))

    vocab = build_vocab(train_samples, val_samples, test_samples)
    with open("data/manifests/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("train:", len(train_samples))
    print("val:", len(val_samples))
    print("test:", len(test_samples))
    print("vocab size:", len(vocab))


if __name__ == "__main__":
    main()
