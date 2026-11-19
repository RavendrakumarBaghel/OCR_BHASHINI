from argparse import ArgumentParser
from dataset import HindiWordDataset
from tokenizer import CTCTextEncoder

parser = ArgumentParser()
parser.add_argument("--manifest", default="data/manifests/train.tsv")
parser.add_argument("--vocab-path", default="data/manifests/vocab.json")
parser.add_argument("--max-samples", type=int, default=1000)
args = parser.parse_args()

encoder = CTCTextEncoder.load(args.vocab_path)
ds = HindiWordDataset(args.manifest, encoder, img_h=32, max_w=256)

bad = 0
total = 0

limit = min(args.max_samples, len(ds)) if args.max_samples > 0 else len(ds)

for i in range(limit):
    image, target, text = ds[i]
    input_len = max(1, image.shape[-1] // 4 + 1)
    target_len = len(target)
    total += 1
    if target_len > input_len:
        bad += 1

print("checked:", total, "of", len(ds))
print("invalid_ctc_samples:", bad)
print("invalid_ratio:", bad / total)
