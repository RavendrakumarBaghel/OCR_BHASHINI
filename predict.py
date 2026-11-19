import sys

sys.dont_write_bytecode = True

import torch

from dataset import HindiWordDataset
from evaluate import beam_decode, greedy_decode
from model import CRNN
from tokenizer import CTCTextEncoder


def main():
    image_path = sys.argv[1]
    beam_width = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    ckpt = torch.load("checkpoints/best_crnn.pt", map_location="cpu")
    encoder = CTCTextEncoder.load(ckpt["vocab_path"])

    model = CRNN(len(encoder))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    helper_ds = HindiWordDataset("data/manifests/val.tsv", encoder)
    image = helper_ds._load_image(image_path).unsqueeze(0)

    with torch.no_grad():
        log_probs = model(image)
        if beam_width > 1:
            result = beam_decode(log_probs.cpu(), encoder, beam_width)[0]
        else:
            pred_ids = log_probs.argmax(2).squeeze(1).tolist()
            result = encoder.decode(pred_ids)

    print(result)


if __name__ == "__main__":
    main()