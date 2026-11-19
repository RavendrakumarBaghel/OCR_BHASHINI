from torchvision.transforms.functional import to_pil_image
from dataset import HindiWordDataset
from tokenizer import CTCTextEncoder

encoder = CTCTextEncoder.load("data/manifests/vocab.json")
ds = HindiWordDataset("data/manifests/train.tsv", encoder, max_w=384)

img, target, text = ds[0]
print("text:", text)
print("encoded:", target.tolist())
print("shape:", img.shape, "min:", img.min().item(), "max:", img.max().item())

to_pil_image(img).save("debug_sample_0.png")
print("saved debug_sample_0.png")
