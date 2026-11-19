from pathlib import Path
import random

from PIL import Image
from PIL import ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from tokenizer import CTCTextEncoder


class HindiWordDataset(Dataset):
    def __init__(self, manifest_path: str, encoder: CTCTextEncoder, img_h=32, max_w=384, augment=False):
        self.encoder = encoder
        self.img_h = img_h
        self.max_w = max_w
        self.augment = augment
        self.samples = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                image_path, text = line.rstrip("\n").split("\t", 1)
                self.samples.append((Path(image_path), encoder.normalize(text)))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path: Path):
        img = Image.open(image_path).convert("L")
        w, h = img.size
        new_w = max(8, min(self.max_w, round(w * self.img_h / h)))
        img = img.resize((new_w, self.img_h))
        img = ImageOps.autocontrast(img)
        x = 1.0 - TF.to_tensor(img)  # black text ~= 1.0, white bg ~= 0.0
        return x

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a (1, H, W) float tensor."""
        # Random small rotation: keeps text legible, simulates tilt variation
        if random.random() < 0.5:
            angle = random.uniform(-4.0, 4.0)
            image = TF.rotate(image, angle, fill=0.0)

        # Random brightness scale: simulates ink density variation
        if random.random() < 0.5:
            factor = random.uniform(0.75, 1.25)
            image = torch.clamp(image * factor, 0.0, 1.0)

        # Random Gaussian noise: simulates paper texture and scan artifacts
        if random.random() < 0.4:
            noise = torch.randn_like(image) * 0.04
            image = torch.clamp(image + noise, 0.0, 1.0)

        # Random width scaling (±10%): simulates writing width variation.
        # Resizes then re-clamps to max_w so CTC lengths stay valid.
        if random.random() < 0.4:
            _, h, w = image.shape
            scale = random.uniform(0.9, 1.1)
            new_w = max(8, min(self.max_w, round(w * scale)))
            image = TF.resize(image, [h, new_w], antialias=True)

        return image

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        image = self._load_image(image_path)
        if self.augment:
            image = self._augment_image(image)
        target = torch.tensor(self.encoder.encode(text), dtype=torch.long)
        return image, target, text


def ctc_collate(batch):
    images, targets, texts = zip(*batch)

    max_w = max(img.shape[-1] for img in images)
    padded_images = []
    input_lengths = []

    for img in images:
        pad_w = max_w - img.shape[-1]
        padded_images.append(F.pad(img, (0, pad_w), value=0.0))  # background
        input_lengths.append(max(1, img.shape[-1] // 4 + 1))

    flat_targets = torch.cat(targets)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    return (
        torch.stack(padded_images),
        flat_targets,
        torch.tensor(input_lengths, dtype=torch.long),
        target_lengths,
        list(texts),
    )