# Hindi OCR: CRNN + CTC

A deep-learning OCR model for recognizing printed Hindi (Devanagari) text lines, built with a **CRNN (Convolutional Recurrent Neural Network)** trained using **CTC (Connectionist Temporal Classification)** loss.

## Overview

| Item | Details |
|------|---------|
| Task | Line-level OCR for printed Hindi text |
| Architecture | CNN feature extractor, recurrent (BiLSTM/GRU) layers, CTC decoding |
| Input | Grayscale text-line image, 900 x 64 px |
| Output | Devanagari text (including spaces) |
| Training data | ~80,000 synthetic Hindi text-line images |
| Framework | PyTorch |

## How it works

1. **CNN** extracts visual features from the text-line image.
2. **RNN** models the left-to-right sequence of those features.
3. **CTC loss** aligns the predicted sequence with the target text, so no per-character annotation is needed.
4. **CTC decoding** (greedy or beam search) converts the output into the final string.

## Project structure

```
hindi-crnn-ctc/
├── notebooks/         # Training and inference notebooks
├── metrics_epoch_*.txt  # Per-epoch evaluation metrics
├── requirements.txt   # Python dependencies
└── README.md
```

> Adjust the tree above to match the actual files in your copy of the repository.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/RavendrakumarBaghel/hindi-crnn-ctc.git
cd hindi-crnn-ctc

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Inference

1. Place your trained weights (`.pt` / `.pth`) in the project folder.
2. Open the inference notebook and set the path to your weights and image.
3. Run all cells to get the predicted Hindi text.

Preprocessing expected by the model:

- Convert the image to grayscale
- Resize to **900 x 64**
- Normalize pixel values the same way as during training

### Training

1. Prepare line images and their text labels.
2. Open the training notebook and set dataset paths and hyperparameters.
3. Run training; metrics are saved per epoch in `metrics_epoch_*.txt`.

## Evaluation

Performance is tracked with:

- **CER** (Character Error Rate)
- **WER** (Word Error Rate)

See the `metrics_epoch_*.txt` files for results across epochs.

## Limitations

- Trained on **synthetic printed text**, so accuracy may drop on handwriting, noisy scans, or unusual fonts.
- Works on **single text lines**, not full pages. Use a line-segmentation step for documents.

## Roadmap

- [ ] Add real-world scanned data for fine-tuning
- [ ] Add a command-line inference script
- [ ] Add line detection for full-page OCR
- [ ] Publish a demo app

## Acknowledgements

- CRNN: Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*
- CTC: Graves et al., *Connectionist Temporal Classification*


