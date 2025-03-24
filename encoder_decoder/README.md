# English-German Neural Machine Translation

This project implements a Transformer-based neural machine translation (NMT) system for English to German translation. The implementation follows the architecture described in the paper "Attention Is All You Need" by Vaswani et al.

## Project Structure

```
encoder_decoder/
├── data/                      # Data directory
│   ├── Customer-sample-English-German-Training-en.txt
│   └── Customer-sample-English-German-Training-de.txt
├── dataset.py                 # Dataset and data loading utilities
├── encoder_decoder.py         # Transformer model implementation
├── train.py                   # Training script
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Features

- Implementation of the Transformer architecture
- Multi-head self-attention mechanism
- Positional encoding
- Layer normalization and residual connections
- English to German translation
- BLEU score evaluation
- Progress tracking and checkpointing

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required spaCy language models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Usage

### Training

To train the model:
```bash
python train.py
```

The training script includes:
- Automatic device selection (CPU/MPS/CUDA)
- Progress tracking
- Checkpoint saving
- Error handling and recovery

### Model Architecture

The model uses the following configuration:
- Embedding dimension: 128
- Number of attention heads: 2
- Number of encoder/decoder layers: 1
- Feed-forward dimension: 128
- Dropout rate: 0.1
- Batch size: 2

## Dependencies

- PyTorch 2.7.0
- spaCy 3.8.4
- NumPy 2.2.3
- tqdm 4.66.2
- sentencepiece 0.2.0
- sacrebleu 2.5.1
- torchtext 0.16.0

## Notes

- The model is configured for CPU training by default for stability
- Checkpoints are saved after each epoch
- The training script includes error handling and recovery mechanisms
- The model uses a reduced size configuration for faster training and testing

## License

This project is open source and available under the MIT License. 