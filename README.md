# Speech Recognition Model

This repository contains a speech recognition model implementation using PyTorch. The model uses CTC (Connectionist Temporal Classification) loss for training and beam search decoding for inference.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── model.py           # Model architecture
├── dataset.py         # Data loading and preprocessing
├── train.py          # Training script
├── utils.py          # Utility functions
└── train.ipynb       # Training notebook
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install ctcdecode:
```bash
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..
```

## Training

You can train the model either using the Python script or the Jupyter notebook:

### Using Python script:
```bash
python train.py
```

### Using Jupyter notebook:
1. Open `train.ipynb`
2. Adjust the configuration parameters if needed
3. Run all cells

## Model Architecture

The model uses a bidirectional LSTM architecture with the following components:
- Multiple LSTM layers
- Dropout for regularization
- Linear projection layer
- CTC loss for training
- Beam search decoding for inference

## Configuration

The main configuration parameters are:
- `input_dim`: Input feature dimension
- `hidden_dim`: LSTM hidden dimension
- `num_layers`: Number of LSTM layers
- `num_classes`: Number of output classes
- `dropout`: Dropout rate
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `epochs`: Number of training epochs
- `patience`: Patience for learning rate scheduler
- `beam_width`: Beam width for decoding

## Data Format

The model expects:
- Features in `.npy` format
- CSV files with training/validation data information
- Directory structure:
  ```
  data/
  ├── features/
  │   ├── sample1.npy
  │   ├── sample2.npy
  │   └── ...
  ├── train.csv
  └── val.csv
  ```
