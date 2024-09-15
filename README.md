# SAR Image Colorization

A deep learning project for colorizing Synthetic Aperture Radar (SAR) images using PyTorch and Generative Adversarial Networks (GANs).

## Overview

This project implements a neural network model to automatically colorize grayscale SAR images, transforming them into realistic RGB representations. The model uses a generator-discriminator architecture trained on paired SAR and optical image datasets.

## Features

- **Deep Learning Model**: Custom generator model for SAR image colorization
- **PyTorch Implementation**: Built using PyTorch framework with CUDA support
- **Data Pipeline**: Efficient data loading and preprocessing for paired image datasets
- **Training Monitoring**: TensorBoard integration for training visualization
- **Inference Pipeline**: Easy-to-use prediction interface for new SAR images
- **Pre-trained Models**: Includes trained model weights for immediate use

## Project Structure

```
Sar_Colorization/
├── SAR_Image_Colorization_Pairs/    # Dataset directory
│   ├── train/                       # Training data pairs
│   └── test/                        # Testing data pairs
├── models/                          # Saved model weights
│   └── generator.pt                 # Pre-trained generator model
├── runs/                           # TensorBoard logs
├── __pycache__/                    # Python cache files
├── base.ipynb                      # Jupyter notebook for experimentation
├── data.py                         # Data loading and preprocessing
├── model.py                        # Neural network model definitions
├── train.py                        # Training script
├── predict.py                      # Inference script
├── utils.py                        # Utility functions
├── requirements.txt                # Python dependencies
├── tile_*.jpg                      # Sample output images
└── README.md                       # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Sar_Colorization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU setup** (optional but recommended):
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

## Dependencies

- PyTorch (with CUDA support recommended)
- NumPy
- scikit-image
- matplotlib
- TensorBoard
- torchsummary
- tqdm

## Usage

### Training

To train the model on your dataset:

```bash
python train.py
```

The training script will:
- Load paired SAR and RGB images from the dataset directory
- Initialize the generator model
- Train using the specified hyperparameters
- Save model checkpoints and training logs

### Inference

To colorize new SAR images:

```bash
python predict.py
```

### Using the Jupyter Notebook

For interactive experimentation and model visualization:

```bash
jupyter notebook base.ipynb
```

## Model Architecture

The project implements a custom generator model with the following key components:

- **Encoder**: Convolutional layers with LeakyReLU activation and BatchNorm
- **Decoder**: Transpose convolutional layers for upsampling
- **Skip Connections**: U-Net style architecture for detail preservation
- **Output**: 3-channel RGB image generation

## Dataset

The model expects paired datasets with:
- **Input**: Grayscale SAR images (single channel)
- **Target**: Corresponding RGB optical images (3 channels)

Dataset structure:
```
SAR_Image_Colorization_Pairs/
├── train/
│   ├── sar_image_1.png
│   ├── rgb_image_1.png
│   └── ...
└── test/
    ├── sar_image_1.png
    ├── rgb_image_1.png
    └── ...
```

## Results

The trained model generates realistic colorized versions of SAR images. Sample results are saved as `tile_*.jpg` files in the project directory.

## Monitoring Training

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir=runs
```

## GPU Requirements

- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum**: 4GB VRAM for training
- **Optimal**: 8GB+ VRAM for larger batch sizes

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built with PyTorch and the deep learning community's contributions
- Inspired by advances in image-to-image translation and GAN architectures
- Dataset preprocessing utilities adapted from scikit-image

## Contact

For questions, issues, or collaborations, please open an issue in the GitHub repository.

---

**Note**: This project was developed as part of research into SAR image processing and deep learning applications in remote sensing.