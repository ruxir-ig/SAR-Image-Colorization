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

**Advanced Training Options:**

```bash
# Train with custom parameters
python train.py --epochs 100 --batch_size 16 --learning_rate 0.0002

# Resume training from checkpoint
python train.py --resume models/generator_checkpoint.pt

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

The training script will:
- Load paired SAR and RGB images from the dataset directory
- Initialize the generator model
- Train using the specified hyperparameters
- Save model checkpoints and training logs to `models/` directory
- Log training metrics to TensorBoard in `runs/` directory

**Monitoring Training Progress:**
- Loss curves and sample outputs are logged every 100 iterations
- Model checkpoints are saved every 10 epochs
- Best model is automatically saved based on validation loss

### Inference

To colorize new SAR images:

```bash
python predict.py
```

**Detailed Inference Examples:**

```bash
# Colorize a single image
python predict.py --input path/to/sar_image.png --output colorized_output.jpg

# Batch process multiple images
python predict.py --input_dir sar_images/ --output_dir colorized_results/

# Use specific model checkpoint
python predict.py --model models/custom_generator.pt --input image.png

# Adjust output quality and size
python predict.py --input image.png --output result.jpg --quality 95 --resize 1024
```

**Programmatic Usage:**

```python
from model import SARModel
from predict import colorize_image
import torch

# Load pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SARModel(device)
model.load_state_dict(torch.load('models/generator.pt'))

# Colorize single image
colorized = colorize_image(model, 'path/to/sar_image.png', device)
```

### Using the Jupyter Notebook

For interactive experimentation and model visualization:

```bash
jupyter notebook base.ipynb
```

The notebook includes:
- **Model Architecture Visualization**: View network structure and parameters
- **Data Exploration**: Analyze dataset statistics and sample pairs
- **Training Visualization**: Real-time loss plots and sample outputs
- **Interactive Inference**: Test the model on custom inputs
- **Result Analysis**: Compare original SAR images with colorized outputs

**Notebook Sections:**
1. **Setup & Dependencies**: Environment configuration and imports
2. **Data Loading**: Dataset exploration and preprocessing
3. **Model Definition**: Architecture details and summary
4. **Training Loop**: Interactive training with live updates
5. **Evaluation**: Model performance metrics and visual results
6. **Custom Inference**: Test on your own SAR images

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

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size in training
python train.py --batch_size 4

# Use CPU for inference if GPU memory is limited
python predict.py --device cpu
```

**Poor Colorization Results:**
- Ensure SAR and RGB images are properly paired in the dataset
- Check that input images are preprocessed correctly (normalized, resized)
- Try training for more epochs or adjusting learning rate
- Verify model checkpoint is loading correctly

**Training Not Converging:**
- Adjust learning rate (try 0.0001 or 0.0005)
- Check data quality and alignment
- Monitor discriminator/generator loss balance
- Ensure sufficient dataset size (minimum 1000+ paired images recommended)

### Performance Tips

- Use mixed precision training for faster training: `--amp`
- Enable data loading optimization: `--num_workers 4`
- Use gradient accumulation for larger effective batch sizes
- Implement learning rate scheduling for better convergence

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Development Guidelines:**
- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include unit tests for new features
- Update documentation for API changes
- Test with both CPU and GPU environments

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