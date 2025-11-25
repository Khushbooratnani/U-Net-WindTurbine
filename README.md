# Windmill Detection using U-Net Segmentation
A deep learning project for detecting windmills in aerial images using a U-Net-based semantic segmentation model.

## U-Net Architecture
![U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)



## Features
- U-Net architecture with skip connections
- COCO JSON to segmentation mask conversion
- TensorFlow/Keras implementation
- Binary segmentation (windmill vs background)
- 640x640 resolution support

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- Jupyter Notebook

## Installation
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow jupyter
```
## Dataset Preparation
1. Organize your dataset:
```bash
dataset/
├── image/        # Original images
└── masks/        # Generated segmentation masks
```
2. Convert COCO JSON annotations to segmentation masks:
```python
generate_segmentation_masks(
    json_file='path/to/annotations.json',
    output_dir='dataset/masks/',
    image_size=(640, 640)
)
```
## Model Architecture
### U-Net Configuration:

- Input: 640x640 RGB images
- Encoder Depth: 4 Down-Sampling blocks
- Decoder Depth: 4 Up-Sampling blocks with skip connections
- Base Filters: 64 (doubling at each level)
- Total Parameters: 31.4 million
- Output: 640x640 mask with 2 channels (background/windmill)

## Training
1. Load and preprocess data:
```python
images, masks = load_dataset(image_dir='dataset/image', 
                            mask_dir='dataset/masks')
```
2. Train the model:
```bash
Epochs: 10
Batch Size: 16
Optimizer: Adam
Loss: Binary Crossentropy
Validation Split: 20%
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/sawaipratap/windmill-detection-unet.git
```
2. Start Jupyter Notebook:
```bash
jupyter notebook Main.ipynb
```
3. Modify paths in the notebook:
```python
image_fol = 'path/to/your/images'
mask_fol = 'path/to/your/masks'
```

## Results
Expected outputs after successful training:
- Training/validation accuracy curves
- IoU (Intersection over Union) metrics
- Example predictions showing:
  - Original image
  - Ground truth mask
  - Predicted mask

## Troubleshooting

Common issues:
- **CUDA Out of Memory**: Reduce batch size
- **Mask Dimension Errors**: Ensure mask generation matches image size
- **COCO Parsing Errors**: Verify annotation format

## Acknowledgments
- U-Net original paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- COCO Dataset format
- TensorFlow/Keras documentation
