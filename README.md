# BactUnet

BactUnet is a deep learning project designed for segmenting *Salmonella enterica* serovar Typhimurium. It works with complex time-lapse microscopy images of primary human epithelial cells. It utilizes a U-Net-based architecture combined with advanced preprocessing techniques, including patching, normalization, and window functions, allowing for accurate segmentation in challenging imaging conditions. The project is highly modular, with several standalone preprocessing steps including synthetic image generation, patchification, normalization, and window function application.

## Features
- **Segmentation of Bacteria**: Focus on segmenting bacteria interacting with human epithelial cells.
- **Advanced Imaging Techniques**: Uses Differential Interference Contrast (DIC) imaging and more.
- **Flexible Preprocessing**: Modular preprocessing including patching, normalization, and windowing.
- **Synthetic Image Generation**: Ability to generate synthetic training images for testing and training purposes.

## Directory Structure
- `src/` : Contains the core source code, including model definitions and preprocessing utilities.
- `data/` : Datasets used for training and validation.
- `notebooks/` : Jupyter notebooks for experiments and visualization.
- `models/` : Trained model checkpoints and architecture files.
- `results/` : Segmented outputs and metrics for evaluation.
- `docs/` : Documentation and usage guidelines.

## Prerequisites
- Python 3.8+
- TensorFlow 2.x
- GPU with CUDA support (recommended)

### Required Python Packages
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Key Components

### 1. Synthetic Image Creation
The script `simple_unet_model.py` provides functionality to create synthetic 2D images featuring concentric circle patterns. These synthetic images can be used to evaluate the robustness of the preprocessing and segmentation models.

```python
from simple_unet_model import create_synthetic_image

img = create_synthetic_image(size=2304)
```

### 2. Preprocessing Pipeline
The preprocessing pipeline includes patching (dividing large images into smaller regions), normalization (adjusting pixel intensity values to a consistent range), and applying window functions (reducing edge artifacts) to images to prepare them for model training. The preprocessing functions are modular and well-documented.

- **Patch and Normalize**: Breaks large images into smaller patches and normalizes them based on specified percentiles.

```python
from simple_unet_model import patch_and_normalize_image

normalized_img = patch_and_normalize_image(img, patch_size=288, pmin=0.1, pmax=99.9)
```

- **Window Functions**: Apply window functions like Hanning, Hamming, Blackman, etc., to reduce edge artifacts in patches.

```python
from simple_unet_model import apply_window_function

windowed_data = apply_window_function(normalized_img, window_type='hanning')
```

### 3. Model Building
The segmentation model is built using U-Net architecture, with convolutional and max-pooling blocks. The model is modular, allowing for easy adjustments or the addition of custom layers.

```python
from simple_unet_model import build_unet

input_shape = (1, 288, 288)
model = build_unet(input_shape)
model.summary()
```

### 4. Training
Training involves a hybrid loss function combining Focal Tversky Loss and Intersection over Union (IoU). These specific loss functions were chosen to effectively handle class imbalance and improve segmentation accuracy, especially in challenging imaging conditions. Training and validation data are provided through custom data generators, which include image augmentations such as rotation, zoom, and flip.

```python
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

model.compile(loss=hybrid_loss, optimizer='adam', metrics=['accuracy'])
model.fit(my_generator, steps_per_epoch=steps_per_epoch, epochs=100, validation_data=validation_datagen, callbacks=callbacks_list)
```

## Usage

### Run Preprocessing
To run the preprocessing steps for patching and normalizing images:

```python
python simple_unet_model.py
```

### Train the Model
To train the segmentation model, adjust the parameters in the training script and run:

```bash
python train_model.py
```

## Contributing
We welcome contributions! Please refer to `CONTRIBUTING.md` for detailed contribution guidelines. To set up the development environment, make sure to install the required dependencies listed in `requirements.txt` and follow the setup instructions in `docs/setup.md`.

To contribute:
1. Fork the repository and create a new branch for your feature or bug fix.
2. Write clear and descriptive commit messages.
3. Open a pull request describing the changes.

## License
This project is licensed under the MIT License. Please see `LICENSE` for more information.



