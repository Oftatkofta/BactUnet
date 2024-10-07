# BactUnet

BactUnet is a deep learning project for segmenting *Salmonella enterica* serovar Typhimurium in complex time-lapse microscopy images of primary human epithelial cells. Using advanced neural network architectures, such as variations of U-nets, BactUnet aims to provide accurate segmentation of bacteria in challenging imaging conditions, enabling in-depth analysis of host-pathogen interactions.

## Features
- Segmentation of bacteria in time-lapse microscopy images
- Focus on wild-type *Salmonella* interacting with primary human epithelial cells
- Utilizes advanced imaging techniques like Differential Interference Contrast (DIC)

## Getting Started

### Prerequisites
- Python 3.8+
- GPU with CUDA support (optional but recommended)

Install the required Python packages:

```bash
pip install torch torchvision numpy scipy matplotlib opencv-python
```

### Directory Structure
- `src/` : Contains the core source code (e.g., data loading, model definition, training scripts)
- `data/` : Datasets (note: datasets are not version-controlled)
- `notebooks/` : Jupyter notebooks for prototyping and visualization
- `docs/` : Documentation and tutorials
- `models/` : Trained model checkpoints
- `results/` : Example outputs and analysis results

### Usage

1. **Training the Model**

   Run the training script:
   ```bash
   python src/train.py --config config.yaml
   ```

   You can modify hyperparameters and settings in `config.yaml`. These include parameters such as:
   - **learning_rate**: Controls how quickly the model updates weights; adjusting this can impact convergence speed.
   - **batch_size**: Determines the number of samples per gradient update; larger sizes can lead to more stable training but require more memory.
   - **num_epochs**: Sets the number of complete passes through the training dataset; more epochs can improve accuracy but may lead to overfitting.
   - **data_augmentation**: Configures on-the-fly data augmentation, which helps improve model generalization.

2. **Inference**

   Run inference on a set of images:
   ```bash
   python src/infer.py --input data/sample_images --output results/segmented_images
   ```
   
   The input images should be in a standard image format (e.g., `.png`, `.jpg`) and organized in a directory. Each image should have sufficient resolution to capture relevant details of the bacterial and host cell interactions.

### Example Results
Example segmentations can be found in the `results/` directory, showcasing the effectiveness of BactUnet in identifying bacterial regions within epithelial cells.

## Contributing
Contributions are welcome! Please check `CONTRIBUTING.md` for guidelines.

To contribute:
1. **Fork** the repository to your GitHub account.
2. **Clone** the forked repository to your local machine.
3. Create a new **branch** for your feature or bug fix.
4. **Commit** your changes with clear and descriptive messages.
5. **Push** the branch to your GitHub repository.
6. Open a **pull request** to the main repository and describe your changes in detail.

We encourage contributions in areas like improving model accuracy, adding new features, refining documentation, and addressing bugs.

## License
This project is licensed under the MIT License. This license allows for reuse, modification, and distribution of the software, provided that proper credit is given to the original authors. See `LICENSE` for more details.

## Acknowledgements
We also thank the members of our research group for their contributions to dataset preparation and model evaluation.

## Contact
For questions, please contact [Your Name] at [your.email@example.com].