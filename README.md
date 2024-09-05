# Face Mask Detection 

This project demonstrates a binary classification problem, where a deep learning model is trained to classify images of people wearing face masks or not wearing face masks. Two models are used: a Convolutional Neural Network (CNN) and a VGG19 transfer learning model.

## Dataset

The dataset used for this project consists of images categorized into two classes:
1. **Train Set**: 10,000 images
2. **Validation Set**: 800 images
3. **Test Set**: 992 images

Each image is resized to 128x128 pixels.

## Model Architecture

### 1. CNN Model
- **Layers**: 
  - 4 convolutional layers
  - Max-pooling layers after each convolutional layer
  - Dense layers with ReLU and Sigmoid activation functions
- **Activation Functions**: 
  - ReLU in hidden layers
  - Sigmoid for binary classification output

### 2. VGG19 (Transfer Learning)
- Pretrained VGG19 model (without the top layers) is used as a feature extractor.
- Additional layers:
  - Flatten
  - Dense layers with ReLU and Sigmoid activations
- **Note**: The VGG19 base model’s layers are frozen during training.

## Data Augmentation

The dataset is augmented using the following techniques:
- Rotation
- Width and height shift
- Shear
- Zoom
- Horizontal flip

## Model Training

Both models were compiled using the following configurations:
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: RMSprop with different learning rates (1e-4 for CNN, 2e-5 for VGG19)
- **Metrics**: Accuracy

## Results

- **CNN Model**: Achieved 98.37% accuracy on the test set.
- **VGG19 Model**: Achieved 99.39% accuracy on the test set.

## Visualizations

Training and validation accuracy/loss were plotted using Matplotlib after model training to monitor performance.

## Model Evaluation

Both models were evaluated on the test set, with the following results:
- **CNN Model**: 98.37% accuracy
- **VGG19 Model**: 99.39% accuracy
