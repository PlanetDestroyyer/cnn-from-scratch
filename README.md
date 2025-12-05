---

CNN Image Classification From Scratch

Using MNIST, Fashion-MNIST, and CIFAR-10

1. Overview

This project implements Convolutional Neural Networks (CNNs) from scratch (without using any pre-trained models) for three different image classification tasks:

MNIST – handwritten digit classification

Fashion-MNIST – clothing item classification

CIFAR-10 – object classification on RGB images


The goal is to:

Design custom CNN architectures for each dataset

Train them end-to-end

Evaluate their performance using multiple metrics

Study how model complexity and regularization strategies change with dataset difficulty


All models are implemented using TensorFlow/Keras, with evaluation done via accuracy, classification report, and confusion matrix.


---

2. Datasets Used

All datasets are loaded directly from tensorflow.keras.datasets.

2.1 MNIST

Type: Handwritten digits

Train images: 60,000

Test images: 10,000

Image size: 28 × 28 (grayscale)

Classes (10): digits 0–9


2.2 Fashion-MNIST

Type: Clothing items

Train images: 60,000

Test images: 10,000

Image size: 28 × 28 (grayscale)

Classes (10):
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot


2.3 CIFAR-10

Type: Natural RGB object images

Train images: 50,000

Test images: 10,000

Image size: 32 × 32 × 3 (RGB)

Classes (10):
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck


All images are normalized to the range [0, 1] before training.


---

3. Model Architectures (Layer Details)

3.1 MNIST – CNN Without Dropout / BatchNorm

Notebook: CNN_from_Scratch_on_MNIST_Data.ipynb

Architecture:

Conv2D(32, 3×3) → ReLU

Conv2D(32, 3×3) → ReLU

MaxPooling2D(2×2)

Conv2D(64, 3×3) → ReLU

Conv2D(64, 3×3) → ReLU

MaxPooling2D(2×2)

Flatten

Dense(128) → ReLU

Dense(10) → ReLU

Dense(64) → ReLU

Dense(10) → Softmax


Parameters:

Total params: 469,172

Trainable params: 469,172


This is a relatively simple CNN with two convolutional blocks followed by fully connected layers, and no explicit regularization (no Dropout / BatchNorm).


---

3.2 Fashion-MNIST – CNN With Dropout

Notebook: CNN_from_sractch_on_Fashion_MNIST_with_Dropout.ipynb

Architecture:

Conv2D(32, 3×3) → ReLU

Conv2D(32, 3×3) → ReLU

MaxPooling2D(2×2)

Conv2D(64, 3×3) → ReLU

Conv2D(64, 3×3) → ReLU

MaxPooling2D(2×2)

Flatten

Dropout

Dense(128) → ReLU

Dense(10) → Softmax


Parameters:

Total params: 467,818

Trainable params: 467,818


Compared to MNIST, this model introduces Dropout after Flatten to reduce overfitting on the more challenging Fashion-MNIST dataset.


---

3.3 CIFAR-10 – CNN With BatchNorm + Dropout + EarlyStopping

Notebook: CNN_from_Sractch_on_Cifar10_Data.ipynb

Architecture:

Conv2D(32, 3×3)

BatchNormalization

ReLU

Conv2D(32, 3×3)

BatchNormalization

ReLU

MaxPooling2D(2×2)

Conv2D(64, 3×3)

BatchNormalization

ReLU

Conv2D(64, 3×3)

BatchNormalization

ReLU

MaxPooling2D(2×2)

Conv2D(128, 3×3) → ReLU

Conv2D(128, 3×3) → ReLU

MaxPooling2D(2×2)

Flatten

Dense(128) → ReLU

Dropout

Dense(64) → ReLU

Dense(10) → Softmax


Parameters:

Total params: 362,346

Trainable params: 361,962


This is the deepest model and uses Batch Normalization in the early and mid convolutional layers, plus Dropout in the dense layers to improve stability and generalization on CIFAR-10.


---

4. Training Setup

Common configuration across all experiments:

Framework: TensorFlow / Keras

Loss: sparse_categorical_crossentropy

Optimizer: Adam

Metric: accuracy

Input preprocessing: pixel values scaled to [0, 1]


Epochs:

MNIST: 5 epochs

Fashion-MNIST: 10 epochs

CIFAR-10: up to 20 epochs with EarlyStopping on val_loss


For CIFAR-10, EarlyStopping is used to stop training when validation loss stops improving, preventing overfitting and unnecessary computation.


---

5. Results and Observations

5.1 MNIST

Epochs: 5

Final validation accuracy: ≈ 98.6%

Test accuracy: ≈ 98.5%

Test loss: ≈ 0.056


Classification report (summary):

Precision, Recall, F1-score for all classes: ~0.98–0.99

Overall accuracy: ~99%

Correctly classified: 9,857 / 10,000

Incorrectly classified: 143


Observations:

Even without Dropout or BatchNorm, the CNN performs extremely well due to the simplicity of MNIST.

Training and validation accuracies are very close, indicating low overfitting.

A relatively shallow CNN is sufficient to achieve high performance on this dataset.



---

5.2 Fashion-MNIST

Epochs: 10

Final validation accuracy: ≈ 92.8%

Test accuracy: ≈ 92.4%

Test loss: ≈ 0.23–0.25


Classification report (summary):

Overall accuracy: 92%

Most classes have Precision/Recall in the range 0.90–0.99

Class 6 (shirt) has a lower F1-score (~0.78), as it is visually similar to other tops/shirts.

Correctly classified: 9,240 / 10,000

Incorrectly classified: 760


Observations:

Adding Dropout helped control overfitting and improved generalization.

Accuracy is lower than MNIST, which is expected due to the higher complexity of Fashion-MNIST.

The model still achieves strong performance above 92% accuracy.



---

5.3 CIFAR-10

Epochs: up to 20 (with EarlyStopping)

Best validation accuracy: ≈ 75–76%

Test accuracy: ≈ 74.0%

Test loss: ≈ 0.79


Classification report (summary):

Overall accuracy: 74%

Stronger performance on:

Vehicle-like classes (e.g., automobile, ship, truck)


Weaker performance on:

Animal classes (e.g., bird, cat, deer, dog) – these are more visually similar and harder to separate


Correctly classified: 7,404 / 10,000

Incorrectly classified: 2,596


Observations:

CIFAR-10 is significantly more challenging than MNIST and Fashion-MNIST.

Batch Normalization improves stability and allows deeper architectures to train effectively.

Dropout + EarlyStopping help reduce overfitting, but the accuracy remains lower due to dataset complexity and the absence of data augmentation or pre-trained backbones.

The model behavior is realistic: structured objects like vehicles are easier; animals with similar textures/shapes are harder.



---

6. Improvements and Tuning Tried

Across the three datasets, the following changes and experiments were made:

Baseline CNN (MNIST):

Started with a simple CNN without explicit regularization to establish a strong baseline on an easier dataset.


Dropout (Fashion-MNIST):

Introduced Dropout after the Flatten layer to reduce overfitting.

Helped improve validation and test performance compared to a plain CNN.


BatchNorm + Dropout + EarlyStopping (CIFAR-10):

Added Batch Normalization after Conv layers to stabilize and speed up training.

Used Dropout in the Dense layer to improve generalization.

Applied EarlyStopping based on validation loss to prevent overfitting and save time.



---

7. Files in This Project

CNN_from_Scratch_on_MNIST_Data.ipynb

CNN implementation and training on MNIST


CNN_from_sractch_on_Fashion_MNIST_with_Dropout.ipynb

CNN with Dropout on Fashion-MNIST


CNN_from_Sractch_on_Cifar10_Data.ipynb

CNN with BatchNorm, Dropout, and EarlyStopping on CIFAR-10


README.md

Project description, model summaries, results, and observations




---

8. How to Run

1. Install dependencies:



pip install tensorflow numpy matplotlib scikit-learn

2. Open any of the notebooks in Jupyter, VS Code, or Google Colab:

CNN_from_Scratch_on_MNIST_Data.ipynb

CNN_from_sractch_on_Fashion_MNIST_with_Dropout.ipynb

CNN_from_Sractch_on_Cifar10_Data.ipynb



3. Run all cells in order to:

Load the dataset

Build the CNN model

Train, evaluate, and visualize results




