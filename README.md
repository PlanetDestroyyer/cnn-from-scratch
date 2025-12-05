# CNN Image Classification From Scratch  
Using MNIST, Fashion-MNIST, and CIFAR-10

## 1. Overview

This project implements **Convolutional Neural Networks (CNNs) from scratch** (no pre-trained models) for three different image classification tasks:

- **MNIST** – handwritten digit classification  
- **Fashion-MNIST** – clothing item classification  
- **CIFAR-10** – object classification on RGB images  

The goal is to design custom CNN architectures, train them end-to-end, and compare how model complexity and regularization change with dataset difficulty.

All models are implemented using **TensorFlow/Keras**, with evaluation done via accuracy, classification report, and confusion matrix.

---

## 2. Datasets Used

All datasets are loaded directly from `tensorflow.keras.datasets`:

### 2.1 MNIST
- 60,000 training images, 10,000 test images  
- Grayscale, 28×28 pixels  
- 10 classes: digits 0–9  

### 2.2 Fashion-MNIST
- 60,000 training images, 10,000 test images  
- Grayscale, 28×28 pixels  
- 10 classes: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot  

### 2.3 CIFAR-10
- 50,000 training images, 10,000 test images  
- RGB, 32×32 pixels  
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  

---

## 3. Model Architectures (Layer Details)

### 3.1 MNIST – CNN Without Dropout / BatchNorm  
Notebook: `CNN_from_Scratch_on_MNIST_Data.ipynb`

**Architecture:**

- `Conv2D(32, 3×3)` → ReLU  
- `Conv2D(32, 3×3)` → ReLU  
- `MaxPooling2D(2×2)`  
- `Conv2D(64, 3×3)` → ReLU  
- `Conv2D(64, 3×3)` → ReLU  
- `MaxPooling2D(2×2)`  
- `Flatten`  
- `Dense(128)` → ReLU  
- `Dense(10)` → ReLU  
- `Dense(64)` → ReLU  
- `Dense(10)` → Softmax  

**Total parameters:** 469,172  
**Trainable params:** 469,172  

This is a relatively simple CNN with two convolutional blocks followed by fully connected layers, without any explicit regularization (no Dropout/BatchNorm).

---

### 3.2 Fashion-MNIST – CNN With Dropout  
Notebook: `CNN_from_sractch_on_Fashion_MNIST_with_Dropout.ipynb`

**Architecture:**

- `Conv2D(32, 3×3)` → ReLU  
- `Conv2D(32, 3×3)` → ReLU  
- `MaxPooling2D(2×2)`  
- `Conv2D(64, 3×3)` → ReLU  
- `Conv2D(64, 3×3)` → ReLU  
- `MaxPooling2D(2×2)`  
- `Flatten`  
- `Dropout`  
- `Dense(128)` → ReLU  
- `Dense(10)` → Softmax  

**Total parameters:** 467,818  
**Trainable params:** 467,818  

Compared to MNIST, this model introduces **Dropout after Flatten** to reduce overfitting on the more challenging Fashion-MNIST dataset.

---

### 3.3 CIFAR-10 – CNN With BatchNorm + Dropout + EarlyStopping  
Notebook: `CNN_from_Sractch_on_Cifar10_Data.ipynb`

**Architecture:**

- `Conv2D(32, 3×3)`  
- `BatchNormalization`  
- `ReLU`  
- `Conv2D(32, 3×3)`  
- `BatchNormalization`  
- `ReLU`  
- `MaxPooling2D(2×2)`  

- `Conv2D(64, 3×3)`  
- `BatchNormalization`  
- `ReLU`  
- `Conv2D(64, 3×3)`  
- `BatchNormalization`  
- `ReLU`  
- `MaxPooling2D(2×2)`  

- `Conv2D(128, 3×3)` → ReLU  
- `Conv2D(128, 3×3)` → ReLU  
- `MaxPooling2D(2×2)`  

- `Flatten`  
- `Dense(128)` → ReLU  
- `Dropout`  
- `Dense(64)` → ReLU  
- `Dense(10)` → Softmax  

**Total parameters:** 362,346  
**Trainable params:** 361,962  

This is the deepest model, using **Batch Normalization** in early/mid conv layers and **Dropout** in the dense layer. This combination stabilizes training and improves generalization on the more complex CIFAR-10 dataset.

---

## 4. Training Setup

Common setup across all models:

- **Framework:** TensorFlow / Keras  
- **Loss:** `sparse_categorical_crossentropy`  
- **Optimizer:** `Adam`  
- **Metrics:** `accuracy`  
- **Input Scaling:** pixel values normalized to `[0, 1]`  

**Epochs:**
- MNIST: 5 epochs  
- Fashion-MNIST: 10 epochs  
- CIFAR-10: up to 20 epochs with **EarlyStopping** based on `val_loss`  

For CIFAR-10, EarlyStopping was used to avoid overfitting once the validation loss stopped improving.

---

## 5. Results and Observations

### 5.1 MNIST

- **Epochs:** 5  
- **Final Validation Accuracy:** ~98.6%  
- **Test Accuracy:** ~98.5%  
- **Test Loss:** ~0.056  

**Classification Report Highlights:**

- Precision, Recall, F1-score ≈ **0.98–0.99** for all classes  
- Overall accuracy ≈ **99%**  
- **Correctly classified:** 9,857 / 10,000  
- **Incorrectly classified:** 143  

**Observations:**

- Even without Dropout or BatchNorm, the CNN performs extremely well due to the simplicity of MNIST.  
- Training and validation accuracies are very close, indicating **low overfitting**.  
- This confirms that a relatively shallow CNN is sufficient for MNIST.

---

### 5.2 Fashion-MNIST

- **Epochs:** 10  
- **Final Validation Accuracy:** ~92.8%  
- **Test Accuracy:** ~92.4%  
- **Test Loss:** ~0.23–0.25  

**Classification Report Highlights:**

- Overall accuracy: **92%**  
- Most classes have Precision/Recall ≈ 0.90–0.99  
- Class `6` (shirt) has slightly lower F1-score (~0.78), which is expected since it is visually similar to other clothing categories.  
- **Correctly classified:** 9,240 / 10,000  
- **Incorrectly classified:** 760  

**Observations:**

- Adding **Dropout** helped control overfitting and improved generalization.  
- Performance is slightly lower than MNIST, reflecting the higher difficulty of Fashion-MNIST.  
- The model still achieves strong performance above 92% accuracy.

---

### 5.3 CIFAR-10

- **Epochs:** up to 20 (with EarlyStopping)  
- **Best Validation Accuracy:** ~75–76%  
- **Test Accuracy:** ~74.0%  
- **Test Loss:** ~0.79  

**Classification Report Highlights:**

- Overall accuracy: **74%**  
- Strong performance on:
  - Class 1 (automobile), Class 8 (ship), Class 9 (truck)  
- Lower performance on:
  - Class 2 (bird), Class 3 (cat), Class 4 (deer), Class 5 (dog) – these are visually more similar and harder to separate.  
- **Correctly classified:** 7,404 / 10,000  
- **Incorrectly classified:** 2,596  

**Observations:**

- CIFAR-10 is significantly more challenging due to color images and complex object shapes.  
- **Batch Normalization** stabilized training and allowed deeper architecture.  
- **Dropout** + **EarlyStopping** helped reduce overfitting, but accuracy is still lower than MNIST/Fashion-MNIST, which is expected for a scratch CNN without pretraining.  
- The model shows realistic class-wise behavior: animals are harder, vehicles are easier.

---

## 6. Improvements and Tuning Tried

Across the three experiments:

- Started with a **basic CNN** on MNIST (no regularization) to understand baseline behavior.  
- Added **Dropout** for Fashion-MNIST to combat mild overfitting and observed improvement in generalization.  
- For CIFAR-10:
  - Introduced **Batch Normalization** after convolutional layers to stabilize and speed up training.
  - Added **Dropout** in dense layers to reduce overfitting.
  - Used **EarlyStopping** to stop training when validation loss stopped improving, preventing unnecessary overfitting and saving compute.

**Potential Future Improvements (Not implemented but possible):**

- Data augmentation for CIFAR-10 (random flips, shifts, etc.)  
- Learning rate scheduling (e.g., `ReduceLROnPlateau`)  
- Deeper architectures or residual connections for CIFAR-10  

---

## 7. Files in This Project

- `CNN_from_Scratch_on_MNIST_Data.ipynb`  
  - CNN implementation and training on MNIST  
- `CNN_from_sractch_on_Fashion_MNIST_with_Dropout.ipynb`  
  - CNN with Dropout on Fashion-MNIST  
- `CNN_from_Sractch_on_Cifar10_Data.ipynb`  
  - CNN with BatchNorm, Dropout, and EarlyStopping on CIFAR-10  
- `README.md`  
  - Project description, model summaries, results, and observations

---

## 8. How to Run

Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn

```

