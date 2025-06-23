# 🧠 Apnea Severity Detection Using EEG and Biosignals

This project applies machine learning and deep learning techniques to classify the severity of sleep apnea based on EEG signals and biosensor data. Two core approaches are implemented:
- **Baseline Model:** RBF Classifier using Gaussian Processes (scikit-learn)
- **Advanced Model:** Deep Neural Network using PyTorch

---

## 📁 Dataset Description

The dataset consists of 675 patient records and includes the following features:

- `EEG_Signal_Amplitude`
- `EEG_Delta_band`, `EEG_Theta_band`, `EEG_Alpha_band`, `EEG_Beta_band`
- `heart_rate`, `skin_conductance`, `skin_temperature`
- `cortisol_level`, `Systolic_BP`, `Diastolic_BP`
- `apnea_Severity` (Target: 0 = Low, 1 = Medium, 2 = High)

---

## ⚙️ Data Preprocessing Steps

- Label encoding for categorical values (e.g., pulse rate, temperature levels)
- Removed unnecessary index columns
- Handled missing values and standardized feature scaling
- Generated correlation matrix and heatmaps for feature relationships

---

## 📊 Baseline Model: RBF Classifier (GaussianProcessClassifier)

- **Algorithm:** Gaussian Process with Radial Basis Function (RBF) kernel
- **Split:** 300 training samples / 375 testing samples
- **Accuracy:** `90.66%`
- **Libraries Used:** `scikit-learn`, `matplotlib`, `seaborn`
- **Prediction Output:** Provides both class index and mapped severity label

---

## 🔥 Deep Learning Model: Neural Network (PyTorch)

### Architecture:
- Fully Connected Layers: `Input → 128 → 64 → Output`
- Dropout: 0.5 for regularization
- Activation: ReLU

### Training:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Epochs: 100
- **Test Accuracy:** `94.07%`

---

## 📈 Visualizations

- **Pair Plots**: For feature distribution comparison
- **Heatmap**: Shows strong correlation between biosignals and apnea severity
- **Model Loss**: Tracked across epochs to confirm convergence

---

## 🛠 Libraries & Tools

- Python 3
- PyTorch
- scikit-learn
- pandas, numpy
- seaborn, matplotlib
- Jupyter Notebook

---

## 📌 Key Takeaways

- Deep learning offers superior accuracy compared to classical models like RBF.
- EEG beta and theta bands along with biosignals are strong predictors of severity.
- Proper preprocessing (label encoding, scaling) is essential for stable model performance.

---

## 👤 Author

**Sri Kalyan Reddy Akiti**
data science and artificial intelligence
---
