# Rice Type Classification using Deep Learning

A comprehensive deep learning project for classifying rice types (Cammeo vs Osmancik) using PyTorch and tabular data analysis. This project implements a neural network classifier with extensive data visualization and analysis capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## 🌾 Overview

This project uses deep learning to classify rice grains into two varieties: **Cammeo** and **Osmancik**. The classification is based on morphological features extracted from rice grain images, including area, perimeter, major axis length, minor axis length, eccentricity, and other geometric properties.

### Key Highlights
- **Binary Classification** of rice types using neural networks
- **Comprehensive Data Analysis** with 12+ visualizations
- **Real-time Inference** capability with user input
- **Training Monitoring** with loss and accuracy tracking
- **Feature Importance Analysis** and correlation studies

## 📊 Dataset

The dataset contains **3,810 rice grain samples** with the following features:

| Feature | Description |
|---------|-------------|
| **Area** | Area of rice grain |
| **Perimeter** | Perimeter of rice grain |
| **MajorAxisLength** | Length of the major axis |
| **MinorAxisLength** | Length of the minor axis |
| **Eccentricity** | Eccentricity measure |
| **ConvexArea** | Convex area of the grain |
| **EquivDiameter** | Equivalent diameter |
| **Extent** | Extent of the grain |
| **Roundness** | Roundness measure |
| **AspectRatio** | Aspect ratio of the grain |
| **Class** | Target variable (Cammeo/Osmancik) |

**Data Source**: [Kaggle - Rice Type Classification Dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)

## ✨ Features

### 🔬 Data Analysis
- **Statistical summaries** and data distribution analysis
- **Correlation matrix** visualization
- **Feature importance** ranking based on target correlation
- **Outlier detection** and analysis
- **Class distribution** visualization

### 🧠 Deep Learning Model
- **Custom PyTorch neural network** with configurable architecture
- **Binary classification** using sigmoid activation
- **GPU/CPU compatibility** with automatic device detection
- **Batch processing** with DataLoader implementation
- **Model summary** with parameter counting

### 📈 Training & Evaluation
- **Train/Validation/Test split** (70%/15%/15%)
- **Real-time training monitoring** with loss and accuracy tracking
- **Comprehensive visualization** of training progress
- **Model performance metrics** across all datasets

### 🎯 Inference System
- **Interactive prediction** with user input
- **Automatic normalization** of input features
- **Real-time classification** results

## 🚀 Installation

### Prerequisites
- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DL_rice_classification
   ```

2. **Install required packages**:
   ```bash
   pip install torch torchvision torchaudio
   pip install opendatasets torchsummary
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   pip install jupyter notebook
   ```

3. **Download the dataset**:
   The notebook automatically downloads the dataset from Kaggle using the `opendatasets` library.

## 💻 Usage

### Running the Complete Pipeline

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook Tabular_Classification.ipynb
   ```

2. **Execute cells sequentially**:
   - **Cells 1-2**: Dataset download and initialization
   - **Cells 3-5**: Data loading and preprocessing
   - **Cells 6-8**: Dataset splitting and PyTorch dataset creation
   - **Cells 9-12**: Model definition and setup
   - **Cell 13**: Training loop execution
   - **Cell 14**: Model testing and evaluation
   - **Cell 15**: Basic training visualization
   - **Cell 16**: Comprehensive data analysis dashboard
   - **Cell 17**: Interactive inference system

### Quick Start Example

```python
# Load and preprocess data
data_df = pd.read_csv("./rice-type-classification/riceClassification.csv")
data_df.dropna(inplace=True)
data_df.drop(["id"], axis=1, inplace=True)

# Normalize features
for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max()

# Train model (see notebook for complete implementation)
model = MyModel().to(device)
# ... training loop ...

# Make predictions
prediction = model(input_tensor)
rice_type = "Cammeo" if prediction < 0.5 else "Osmancik"
```

## 🏗️ Model Architecture

### Neural Network Structure
```
Input Layer (10 features) → Hidden Layer (10 neurons) → Output Layer (1 neuron)
                          ↓
                    Sigmoid Activation
```

### Technical Specifications
- **Input Features**: 10 morphological measurements
- **Hidden Neurons**: 10 (configurable)
- **Activation Function**: Sigmoid (for binary classification)
- **Loss Function**: Binary Cross Entropy (BCELoss)
- **Optimizer**: Adam with learning rate 1e-3
- **Batch Size**: 32
- **Training Epochs**: 10

### Model Parameters
```
Total Parameters: 121
Trainable Parameters: 121
Model Size: ~0.5 KB
```

## 📊 Results

### Model Performance
- **Training Accuracy**: ~99.5%
- **Validation Accuracy**: ~92.8%
- **Test Accuracy**: ~93.2%

### Training Characteristics
- **Convergence**: Fast convergence within 10 epochs
- **Loss Reduction**: Significant improvement from initial to final loss
- **Stability**: Consistent validation performance

### Feature Importance Ranking
Based on correlation with target variable:
1. **Area** - Highest correlation with rice type
2. **Perimeter** - Strong morphological indicator
3. **MajorAxisLength** - Important geometric feature
4. **ConvexArea** - Significant shape descriptor

## 📈 Visualizations

The project includes a comprehensive visualization dashboard with 12 different plots:

### 🔍 Data Analysis Plots
1. **Class Distribution Pie Chart** - Shows balance between rice types
2. **Feature Correlation Heatmap** - Inter-feature relationships
3. **Feature Distribution Histograms** - Statistical distributions
4. **Box Plots by Class** - Feature differences between rice types

### 📉 Training Monitoring
5. **Training vs Validation Loss** - Model learning progress
6. **Training vs Validation Accuracy** - Performance tracking
7. **Loss Reduction Rate** - Learning efficiency per epoch

### 🎯 Model Analysis
8. **Feature Importance Bar Chart** - Ranked feature contributions
9. **Feature Scatter Plot** - 2D visualization of top features
10. **Performance Comparison** - Train/Val/Test accuracy comparison

### 🔧 Data Processing
11. **Normalization Effect** - Before/after preprocessing comparison
12. **Outlier Analysis** - Outlier count by feature

## 📁 Project Structure

```
DL_rice_classification/
│
├── Tabular_Classification.ipynb    # Main notebook with complete pipeline
├── rice-type-classification/       # Dataset directory
│   └── riceClassification.csv     # Rice classification dataset
```

## 🔧 Technical Details

### Data Preprocessing
- **Missing Value Handling**: Automatic removal of null values
- **Feature Normalization**: Min-max scaling (division by maximum value)
- **ID Column Removal**: Eliminates non-predictive identifier

### Training Strategy
- **Data Splitting**: Stratified split maintaining class balance
- **Batch Processing**: Efficient memory usage with batch size 32
- **Gradient Optimization**: Adam optimizer with learning rate scheduling
- **Validation Monitoring**: Real-time overfitting detection

### Device Compatibility
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Note: Change 'cuda' to 'mps' for Apple Silicon Macs
```

### Memory Optimization
- **Tensor Operations**: GPU-accelerated computations
- **Batch Loading**: Efficient data pipeline with DataLoader
- **Gradient Management**: Proper gradient zeroing and backpropagation

## 🤝 Contributing

Contributions are welcome! Here are ways you can contribute:

### Enhancement Ideas
- [ ] **Model Architecture**: Experiment with deeper networks or different architectures
- [ ] **Hyperparameter Tuning**: Implement grid search or Bayesian optimization
- [ ] **Data Augmentation**: Add synthetic data generation techniques
- [ ] **Cross-Validation**: Implement k-fold cross-validation
- [ ] **Deployment**: Create web API or mobile app interface
- [ ] **Model Interpretability**: Add SHAP or LIME explanations

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Dataset**: Thanks to the creators of the Rice Type Classification dataset on Kaggle
- **PyTorch Team**: For the excellent deep learning framework
- **Scikit-learn**: For preprocessing and evaluation utilities
- **Matplotlib/Seaborn**: For comprehensive visualization capabilities

