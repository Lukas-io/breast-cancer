# 🩺 Breast Cancer Prediction System

A web-based tool that predicts breast cancer classification (Malignant or Benign) using machine learning. This system leverages a trained Support Vector Machine (SVM) model on the `sklearn` breast cancer dataset to provide early risk assessment based on diagnostic measurements.

## 💡 Features

- **High Accuracy**: Predictions powered by Support Vector Machine (SVM) algorithm
- **Comprehensive Analysis**: Processes 30 diagnostic features (mean, error, and worst values)
- **User-Friendly Interface**: Clean, responsive UI with intuitive tabbed layout
- **Full-Stack Solution**: Python Flask backend with HTML/CSS/JS frontend
- **Detailed Results**: Clear explanation of prediction with relevant patient data
- **Privacy-Focused**: Locally hosted for data security and offline usage

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/breast-cancer-predictor.git
   cd breast-cancer-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install flask joblib scikit-learn numpy flask-cors
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser and navigate to http://127.0.0.1:5000

## 🧠 How It Works

1. User inputs the required diagnostic measurements
2. Data is preprocessed using StandardScaler
3. SVM model evaluates the scaled input
4. Results are displayed as:
   - ✅ **Benign**: Non-Cancerous
   - ⚠️ **Malignant**: Cancerous

## 🗂️ Project Structure

```
breast-cancer-predictor/
├── app.py                  # Flask application server
├── models/
│   ├── scaler.pkl          # Trained StandardScaler
│   └── svm_model.pkl       # Trained SVM model
├── static/
│   ├── css/                # Stylesheet files
│   └── js/                 # JavaScript files
└── templates/
    └── index.html          # Web interface
```

## 🛠️ Model Training

The SVM model was trained on the Wisconsin Breast Cancer dataset from scikit-learn with the following steps:

1. Data preprocessing and feature scaling
2. Model selection and hyperparameter tuning
3. Cross-validation for performance assessment

To retrain the model (optional):
```bash
python train.py
```

## ⚠️ Disclaimer

This tool is for **educational and demonstration purposes only**. It is **not intended for clinical use** and should not replace professional medical diagnosis. Always consult qualified healthcare providers for medical advice and diagnosis.

## 📊 Dataset

The Wisconsin Breast Cancer dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses, describing characteristics of cell nuclei.

## 📜 License

[MIT License](LICENSE)

## 📧 Contact

For questions or feedback, please open an issue on this repository or contact [your-email@example.com].
