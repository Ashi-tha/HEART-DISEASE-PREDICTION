# ❤️ Heart Disease Prediction System

An AI-powered web application for cardiovascular risk assessment using machine learning.

🔗 **[Live Demo](#https://heart-disease-prediction-p6kofpa2yhuapjxfyxldau.streamlit.app/)** 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

---

## 🎯 Overview

This system predicts heart disease risk based on 13 clinical parameters using a Support Vector Machine (SVM) model with **86.11% F1 score** and **93.94% recall**.

### Key Features

- 🎨 **Beautiful Interactive Interface** - Built with Streamlit
- 📊 **Real-time Risk Assessment** - Instant predictions with visual gauges
- 🎯 **High Accuracy** - 83.61% accuracy on test data
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- 🔒 **Privacy-First** - No data storage, all processing in-memory

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Deploy to Streamlit Cloud (FREE)

1. **Fork this repository** to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and `streamlit_app.py`
6. Click "Deploy"!

**Done!** Your app is now live in ~2 minutes 🎉

📖 **Detailed deployment guide:** See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 83.61% |
| **Precision** | 79.49% |
| **Recall** | 93.94% |
| **F1 Score** | 86.11% |

**Model Type:** Support Vector Machine (SVM) with RBF kernel

**Training Data:** 302 patient records from UCI Heart Disease Dataset

**Features:** 13 clinical parameters including age, sex, blood pressure, cholesterol, ECG results, and cardiac tests

---

## 🏥 Clinical Parameters

The system analyzes:

### Demographics
- Age (20-100 years)
- Sex (Male/Female)

### Clinical Measurements
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Maximum Heart Rate

### Cardiac Assessment
- Chest Pain Type
- Resting ECG Results
- Exercise Induced Angina
- ST Depression
- Slope of Peak Exercise ST
- Number of Major Vessels
- Thalassemia Status

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Framework:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
heart_disease_prediction/
├── streamlit_app.py           # Main application
├── train_model.py             # Model training script
├── test_model.py              # Testing utilities
├── best_model.pkl             # Trained SVM model
├── scaler.pkl                 # Feature scaler
├── requirements.txt           # Python dependencies
├── heart_disease_data.csv     # Training dataset
├── .streamlit/
│   └── config.toml            # App configuration
└── README.md                  # This file
```

---

## 🧪 Testing

Run the test script to verify model performance:

```bash
python test_model.py
```

This will test the model with sample cases and display performance metrics.

---

## 🔄 Retraining the Model

To retrain with new data:

```bash
python train_model.py
```

This will:
1. Load the dataset
2. Preprocess the data
3. Train multiple models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
4. Select the best performer
5. Save the model and scaler

---

## 💡 Usage Examples

### Web Interface

1. Open the app (locally or deployed)
2. Fill in patient information across three tabs
3. Click "🚀 Predict Heart Disease Risk"
4. View results with risk level and recommendations

### Programmatic Usage

```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare patient data
patient_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]

# Scale and predict
scaled_data = scaler.transform([patient_data])
prediction = model.predict(scaled_data)

print("Disease detected" if prediction[0] == 1 else "No disease")
```

---

## ⚠️ Important Disclaimer

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This system is NOT a medical device and should NOT be used for:
- Clinical diagnosis
- Treatment decisions
- Replacing professional medical advice

**Always consult qualified healthcare professionals for medical decisions.**

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests

---

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical data regulations in your jurisdiction.

---

## 🌟 Acknowledgments

- **Dataset:** UCI Machine Learning Repository - Heart Disease Dataset
- **Frameworks:** Streamlit, Scikit-learn
- **Community:** Streamlit and ML communities for support

---

## 🎓 Citations

If using this project in academic research:

```bibtex
@misc{heart_disease_prediction,
  title={Heart Disease Prediction System},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/YOUR-USERNAME/heart-disease-prediction}
}
```

---

## 📈 Future Enhancements

- [ ] Add more ML models (XGBoost, Neural Networks)
- [ ] Implement SHAP values for explainability
- [ ] Add patient history tracking
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Integration with EHR systems
- [ ] Real-time monitoring dashboard

---

**Made with ❤️ using Streamlit and Scikit-learn**

*Last updated: March 2026*
