<h1 align="center">Advanced ML Housing Price Prediction Model</h1>
<p align="center">
  <img src="housing_banner.png" alt="Housing Price Prediction" width="500"/>
</p>

A comprehensive machine learning project demonstrating a complete machine learning pipeline for predicting housing prices using the California Housing dataset. The notebook showcases advanced ML techniques, including multiple algorithm comparisons, hyperparameter tuning, and comprehensive model evaluation with overfitting prevention.

---

## 🚀 Table of Contents

- [✨ Features](#-features)
- [🗂️ Project Structure](#️-project-structure)
- [📊 Dataset Overview](#-dataset-overview)
- [🛠️ Installation](#️-installation)
- [▶️ Usage](#️-usage)
- [📈 Results & Visualizations](#-results--visualizations)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- **Dataset**: California Housing Dataset (20,640 samples, 8 features)
- **Multiple Algorithms**:  
  - Linear Regression (baseline)  
  - Ridge & Lasso Regression (regularization)  
  - Random Forest (ensemble bagging)  
  - Gradient Boosting (ensemble boosting)  
  - Support Vector Regression (kernel methods)
- **Advanced ML Techniques**:  
  - GridSearchCV for hyperparameter optimization  
  - 5-fold Cross-validation for robust evaluation  
  - Train/Validation/Test splits (70%-15%-15%)  
  - Feature scaling with StandardScaler
- **Performance**:  
  - **Best Model**: Gradient Boosting Regressor  
  - **Test Accuracy**: 82.98% (R² score)  
  - **Overfitting Control**: 8.83% gap (excellent generalization)  
  - **Test MSE**: 0.2231
- **Professional Visualization**: Comprehensive EDA, feature importance, residual analysis
- **Production Ready**: Model serialization and prediction functions
- **Technologies Used**: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 🗂️ Project Structure

```
├── .gitignore
├── LICENSE
├── README.md
├── housing_banner.png
├── requirements.txt
├── Housing_Price_Prediction.ipynb

```

- **`Housing_Price_Prediction.ipynb`**: Complete ML pipeline with professional documentation  
- **`requirements.txt`**: Project dependencies  

---

## 📊 Dataset Overview

The **California Housing Dataset** contains information about housing districts in California from the 1990 census:

| Feature       | Description                            | Range               |
|--------------|----------------------------------------|---------------------|
| **MedInc**    | Median income in block group           | 0.5 - 15.0          |
| **HouseAge**  | Median house age in block group        | 1 - 52 years        |
| **AveRooms**  | Average number of rooms per household  | 0.85 - 141.9        |
| **AveBedrms** | Average number of bedrooms per household | 0.33 - 34.1       |
| **Population**| Block group population                 | 3 - 35,682          |
| **AveOccup**  | Average number of household members    | 0.69 - 1,243        |
| **Latitude**  | House block group latitude             | 32.54 - 41.95       |
| **Longitude** | House block group longitude            | -124.35 - -114.31   |

**Target**: Median house value in hundreds of thousands of dollars

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/X-XENDROME-X/Advanced-ML-Housing-Prediction-Model.git

2. **Set up a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Housing_Price_Prediction.ipynb
   ```

---

## ▶️ Usage

### 🔬 Complete Analysis

Run through the notebook sections:

1. **Data Loading & Exploration**: Understanding the California housing market  
2. **Statistical Analysis**: Comprehensive data quality assessment  
3. **Exploratory Data Analysis**: Visual insights and feature relationships  
4. **Data Preprocessing**: Feature scaling and train/test splits  
5. **Model Development**: Training and comparing 6 different algorithms  
6. **Hyperparameter Tuning**: Optimizing model performance  
7. **Model Evaluation**: Comprehensive testing and validation  
8. **Results Visualization**: Professional plots and interpretability  

### 🎯 Quick Prediction

```python
from sklearn.externals import joblib  # or import joblib directly

# Load trained model
model = joblib.load('models/housing_price_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Make prediction
def predict_house_price(median_income, house_age, avg_rooms, avg_bedrooms,
                        population, avg_occupancy, latitude, longitude):
    features = [[median_income, house_age, avg_rooms, avg_bedrooms,
                 population, avg_occupancy, latitude, longitude]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction

# Example: San Francisco area property
price = predict_house_price(8.5, 15, 6.5, 1.1, 3000, 2.8, 37.78, -122.42)
print(f"Predicted Price: ${price:.2f} hundred thousand")
```

---

## 📈 Results & Visualizations

### Key Insights

1. 📈 **Gradient Boosting** achieved the best balance of accuracy and generalization  
2. 🎯 **Median Income** is the strongest predictor (0.688 correlation with price)  
3. 🗺️ **Geographic Location** significantly impacts housing values  
4. ⚖️ **Ensemble Methods** outperformed linear models substantially  
5. 🛡️ **Overfitting Control** successfully implemented across all models  

### Model Performance

- **Best Model**: Gradient Boosting Regressor
- **Test Accuracy**: 82.98% (R² = 0.8298)
- **Generalization**: Excellent (8.83% overfitting gap)

### Key Insights
- **Median Income** is the strongest predictor of housing prices
- **Geographic location** (Latitude/Longitude) significantly impacts prices
- **Ensemble methods** outperformed linear models
- **Proper regularization** successfully prevented overfitting

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add new feature'`)  
4. Push to the branch (`git push origin feature-name`)  
5. Open a Pull Request  

**Areas for contribution**:

- Additional feature engineering techniques  
- Model optimization and tuning  
- Visualization improvements  
- Documentation enhancements  
- Production deployment examples  

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

