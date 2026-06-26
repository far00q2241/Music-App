# 🎧 Music Listening Time Prediction using Machine Learning

A machine learning project that predicts **daily music listening time (in minutes)** using user listening behavior. The project demonstrates data preprocessing, feature selection, model training, evaluation, and deployment using Streamlit.

---

## 📌 Project Overview

The objective of this project is to predict how many minutes a user listens to music each day based on their listening habits.

After preprocessing the data and selecting the most important features, a **Random Forest Regressor** was trained to accurately predict users' daily listening time.

---

## 🎯 Objectives

* Predict daily music listening time.
* Identify the most influential user behavior features.
* Reduce unnecessary features using feature importance.
* Build and deploy a machine learning web application using Streamlit.

---

## 📊 Dataset Information

* **Total Records:** 4,000
* **Original Features:** 15
* **Features after Encoding:** 55+

### Target Variable

* `daily_listening_minutes`

### Important Features

* Age
* Songs Per Day
* Playlists Count
* Skip Rate (%)
* Country
* Platform
* Subscription Type
* Top Genre
* Top Artist
* Top Mood

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns
* Handled missing values
* Encoded categorical variables
* Converted data types
* Performed feature engineering
* Selected the most important features using Random Forest Feature Importance

---

## 📈 Exploratory Data Analysis

Correlation analysis showed that **Songs Per Day** has a very strong relationship with the target variable.

| Feature       | Correlation |
| ------------- | ----------: |
| Songs Per Day |  **0.9797** |

This indicates that user listening behavior is the strongest predictor of daily listening time.

---

## 🧠 Feature Selection

Feature importance analysis identified the following four features as the most useful:

```python
X = df[['age',
        'songs_per_day',
        'playlists_count',
        'skip_rate_pct']]

y = df['daily_listening_minutes']
```

Reducing the feature set from **55 features to only 4** improved model simplicity while maintaining excellent predictive performance.

---

## 🤖 Machine Learning Model

**Algorithm Used**

* Random Forest Regressor

### Model Parameters

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
```

---

## 📊 Model Performance

### Test Results

| Metric   |             Score |
| -------- | ----------------: |
| R² Score |        **0.9540** |
| RMSE     | **18.14 minutes** |

### Cross Validation (5-Fold)

| Metric        |      Score |
| ------------- | ---------: |
| Mean R² Score | **0.9531** |

---

## 💡 Key Findings

* Songs Per Day is the most influential feature.
* User behavior features outperform demographic features.
* Feature selection reduced model complexity significantly.
* The final model achieved excellent prediction accuracy with only four input features.

---

## 🚀 Streamlit Web Application

The trained model was deployed using **Streamlit**.

### User Inputs

* Age
* Songs Per Day
* Playlists Count
* Skip Rate (%)

### Output

* Predicted Daily Listening Time (Minutes)

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 📁 Project Structure

```
Music-App/
│
├── app.py
├── music_model.pkl
├── requirements.txt
├── README.md
├── notebook.ipynb
└── images/
    └── app.png
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Music-App.git
```

Move into the project directory:

```bash
cd Music-App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📷 Application Preview

Add a screenshot of your Streamlit application here.

```
images/app.png
```

---

## 🔮 Future Improvements

* Compare with XGBoost and LightGBM.
* Add SHAP explainability.
* Deploy using Docker.
* Collect real-world user data.
* Improve UI with interactive visualizations.

---

## 👨‍💻 Author

**Farooq Khan**

Machine Learning | Data Science | Python

If you found this project useful, feel free to ⭐ the repository.
