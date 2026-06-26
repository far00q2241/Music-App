# 🎧 Predicting Daily Music Listening Time Using User Behavior Data

## 📌 Project Overview

This project builds a machine learning model to predict **daily listening minutes** of music users based on their behavioral and demographic data.

The main goal is to identify which user activity patterns most strongly influence listening time and develop an accurate and interpretable regression model.

---

## 🎯 Objective

- Predict `daily_listening_minutes`
- Identify key behavioral drivers of music consumption
- Reduce noise from high-dimensional categorical features
- Build a high-performance regression model

---

## 📊 Dataset

- **Rows:** 4000
- **Initial Features:** 15
- **Engineered Features:** 55+

### Key Features:
- `age` – User age  
- `songs_per_day` – Number of songs listened per day  
- `playlists_count` – Number of playlists used  
- `skip_rate_pct` – Percentage of skipped songs  
- `country` – User location  
- `platform` – Music platform used  
- `subscription` – Subscription type  
- `top_genre`, `top_artist`, `top_mood`

### Target Variable:
- `daily_listening_minutes`

---

## 🧹 Data Preprocessing

- Handled missing and inconsistent values
- Cleaned mixed-format numeric fields
- Encoded categorical variables
- Performed feature engineering
- Reduced feature space from 55 → 4 key features based on importance analysis

---

## 🔍 Exploratory Data Analysis

A very strong relationship was observed between:



This indicates that user activity level is the dominant factor in predicting listening time.

---

## 🧠 Feature Selection

Feature importance analysis (Random Forest) revealed:

- `songs_per_day` → ~75% importance (dominant feature)
- `skip_rate_pct` → strong behavioral signal
- `playlists_count` → moderate contribution
- `age` → minor contribution

### Final Feature Set:

```python
X = df[['age', 'songs_per_day', 'playlists_count', 'skip_rate_pct']]
y = df['daily_listening_minutes']
