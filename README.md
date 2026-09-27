# ⚽ FIFA Player Price Predictor

An End-to-End Machine Learning Web Application that estimates football players' market values in Euros (€) using EA Sports FIFA data, Random Forest Regression, and Streamlit.

![App Screenshot](images/stadium.jpg)

## 📌 Features

- **Interactive UI:** Dynamic attribute sliders built with Streamlit.
- **FUT Card UI:** Displays price predictions inside a custom FIFA-style UI card.
- **Skewness Mitigation:** Uses Log Transformation (`np.log1p`) to handle heavily skewed market value data.
- **Overfitting Control:** Regularized Random Forest parameters (`max_depth=8`, `min_samples_leaf=12`).

## 🛠️ Tech Stack

- **Language:** Python
- **ML / Data Science:** Pandas, NumPy, Scikit-Learn, Joblib
- **Web Framework:** Streamlit

## 📊 Model Performance

- **$R^2$ Score:** `~0.955`
- **Mean Absolute Error (MAE):** `~€666,583`

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yousef6867/FIFA-Player-Price-Predictor.git]
   cd FIFA-Player-Price-Predictor
   ```
