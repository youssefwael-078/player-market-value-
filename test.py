

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


df = pd.read_csv("players_22.csv", low_memory=False)
print("Original Shape:", df.shape)


plt.figure()
df['age'].hist()
plt.title('Age Distribution (Before Cleaning)')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.show()

plt.figure()
df['overall'].hist()
plt.title('Overall Rating Distribution (Before Cleaning)')
plt.xlabel('Overall Rating')
plt.ylabel('Number of Players')
plt.show()

plt.figure()
df['value_eur'].hist(bins=50)
plt.title('Player Market Value Distribution (Before Cleaning)')
plt.xlabel('Market Value (EUR)')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.scatter(df['age'], df['value_eur'], alpha=0.3)
plt.title('Age vs Market Value (Before Cleaning)')
plt.xlabel('Age')
plt.ylabel('Market Value (EUR)')
plt.show()

plt.figure()
plt.boxplot(df['age'].dropna())
plt.title('Age Outliers (Before Cleaning)')
plt.ylabel('Age')
plt.show()


cols_to_drop = [
    'player_url', 'player_face_url',
    'nation_flag_url', 'club_logo_url',
    'club_flag_url', 'sofifa_id'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

#
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

def convert_money(value):
    if isinstance(value, str):
        value = value.replace('€', '')
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'K' in value:
            return float(value.replace('K', '')) * 1_000
    return value

for col in ['value_eur', 'wage_eur', 'release_clause_eur']:
    if col in df.columns:
        df[col] = df[col].apply(convert_money)

df.drop_duplicates(inplace=True)
df = df[(df['age'] >= 16) & (df['age'] <= 45)]

print("After Cleaning Shape:", df.shape)



plt.figure()
df['age'].hist()
plt.title('Age Distribution (After Cleaning)')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.show()

plt.figure()
df['overall'].hist()
plt.title('Overall Rating Distribution (After Cleaning)')
plt.xlabel('Overall Rating')
plt.ylabel('Count')
plt.show()

plt.figure()
df['value_eur'].hist(bins=50)
plt.title('Market Value Distribution (After Cleaning)')
plt.xlabel('Market Value (EUR)')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.scatter(df['age'], df['value_eur'], alpha=0.3)
plt.title('Age vs Market Value (After Cleaning)')
plt.xlabel('Age')
plt.ylabel('Market Value (EUR)')
plt.show()

plt.figure()
plt.boxplot(df['age'])
plt.title('Age Boxplot (After Cleaning)')
plt.ylabel('Age')
plt.show()



features = [
    'age', 'potential',
    'pace', 'shooting', 'passing',
    'dribbling', 'defending', 'physic'
]

X = df[features]


y = np.log1p(df['value_eur'])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



rf = RandomForestRegressor(
    n_estimators=30,
    max_depth=8,
    min_samples_split=30,
    min_samples_leaf=12,
    max_features=0.6,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)



nn = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
nn.fit(X_train_scaled, y_train)



y_pred_rf = rf.predict(X_test)
y_pred_nn = nn.predict(X_test_scaled)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_rf))

r2_nn = r2_score(y_test, y_pred_nn)
mae_nn = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_nn))

print("\n--- Model Performance Comparison ---")
print(f"Random Forest  -> R2: {round(r2_rf, 3)} | MAE (EUR): {round(mae_rf, 2)}")
print(f"Neural Network -> R2: {round(r2_nn, 3)} | MAE (EUR): {round(mae_nn, 2)}")



comparison = pd.DataFrame({
    'Actual Value (€)': np.expm1(y_test[:10]),
    'RF Predicted (€)': np.expm1(y_pred_rf[:10]),
    'NN Predicted (€)': np.expm1(y_pred_nn[:10])
})
print("\nFirst 10 Players Comparison:")
print(comparison)



joblib.dump(rf, "player_price_rf_model.pkl")
joblib.dump(nn, "player_price_nn_model.pkl")
joblib.dump(scaler, "scaler.pkl")


metrics = {
    "Random Forest": {
        "r2": round(float(r2_rf), 4),
        "mae_eur": round(float(mae_rf), 2)
    },
    "Neural Network": {
        "r2": round(float(r2_nn), 4),
        "mae_eur": round(float(mae_nn), 2)
    }
}
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\nModels, Scaler and Metrics saved successfully!")
