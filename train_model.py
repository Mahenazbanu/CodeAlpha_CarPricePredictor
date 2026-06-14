"""
Car Price Prediction - Model Trainer
Run this script once to train and save the ML model.
Usage: python train_model.py --data car_data.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

CURRENT_YEAR = 2026


def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    # Rename columns for consistency
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "Selling_type": "Seller_Type",
        "Driven_kms": "Kms_Driven",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Feature engineering
    df["Car_Age"] = CURRENT_YEAR - df["Year"]

    # Encode categoricals
    le_fuel = LabelEncoder()
    le_seller = LabelEncoder()
    le_trans = LabelEncoder()
    le_car = LabelEncoder()

    df["Fuel_Type_enc"] = le_fuel.fit_transform(df["Fuel_Type"])
    df["Seller_Type_enc"] = le_seller.fit_transform(df["Seller_Type"])
    df["Transmission_enc"] = le_trans.fit_transform(df["Transmission"])
    df["Car_Name_enc"] = le_car.fit_transform(df["Car_Name"])

    features = [
        "Car_Name_enc", "Car_Age", "Present_Price", "Kms_Driven",
        "Fuel_Type_enc", "Seller_Type_enc", "Transmission_enc", "Owner"
    ]
    X = df[features]
    y = df["Selling_Price"]

    encoders = {
        "fuel": le_fuel,
        "seller": le_seller,
        "transmission": le_trans,
        "car_name": le_car,
    }
    return X, y, encoders, df


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "y_test": y_test.values,
            "y_pred": preds,
        }
        print(f"  {name:25s}  R²={results[name]['r2']:.4f}  MAE={results[name]['mae']:.4f}  RMSE={results[name]['rmse']:.4f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\n✅ Best model: {best_name} (R²={results[best_name]['r2']:.4f})")
    return results, best_name, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="car_data.csv")
    parser.add_argument("--out", default="car_price_model.pkl")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Dataset not found: {args.data}")
        return

    print("🔄 Loading and preprocessing data …")
    X, y, encoders, df = load_and_preprocess(args.data)
    print(f"   Dataset shape: {df.shape}")

    print("\n📊 Training models …")
    results, best_name, X_test, y_test = train_and_evaluate(X, y)

    best_model = results[best_name]["model"]
    feature_names = list(X.columns)

    bundle = {
        "model": best_model,
        "best_name": best_name,
        "encoders": encoders,
        "feature_names": feature_names,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
        "current_year": CURRENT_YEAR,
        "car_names": sorted(df["Car_Name"].unique().tolist()),
    }

    joblib.dump(bundle, args.out)
    print(f"\n💾 Model bundle saved → {args.out}")


if __name__ == "__main__":
    main()