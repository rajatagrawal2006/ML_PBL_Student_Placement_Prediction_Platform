# Data utilities for loading and generating student performance data

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

# Flag to decide whether to include earlier term grades (G1, G2)
USE_EARLIER_GRADES = False

# Local folder for datasets
DATA_DIR = "data"
STUDENT_CSV_PATH = os.path.join(DATA_DIR, "student-mat.csv")
STUDENT_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"


def load_student_dataset():
    """
    Loads the student performance dataset.
    Downloads the archive if not already cached locally in the data folder.
    Returns (X, y_class, y_reg, class_names).
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download and extract if file does not exist
    if not os.path.exists(STUDENT_CSV_PATH):
        zip_path = os.path.join(DATA_DIR, "student.zip")
        try:
            req = urllib.request.Request(
                STUDENT_ZIP_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=45) as response, open(zip_path, "wb") as out_file:
                out_file.write(response.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract("student-mat.csv", DATA_DIR)
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download the student performance dataset: {e}. "
                "Please check your internet connection or use synthetic/CSV data."
            )

    # Read the extracted CSV with semicolon delimiter
    try:
        df = pd.read_csv(STUDENT_CSV_PATH, sep=";")
    except Exception as e:
        raise RuntimeError(f"Error reading the student performance dataset: {e}")

    if "G3" not in df.columns:
        raise ValueError("Invalid format: G3 column missing from the dataset.")

    # Targets: classification (pass: G3 >= 10) and regression (G3 numeric)
    y_reg = df["G3"].astype(float).copy()
    y_class = (df["G3"] >= 10).astype(int).copy()
    class_names = ("Fail", "Pass")

    # Drop target and earlier grades depending on setting
    if USE_EARLIER_GRADES:
        drop_cols = ["G3"]
    else:
        drop_cols = ["G1", "G2", "G3"]

    features_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # One-hot encode categorical text columns
    X = pd.get_dummies(features_df, drop_first=True).astype(float)

    return X, y_class, y_reg, class_names


def generate_synthetic_data(n_samples=395, noise=0.25, seed=42):
    """
    Generates synthetic student performance data with 5 numeric features.
    Returns (X, y_class, y_reg, class_names).
    """
    rng = np.random.default_rng(seed)

    # 5 realistic student features
    study_hours = rng.normal(5.0, 1.8, size=n_samples).clip(0.5, 12.0)
    attendance = rng.uniform(50.0, 100.0, size=n_samples)
    previous_score = rng.normal(65.0, 15.0, size=n_samples).clip(10.0, 95.0)
    sleep_hours = rng.normal(7.0, 1.2, size=n_samples).clip(4.0, 10.0)
    phone_hours = rng.normal(3.5, 1.5, size=n_samples).clip(0.5, 9.0)

    # Formula for score on a 0-20 scale
    base_score = (
        1.0 * study_hours
        + 0.10 * attendance
        + 0.10 * previous_score
        + 0.30 * sleep_hours
        - 0.50 * phone_hours
        - 6.0
    )

    # Add random noise scaled by user parameter
    noise_term = rng.normal(0, 1.0, size=n_samples) * (0.6 + float(noise) * 1.8)
    final_score = (base_score + noise_term).clip(2.0, 19.0)

    X = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "phone_hours": phone_hours
    }).astype(float)

    y_reg = pd.Series(final_score, name="final_score").astype(float)
    median_val = float(y_reg.median())
    y_class = (y_reg >= median_val).astype(int)
    class_names = ("Fail", "Pass")

    return X, y_class, y_reg, class_names


def load_csv_dataset(path, target_column):
    """
    Loads any CSV dataset and preprocesses features and targets.
    Returns (X, y_class, y_reg, class_names).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV file: {e}")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the CSV file. "
            f"Available columns: {', '.join(df.columns)}"
        )

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])
    if len(df) == 0:
        raise ValueError("Dataset is empty after dropping rows with missing target values.")

    # Drop id and name columns if present (not the target column)
    cols_to_drop = [
        c for c in df.columns
        if c.strip().lower() in ["id", "name"] and c != target_column
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    target_series = df[target_column]
    features_df = df.drop(columns=[target_column])

    # Determine if target is numeric or categorical
    if pd.api.types.is_numeric_dtype(target_series):
        unique_vals = np.sort(target_series.unique())
        if len(unique_vals) == 2:
            # Binary numeric target (e.g. 0 and 1)
            val_low, val_high = unique_vals[0], unique_vals[1]
            low_str = str(int(val_low)) if float(val_low).is_integer() else str(val_low)
            high_str = str(int(val_high)) if float(val_high).is_integer() else str(val_high)
            y_class = (target_series == val_high).astype(int)
            y_reg = None
            class_names = (low_str, high_str)
        elif len(unique_vals) > 2:
            # Continuous numeric target with more than 2 distinct values
            y_reg = target_series.astype(float).copy()
            med = float(y_reg.median())
            y_class = (y_reg > med).astype(int)
            class_names = ("Below median", "Above median")
            if len(np.unique(y_class)) < 2:
                raise ValueError(
                    "Splitting target above median produced only one class. "
                    "Values may be largely identical."
                )
        else:
            raise ValueError("Numeric target must have at least 2 distinct values.")
    else:
        # Categorical / text target
        y_reg = None
        unique_cats = list(target_series.dropna().unique())
        n_cats = len(unique_cats)
        if n_cats != 2:
            raise ValueError(
                f"A text target must have exactly 2 categories, but this column has {n_cats}."
            )

        pos_keywords = {"yes", "y", "true", "pass", "positive", "1"}
        cat0, cat1 = unique_cats[0], unique_cats[1]
        cat0_clean = str(cat0).strip().lower()
        cat1_clean = str(cat1).strip().lower()

        if cat0_clean in pos_keywords and cat1_clean not in pos_keywords:
            pos_cat, neg_cat = cat0, cat1
        elif cat1_clean in pos_keywords and cat0_clean not in pos_keywords:
            pos_cat, neg_cat = cat1, cat0
        else:
            # If neither matches positive keywords, choose the alphabetically last category
            sorted_cats = sorted(unique_cats, key=lambda x: str(x).lower())
            neg_cat = sorted_cats[0]
            pos_cat = sorted_cats[1]

        y_class = (target_series == pos_cat).astype(int)
        class_names = (str(neg_cat), str(pos_cat))

    # One-hot encode remaining text columns
    X = pd.get_dummies(features_df, drop_first=True).astype(float)

    return X, y_class, y_reg, class_names
