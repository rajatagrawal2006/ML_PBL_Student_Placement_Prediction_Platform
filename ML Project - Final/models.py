# Machine learning models, pipelines, metrics, and training routines

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Standard options matching requirements
CLASSIFIERS = ["Logistic Regression", "Decision Tree", "KNN", "SVM", "Naive Bayes"]
SCALERS = ["StandardScaler", "MinMaxScaler"]
FEATURE_METHODS = ["SelectKBest", "PCA"]
REGRESSORS = ["Linear", "Polynomial", "Gradient Descent"]


def split_data(X, y_class, y_reg=None, test_size=0.2):
    """
    Splits features and targets into train and test sets (80/20) with stratified class split.
    """
    counts = np.bincount(y_class)
    can_stratify = len(counts) > 1 and np.min(counts) >= 2
    strat = y_class if can_stratify else None

    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, stratify=strat, random_state=42
    )

    X_train = X.iloc[idx_train].copy()
    X_test = X.iloc[idx_test].copy()
    yc_train = y_class.iloc[idx_train].copy()
    yc_test = y_class.iloc[idx_test].copy()

    if y_reg is not None:
        yr_train = y_reg.iloc[idx_train].copy()
        yr_test = y_reg.iloc[idx_test].copy()
    else:
        yr_train = None
        yr_test = None

    return X_train, X_test, yc_train, yc_test, yr_train, yr_test


def build_classification_pipeline(model, scaler_name, feature_method, n_features, probability=False):
    """
    Builds an end-to-end scikit-learn pipeline with imputer, scaler,
    feature selection or extraction, and classifier.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]

    # Scaler step
    if scaler_name == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))
    else:
        steps.append(("scaler", StandardScaler()))

    # Feature reduction step
    if feature_method == "PCA":
        k_pca = min(10, n_features)
        steps.append(("features", PCA(n_components=k_pca, random_state=42)))
    else:
        k_best = min(10, n_features)
        steps.append(("features", SelectKBest(score_func=f_classif, k=k_best)))

    # Classifier step
    # Decision tree using entropy criterion conceptually mirrors the ID3 algorithm
    if model == "Decision Tree":
        clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=5, class_weight="balanced", random_state=42
        )
    elif model == "KNN":
        clf = KNeighborsClassifier(n_neighbors=7)
    elif model == "SVM":
        clf = SVC(kernel="rbf", class_weight="balanced", probability=probability, random_state=42)
    elif model == "Naive Bayes":
        clf = GaussianNB(priors=[0.5, 0.5])
    else:
        # Default: Logistic Regression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    steps.append(("classifier", clf))
    return Pipeline(steps)


def train_classifier(model, scaler_name, feature_method, X_train, y_train, X_test, y_test):
    """
    Trains a classification pipeline and evaluates on the test set.
    Returns a dict with metrics, y_pred, and confusion matrix with labels=[0, 1].
    """
    pipe = build_classification_pipeline(
        model, scaler_name, feature_method, X_train.shape[1], probability=False
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "metrics": {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
        },
        "y_pred": y_pred,
        "cm": cm,
        "pipeline": pipe,
    }


def fit_classifier_for_prediction(model, scaler, feature, X_train, y_train):
    """
    Builds and fits a classification pipeline for inference.
    Enables probability=True for SVM so predict_proba works.
    """
    pipe = build_classification_pipeline(
        model, scaler, feature, X_train.shape[1], probability=True
    )
    pipe.fit(X_train, y_train)
    return pipe


def compare_all_models(X_train, y_train, X_test, y_test, scaler_name="StandardScaler", feature_method="SelectKBest"):
    """
    Trains all five classifiers and returns a list of dictionaries with their performance metrics.
    """
    results = []
    for name in CLASSIFIERS:
        res = train_classifier(name, scaler_name, feature_method, X_train, y_train, X_test, y_test)
        met = res["metrics"]
        results.append({
            "Model": name,
            "Accuracy": met["Accuracy"],
            "Precision": met["Precision"],
            "Recall": met["Recall"],
            "F1 Score": met["F1 Score"],
        })
    return results


def top_features(X_train, y_train, n=8):
    """
    Imputes missing values with median, runs ANOVA F-test,
    and returns the top n column names with highest F-scores.
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)
    scores, _ = f_classif(X_imp, y_train)
    scores = np.nan_to_num(scores, nan=0.0)

    cols = list(X_train.columns)
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:min(n, len(cols))]
    return [cols[i] for i in top_indices]


def calculate_regression_metrics(y_true, y_pred, p_features):
    """
    Computes MAE, MSE, RMSE, R2, and Adjusted R2.
    """
    n = len(y_true)
    p = p_features
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    if n - p - 1 > 0:
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = r2

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adj_r2,
    }


def custom_gradient_descent(X_train_arr, y_train_arr, X_test_arr, lr=0.1, max_iter=2000):
    """
    Handwritten linear regression via Batch Gradient Descent using MSE/2 cost.
    Adds a bias column of ones, tracks cost history, and supports early stopping.
    """
    m = len(y_train_arr)
    X_b_train = np.c_[np.ones((m, 1)), X_train_arr]
    X_b_test = np.c_[np.ones((len(X_test_arr), 1)), X_test_arr]

    num_params = X_b_train.shape[1]
    weights = np.zeros(num_params)
    cost_history = []

    initial_err = X_b_train.dot(weights) - y_train_arr
    initial_cost = (1.0 / (2.0 * m)) * np.sum(initial_err ** 2)
    cost_history.append(initial_cost)

    for _ in range(max_iter):
        preds = X_b_train.dot(weights)
        errors = preds - y_train_arr
        gradient = (1.0 / m) * X_b_train.T.dot(errors)
        weights = weights - lr * gradient

        current_err = X_b_train.dot(weights) - y_train_arr
        cost = (1.0 / (2.0 * m)) * np.sum(current_err ** 2)
        cost_history.append(cost)

        # Early stopping when cost change is below 1e-6
        if abs(cost_history[-2] - cost) < 1e-6:
            break

    y_test_pred = X_b_test.dot(weights)
    return y_test_pred, cost_history


def train_regression(method, X_train, y_train, X_test, y_test, lr=0.1, max_iter=2000):
    """
    Trains Linear, Polynomial, or Gradient Descent regression.
    Returns a dict with y_pred, metrics dict, and history (cost list).
    """
    n_features = X_train.shape[1]
    cost_history = []
    y_test_arr = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

    if method == "Polynomial":
        # Pipeline: SimpleImputer -> StandardScaler -> SelectKBest -> PolynomialFeatures -> LinearRegression
        k_sel = min(5, n_features)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_regression, k=k_sel)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        poly_step = pipe.named_steps["poly"]
        p_features = poly_step.n_output_features_
        metrics = calculate_regression_metrics(y_test_arr, y_pred, p_features)

    elif method == "Gradient Descent":
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
        X_te_scaled = scaler.transform(imputer.transform(X_test))

        y_train_arr = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        y_pred, cost_history = custom_gradient_descent(
            X_tr_scaled, y_train_arr, X_te_scaled, lr=lr, max_iter=max_iter
        )
        metrics = calculate_regression_metrics(y_test_arr, y_pred, n_features)

    else:
        # Default: Standard Linear Regression
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = calculate_regression_metrics(y_test_arr, y_pred, n_features)

    return {
        "y_pred": y_pred,
        "metrics": metrics,
        "history": cost_history,
    }


def run_kmeans_clustering(X, k=3):
    """
    Standardizes features, performs K-Means clustering,
    and returns labels, 2D PCA projection, cluster size counts, and inertia.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_prep = scaler.fit_transform(imputer.fit_transform(X))

    k_clean = max(2, min(int(k), 6))
    kmeans = KMeans(n_clusters=k_clean, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_prep)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_prep)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    return labels, pca_2d, cluster_counts, kmeans.inertia_


def calculate_elbow_curve(X, max_k=8):
    """
    Calculates K-Means inertia for k from 1 to max_k on imputed and standardized X.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_prep = scaler.fit_transform(imputer.fit_transform(X))

    k_values = list(range(1, max_k + 1))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_prep)
        inertias.append(km.inertia_)

    return k_values, inertias
