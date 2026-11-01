# Student Performance Predictor

A complete, working machine learning project built with Python and Tkinter for analyzing, modeling, and predicting student academic success. The application features an interactive GUI that loads or generates datasets, performs data preprocessing, trains multiple classification and regression algorithms, visualizes metrics and plots, performs K-Means clustering, and provides single-sample student performance inference.

---

## 1. Project Overview

The **Student Performance Predictor** consists of 5 interactive tabs:
1. **1. Data**: Load the mathematics course student performance dataset (cached in `data/student-mat.csv`), load custom CSV files with flexible target handling, or generate synthetic student records with adjustable sample size and noise. Displays metric cards for rows, features, and class distribution, plus an 8-factor preview table.
2. **2. Classification**: Train and evaluate 5 different classifiers (**Logistic Regression**, **Decision Tree**, **KNN**, **SVM**, and **Naive Bayes**) with customizable feature scaling (**StandardScaler**, **MinMaxScaler**) and feature processing (**SelectKBest**, **PCA**). Includes a multi-model comparison table, a sample test predictions table, and an embedded confusion matrix plot.
3. **3. Regression**: Train continuous grade prediction models (**Linear Regression**, **Polynomial Regression** with degree 2 and feature selection, and a handcrafted **Batch Gradient Descent** algorithm with MSE/2 cost tracking). Displays MAE, MSE, RMSE, R², Adjusted R², actual vs. predicted scatter plots, and cost convergence curves.
4. **4. Clustering**: Perform unsupervised **K-Means Clustering** ($k = 2$ to $6$) with cluster distribution breakdowns and 2D PCA projections, along with an **Elbow Method** plot ($k = 1$ to $8$) to determine the optimal number of clusters.
5. **5. Predict**: Enter student factor values for the top 8 predictive features (determined via ANOVA F-test) to obtain a live **Pass / Fail** prediction with confidence probabilities.

---

## 2. Project Structure

```text
ML Project/
├── data/                    # Local cache folder for the dataset
│   └── student-mat.csv      # Mathematics course student records
├── data_utils.py            # Data loading, generation, cleaning, and preprocessing
├── models.py                # Pipelines, classifiers, regression models, gradient descent, clustering
├── main.py                  # Tkinter graphical user interface with 5 tabs and embedded plots
├── requirements.txt         # Required Python packages
└── README.md                # Documentation, dataset details, and viva/report theory notes
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10 or newer (tested on Windows, macOS, and Linux).

### Steps to Run

1. **Open the project folder:**
   ```bash
   cd "ML Project"
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   python main.py
   ```

---

## 4. Dataset Description

- **Source Archive**: [Student Performance Dataset Zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip)
- **File Used**: `student-mat.csv` (Mathematics course, 395 student records, ~30 original attributes).
- **Classification Target**: Pass (`G3 >= 10`, coded as 1) vs Fail (`G3 < 10`, coded as 0). Class names: `("Fail", "Pass")`.
- **Regression Target**: Numeric final examination grade `G3` (range 0 to 20).

### Why are G1 and G2 Excluded by Default?
In the dataset, `G1` and `G2` represent the first and second period examination marks. Because they occur right before the final examination (`G3`), their correlation with `G3` exceeds 0.85. If `G1` and `G2` remain in the feature set, models trivially rely on them, overshadowing social, demographic, and behavioral factors. 

By excluding `G1` and `G2` (`USE_EARLIER_GRADES = False` at the top of `data_utils.py`), the learning algorithms discover whether lifestyle and study habits—such as study hours, attendance, family support, and alcohol consumption—can predict whether a student will succeed or struggle. If you wish to test with prior grades included, set `USE_EARLIER_GRADES = True`.

---

## 5. Theory Notes for College Report & Viva

### 5.1 Supervised vs Unsupervised vs Reinforcement Learning
- **Supervised Learning**: The algorithm learns from labeled training examples $(X, y)$ to approximate a mapping function $f(X) \approx y$. Tasks include Classification (predicting discrete labels like Pass/Fail) and Regression (predicting continuous values like final grade).
- **Unsupervised Learning**: The algorithm works on unlabeled inputs $X$ to uncover underlying structures, groupings, or representations without pre-existing target labels (e.g., K-Means Clustering and PCA).
- **Reinforcement Learning (RL)**: An autonomous agent learns how to act in an environment through trial-and-error, receiving scalar rewards or penalties for its actions to learn an optimal policy that maximizes cumulative reward over time.

### 5.2 Linear and Logistic Regression
- **Linear Regression**: Fits a linear relationship between continuous dependent variable $y$ and independent features $X$:
  $$\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p = Xw$$
  Parameters are estimated by minimizing Mean Squared Error (MSE).
- **Logistic Regression**: Used for binary classification (0 or 1). It applies the sigmoid (logistic) function to a linear combination of features:
  $$P(y=1|X) = \sigma(Xw) = \frac{1}{1 + e^{-Xw}}$$
  If $P(y=1|X) \ge 0.5$, the sample is classified as class 1 (Pass); otherwise class 0 (Fail). Class-weight balancing prevents majority-class bias.

### 5.3 Polynomial Regression
Extends linear regression to model non-linear patterns by introducing higher-order terms ($x_1^2, x_1 x_2, \dots$). To prevent feature explosion and severe overfitting, the pipeline first selects the top informative features using an ANOVA regression F-test (`SelectKBest(f_regression, k=5)`), then applies `PolynomialFeatures(degree=2, include_bias=False)`. The Adjusted $R^2$ calculation penalizes model complexity using the total number of polynomial output terms ($p = \text{poly.n\_output\_features\_}$).

### 5.4 Gradient Descent and Convergence
Gradient Descent is an iterative optimization method that adjusts model weights in the direction of steepest descent:
- **Weight Update**:
  $$w := w - \alpha \nabla_w J(w)$$
  where $\alpha$ is the learning rate and $\nabla_w J(w)$ is the gradient vector.
- **Cost Function (MSE / 2)**:
  $$J(w) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$
- **Gradient**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (Xw - y)$$
- **Convergence**: As weights approach their optimal values, the gradient approaches zero and the cost curve flattens. Early stopping terminates training when $|J_t - J_{t-1}| < 10^{-6}$.

### 5.5 Normalization vs Standardization
- **Normalization (MinMaxScaler)**: Rescales feature values to $[0, 1]$:
  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
- **Standardization (StandardScaler)**: Centers feature values to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
  $$x_{\text{std}} = \frac{x - \mu}{\sigma}$$
  Standardization is robust to outliers and matches algorithms that assume normally distributed variables.

### 5.6 Feature Selection (SelectKBest) vs Feature Extraction (PCA)
- **SelectKBest (Feature Selection)**: Computes statistical significance scores (such as ANOVA F-test `f_classif`) for each individual feature against the target and selects the top $k$ features while retaining their original real-world interpretation.
- **PCA (Principal Component Analysis / Feature Extraction)**: Projects data onto orthogonal axes that maximize variance. It compresses high-dimensional feature spaces into $k$ uncorrelated principal components, reducing dimensionality at the cost of direct feature interpretability.

### 5.7 Decision Trees and ID3
Decision Trees recursively partition feature space into pure subsets. The **ID3** algorithm uses **Entropy** to quantify uncertainty and **Information Gain** to select splitting attributes:
$$\text{Entropy}(S) = -\sum_{i} p_i \log_2(p_i)$$
In this project, `DecisionTreeClassifier(criterion="entropy", max_depth=5, class_weight="balanced")` implements entropy-based splitting with constrained depth to avoid overfitting.

### 5.8 K-Nearest Neighbors (KNN)
KNN is an instance-based non-parametric classifier. Given a new query sample, it computes distances (such as Euclidean distance) to all training points, identifies the $k$ nearest neighbors ($k=7$), and assigns the majority class. Feature scaling is essential so large-scale features do not dominate distance calculations.

### 5.9 Support Vector Machines (SVM)
SVM finds an optimal hyperplane that separates classes with the maximum geometric margin. When linear boundaries are insufficient, the Radial Basis Function (RBF) kernel implicitly projects features into a higher-dimensional space where linear separation is achievable. Enabling `probability=True` allows calibrated posterior probability estimation.

### 5.10 Naive Bayes
Based on Bayes' Theorem, Naive Bayes assumes features are conditionally independent given the class label:
$$P(y|X) = \frac{P(y) \prod_{j=1}^p P(x_j|y)}{P(X)}$$
Gaussian Naive Bayes models continuous features with Gaussian distributions and trains quickly with minimal data.

### 5.11 Bayesian Belief Networks (BBN)
A Bayesian Belief Network is a Directed Acyclic Graph (DAG) where nodes represent random variables and directed edges represent conditional dependencies. Unlike Naive Bayes, BBNs explicitly model variable interdependencies and conditional independence, enabling probabilistic inference under uncertainty.

### 5.12 K-Means Clustering and the Elbow Method
- **K-Means**: Partitions samples into $k$ clusters by iteratively assigning points to the nearest centroid and updating centroids to the cluster mean.
- **Inertia (Within-Cluster Sum of Squares / WCSS)**:
  $$\text{Inertia} = \sum_{k=1}^K \sum_{x \in C_k} ||x - \mu_k||^2$$
- **Elbow Method**: Plots Inertia across $k = 1, \dots, 8$. The optimal $k$ corresponds to the inflection point where additional clusters yield diminishing returns in inertia reduction.

---

## 6. Evaluation Metrics Summary

### Regression
1. **Mean Absolute Error (MAE)**: $\frac{1}{n}\sum |y_i - \hat{y}_i|$.
2. **Mean Squared Error (MSE)**: $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$.
3. **Root Mean Squared Error (RMSE)**: $\sqrt{\text{MSE}}$.
4. **$R^2$ Score**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
5. **Adjusted $R^2$**: $1 - (1 - R^2)\frac{n - 1}{n - p - 1}$, where $p$ is the number of features.

### Classification
1. **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$.
2. **Precision**: $\frac{TP}{TP + FP}$.
3. **Recall**: $\frac{TP}{TP + FN}$.
4. **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
5. **Confusion Matrix**: $2 \times 2$ table showing True Negatives, False Positives, False Negatives, and True Positives.
