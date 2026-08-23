import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_prep import prepare_data

# ============================================================
# I. Definition Functions
# ============================================================


# =========================== Chart ===========================

def drawing(length_x, width_y, model_name):

    counts = np.unique(width_y, return_counts=True)[1]

    plt.figure(figsize=(14, 7))

    plt.scatter(
        length_x,
        width_y,
        alpha=0.6
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.title(model_name)

    plt.figtext(
        0.5,
        0.01,
        f"Class '0' = {counts[0]} and Class '1' = {counts[1]}",
        ha="center",
        fontsize=18,
        bbox={
            "facecolor": "orange",
            "alpha": 0.5,
            "pad": 5
        }
    )

    plt.show()


# ====================== Confusion Matrix ======================

def draw_confusion_matrices(y_test, predictions):

    model_names = [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "MLP"
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 7)
    )

    axes = axes.ravel()

    for i in range(4):

        cm = confusion_matrix(
            y_test,
            predictions[i]
        )

        axes[i].imshow(cm)

        axes[i].set_title(
            model_names[i]
        )

        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])

        for row in range(2):

            for col in range(2):

                axes[i].text(
                    col,
                    row,
                    cm[row, col],
                    ha="center",
                    va="center"
                )

    plt.tight_layout()
    plt.show()


# ====================== Logistic Regression ======================

def logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    special_name
):

    LR = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    LR.fit(
        X_train,
        y_train
    )

    y_prediction = LR.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    F1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    print("Prediction distribution:")

    print(
        np.unique(
            y_prediction,
            return_counts=True
        )
    )

    drawing(
        np.arange(len(X_test)),
        y_prediction,
        f"Logistic Regression Prediction - {special_name}"
    )

    return [
        LR,
        y_prediction,
        accuracy,
        precision,
        recall,
        F1
    ]


# ====================== K-Nearest Neighbors ======================

def knn(
    X_train,
    X_test,
    y_train,
    y_test,
    special_name,
    k=5
):

    KN = KNeighborsClassifier(
        n_neighbors=k
    )

    KN.fit(
        X_train,
        y_train
    )

    y_prediction = KN.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    F1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    print("Prediction distribution:")

    print(
        np.unique(
            y_prediction,
            return_counts=True
        )
    )

    drawing(
        np.arange(len(X_test)),
        y_prediction,
        f"KNN Prediction - {special_name}"
    )

    return [
        KN,
        y_prediction,
        accuracy,
        precision,
        recall,
        F1
    ]


# ====================== Decision Tree ======================

def decision_tree(
    X_train,
    X_test,
    y_train,
    y_test,
    special_name,
    max_d=None
):

    DTC = DecisionTreeClassifier(
        max_depth=max_d,
        random_state=42
    )

    DTC.fit(
        X_train,
        y_train
    )

    y_prediction = DTC.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    F1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    print("Prediction distribution:")

    print(
        np.unique(
            y_prediction,
            return_counts=True
        )
    )

    drawing(
        np.arange(len(X_test)),
        y_prediction,
        f"Decision Tree Prediction - {special_name}"
    )

    return [
        DTC,
        y_prediction,
        accuracy,
        precision,
        recall,
        F1
    ]


# =========================== MLP ===========================

def mlp(
    X_train,
    X_test,
    y_train,
    y_test,
    special_name
):

    MLP = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42
    )

    MLP.fit(
        X_train,
        y_train
    )

    y_prediction = MLP.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    F1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    print(
        np.unique(
            y_prediction,
            return_counts=True
        )
    )

    drawing(
        np.arange(len(X_test)),
        y_prediction,
        f"MLP Prediction - {special_name}"
    )

    return [
        MLP,
        y_prediction,
        accuracy,
        precision,
        recall,
        F1
    ]


# ============================================================
# 1. Load and Prepare Data
# ============================================================

file_name = "creditcard.csv"
dir_file = "data"

data_path = os.path.join(
    "..",
    dir_file,
    file_name
)

df, features, target = prepare_data(
    data_path
)


# ============================================================
# 2. Separate Features and Target
# ============================================================

X = df[features]
y = df[target]

print(
    "\n========== Features and Target ==========\n"
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# 3. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    "\n========== Train / Test Split ==========\n"
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# ============================================================
# 4. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "\n========== Feature Scaling ==========\n"
)

print(
    f"X_train_scaled shape: {X_train_scaled.shape}"
)

print(
    f"X_test_scaled shape:  {X_test_scaled.shape}"
)


# ============================================================
# 5. Training Logistic Regression
# ============================================================

print(
    "\n=============== Logistic Regression ===============\n"
)


# -------------------- Scaled --------------------

print("Scaled training result:")

(
    LRC,
    y_predict_LRC,
    acc_default_LRC,
    precision_default_LRC,
    recall_default_LRC,
    f1_default_LRC
) = logistic_regression(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "Scaled"
)

scores_default_LRC = pd.DataFrame({
    "accuracy": [acc_default_LRC],
    "precision": [precision_default_LRC],
    "recall": [recall_default_LRC],
    "f1": [f1_default_LRC]
})

print(
    scores_default_LRC
)


# -------------------- Unscaled --------------------

print("\nUnscaled training result:")

(
    LRC_unscaled,
    y_predict_unscaled_LRC,
    acc_unscaled_LRC,
    precision_unscaled_LRC,
    recall_unscaled_LRC,
    f1_unscaled_LRC
) = logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    "Unscaled"
)

scores_unscaled_LRC = pd.DataFrame({
    "accuracy": [acc_unscaled_LRC],
    "precision": [precision_unscaled_LRC],
    "recall": [recall_unscaled_LRC],
    "f1": [f1_unscaled_LRC]
})

print(
    scores_unscaled_LRC
)


# ============================================================
# 6. Training K-Nearest Neighbors
# ============================================================

print(
    "\n=============== K-Nearest Neighbors ===============\n"
)


# -------------------- Scaled --------------------

print("Scaled training result:")

(
    KNN,
    y_predict_KNN,
    acc_default_KNN,
    precision_default_KNN,
    recall_default_KNN,
    f1_default_KNN
) = knn(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "Scaled"
)

scores_default_KNN = pd.DataFrame({
    "accuracy": [acc_default_KNN],
    "precision": [precision_default_KNN],
    "recall": [recall_default_KNN],
    "f1": [f1_default_KNN]
})

print(
    scores_default_KNN
)


# -------------------- Unscaled --------------------

print("\nUnscaled training result:")

(
    KNN_unscaled,
    y_predict_unscaled_KNN,
    acc_unscaled_KNN,
    precision_unscaled_KNN,
    recall_unscaled_KNN,
    f1_unscaled_KNN
) = knn(
    X_train,
    X_test,
    y_train,
    y_test,
    "Unscaled"
)

scores_unscaled_KNN = pd.DataFrame({
    "accuracy": [acc_unscaled_KNN],
    "precision": [precision_unscaled_KNN],
    "recall": [recall_unscaled_KNN],
    "f1": [f1_unscaled_KNN]
})

print(
    scores_unscaled_KNN
)


# ============================================================
# 7. Training Decision Tree
# ============================================================

print(
    "\n=============== Decision Tree Classifier ===============\n"
)

print("Default training result:")

(
    DTC,
    y_predict_DTC,
    acc_default_DTC,
    precision_default_DTC,
    recall_default_DTC,
    f1_default_DTC
) = decision_tree(
    X_train,
    X_test,
    y_train,
    y_test,
    "Default"
)

scores_default_DTC = pd.DataFrame({
    "accuracy": [acc_default_DTC],
    "precision": [precision_default_DTC],
    "recall": [recall_default_DTC],
    "f1": [f1_default_DTC]
})

print(
    scores_default_DTC
)


# ============================================================
# 8. Multi Layer Perceptron
# ============================================================

print(
    "\n=============== Multi Layer Perceptron ===============\n"
)

print("Scaled training result:")

(
    MLP,
    y_predict_MLP,
    acc_default_MLP,
    precision_default_MLP,
    recall_default_MLP,
    f1_default_MLP
) = mlp(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "Scaled"
)

scores_default_MLP = pd.DataFrame({
    "accuracy": [acc_default_MLP],
    "precision": [precision_default_MLP],
    "recall": [recall_default_MLP],
    "f1": [f1_default_MLP]
})

print(
    scores_default_MLP
)


# ============================================================
# 9. Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "MLP": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=100,
                random_state=42
            )
        )
    ])
}


scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


cv_results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean()
    })


cv_results = pd.DataFrame(
    cv_results
)


print(
    "\n========== 5-Fold Stratified Cross Validation ==========\n"
)

print(
    cv_results.round(4).to_string(index=False)
)


# ============================================================
# 10. Total Result and Compare
# ============================================================

model_name_list = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "MLP"
]

accuracy_total_list = [
    acc_default_LRC,
    acc_default_KNN,
    acc_default_DTC,
    acc_default_MLP
]

precision_total_list = [
    precision_default_LRC,
    precision_default_KNN,
    precision_default_DTC,
    precision_default_MLP
]

recall_total_list = [
    recall_default_LRC,
    recall_default_KNN,
    recall_default_DTC,
    recall_default_MLP
]

f1_total_list = [
    f1_default_LRC,
    f1_default_KNN,
    f1_default_DTC,
    f1_default_MLP
]

total_result_table = pd.DataFrame({
    "Model": model_name_list,
    "Accuracy": accuracy_total_list,
    "Precision": precision_total_list,
    "Recall": recall_total_list,
    "F1": f1_total_list
})


print(
    "\n========== Test Set Results ==========\n"
)

print(
    total_result_table.round(4).to_string(index=False)
)


# ============================================================
# 11. Confusion Matrix
# ============================================================

draw_confusion_matrices(
    y_test,
    [
        y_predict_LRC,
        y_predict_KNN,
        y_predict_DTC,
        y_predict_MLP
    ]
)