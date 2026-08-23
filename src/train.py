import os  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score ,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_prep import prepare_data

# ============================================================
# I. definition functions
# ============================================================

## ===================  chart  ===================

def drawing(lenght_x ,wigdth_y ,model_name):

    entities = np.unique_counts(wigdth_y)[1]


    plt.figure(figsize=(14,8))
    plt.scatter(
        lenght_x ,
        wigdth_y ,
        alpha=0.6
        )
    plt.xlabel("sampel digit")
    plt.ylabel("Class")
    plt.title(model_name)

    plt.figtext(0.5, 0.01, f"Class '0' = {entities[0]} and Class '1' = {entities[1]}", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    plt.show()

## ===================  logestic_reggresion  ===================

def logestic_reggresion(X_train ,X_test ,y_train ,y_test,special_name):

    LR = LogisticRegression()
    LR = LR.fit(X_train ,y_train)
    y_prediction = LR.predict(X_test)

    accuracy = accuracy_score(y_test ,y_prediction)
    precision = precision_score(y_test ,y_prediction)
    recall = recall_score(y_test ,y_prediction)
    F1 = f1_score(y_test ,y_prediction)

    print(np.unique_counts(y_prediction))

    drawing(np.array(range(len(X_test))) ,y_prediction ,f"Logestic Reggresion prediction {special_name}")
    return [LR ,y_prediction ,accuracy ,precision ,recall ,F1]

## ===================  Knearset_neighbore  ===================

def Knearset_neighbore(X_train ,X_test ,y_train ,y_test,special_name ,k=5):

    KN = KNeighborsClassifier(n_neighbors = k)
    KN = KN.fit(X_train ,y_train)
    y_prediction = KN.predict(X_test)

    accuracy = accuracy_score(y_test ,y_prediction)
    precision = precision_score(y_test ,y_prediction)
    recall = recall_score(y_test ,y_prediction)
    F1 = f1_score(y_test ,y_prediction)

    print(np.unique_counts(y_prediction))

    drawing(np.array(range(len(X_test))) ,y_prediction ,f"KNeighborsClassifier prediction {special_name}")
    return [KN ,y_prediction ,accuracy ,precision ,recall ,F1]

## ===================  decision_tree  ===================

def decision_tree(X_train ,X_test ,y_train ,y_test,special_name ,max_d = None):

    DTC = DecisionTreeClassifier(max_depth = max_d)
    DTC = DTC.fit(X_train ,y_train)
    y_prediction = DTC.predict(X_test)

    accuracy = accuracy_score(y_test ,y_prediction)
    precision = precision_score(y_test ,y_prediction)
    recall = recall_score(y_test ,y_prediction)
    F1 = f1_score(y_test ,y_prediction)

    print(np.unique_counts(y_prediction))

    drawing(np.array(range(len(X_test))) ,y_prediction ,f"Decision Tree Classifier prediction {special_name}")
    return [DTC ,y_prediction ,accuracy ,precision ,recall ,F1]


# ============================================================
# 1. Load and Prepare Data
# ============================================================

file_name = "creditcard.csv"
dir_file = "data"
data_path = os.path.join(".." , dir_file , file_name)

df, features, target = prepare_data(data_path)


# ============================================================
# 2. Separate Features and Target
# ============================================================

X = df[features]

y = df[target]

print("\n========== Features and Target ==========\n")

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

print("\n========== Train / Test Split ==========\n")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# ============================================================
# 4. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n========== Feature Scaling ==========\n")

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")


# ============================================================
# 5 training LogesticReggression
# ============================================================

print("""\n===============   trainig models (Logestic Reggression)   ===============\n""")

print("scaled trained result")
LRC ,y_predict_LRC , acc_default_LRC ,precision_default_LRC ,recall_default_LRC ,f1_default_LRC = logestic_reggresion(X_train_scaled ,X_test_scaled ,y_train ,y_test,"default")
print()

scores_deaflut_LRC = pd.DataFrame({"accuray":acc_default_LRC ,"precision":precision_default_LRC ,"recall":recall_default_LRC ,"f1":f1_default_LRC})
print(scores_deaflut_LRC)


print("unscaled trained result")
y_predict_unscaled_LRC , acc_unscaled_LRC ,precision_unscaled_LRC ,recall_unscaled_LRC ,f1_unscaled_LRC = logestic_reggresion(X_train ,X_test ,y_train ,y_test, "unscaled")
print()
scores_unscaled_LRC = pd.DataFrame({"accuray":acc_unscaled_LRC ,"precision":precision_unscaled_LRC ,"recall":recall_unscaled_LRC ,"f1":f1_unscaled_LRC})
print(scores_deaflut_LRC)
# ============================================================
# 6 training KNeighborsClassifier
# ============================================================

print("""\n===============   trainig models (KNeighborsClassifier)   ===============\n""")

print("scaled trained result")
KNN ,y_predict_KNN , acc_default_KNN ,precision_default_KNN ,recall_default_KNN ,f1_default_KNN = Knearset_neighbore(X_train_scaled ,X_test_scaled ,y_train ,y_test ,"default")
print()

print("unscaled trained result")
y_predict_unscaled_KNN , acc_unscaled_KNN ,precision_unscaled_KNN ,recall_unscaled_KNN ,f1_unscaled_KNN = Knearset_neighbore(X_train ,X_test ,y_train ,y_test , "unscaled")
print()
# ============================================================
# 7 training DecisionTreeClassifier
# ============================================================

print("""\n===============   trainig models (DecisionTreeClassifier)   ===============\n""")

print("default trained result")
DTC ,y_predict_DTC , acc_default_DTC ,precision_default_DTC ,recall_default_DTC ,f1_default_DTC = decision_tree(X_train ,X_test ,y_train ,y_test ,"default")
