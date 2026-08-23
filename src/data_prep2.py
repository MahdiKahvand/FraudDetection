import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------
# Load Dataset
# --------------------------------

data_dir = "data"
data_path = os.path.join("..", data_dir, "creditcard.csv")

df = pd.read_csv(data_path)


# --------------------------------
# Dataset Information
# --------------------------------

number_sample, number_columns = df.shape

print(f"Number of samples: {number_sample}")

number_feature = number_columns - 1

print(f"Number of features: {number_feature}\n")


# --------------------------------
# Missing Values
# --------------------------------

missing_values = df.isnull().sum()

print(f"Missing values:\n{missing_values}\n")


# --------------------------------
# Class Distribution
# --------------------------------

print(
    f"Class distribution:\n"
    f"{df['Class'].value_counts()}\n"
)


# --------------------------------
# Class Ratio
# --------------------------------

print("Class ratio:")

print(df["Class"].value_counts(normalize=True))

print()


# --------------------------------
# Duplicate Analysis
# --------------------------------

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

duplicate_ratio = duplicate_count / len(df)

print(f"Duplicate ratio: {duplicate_ratio}")

print()


# --------------------------------
# Descriptive Statistics
# --------------------------------

print("Descriptive statistics:")

print(df.describe())

print()


# --------------------------------
# Remove Duplicate Rows
# --------------------------------

df = df.drop_duplicates().reset_index(drop=True)

print(
    f"Number of samples after removing duplicates: "
    f"{df.shape[0]}"
)


# --------------------------------
# Data Information
# --------------------------------

print("\nData info:")

df.info()

print()


# --------------------------------
# Features and Target
# --------------------------------

target = df.columns.to_list()[-1]

features = df.columns.drop(target).to_list()

print(f"Target: {target}")

print()

print(f"Features: {features}")

print()

print("Data types:")

print(df.dtypes)


# --------------------------------
# Train / Test Split
# --------------------------------

X = df[features]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# --------------------------------
# Feature Scaling
# --------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)