import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Making path and load dataset
data_dir = "data"
data_path = os.path.join(".." , data_dir , "creditcard.csv")

df = pd.read_csv(data_path)


# gathering the information from our data (feature and sampels)

number_sample , number_feature = list(df.shape)
print(f"Number of sample :{number_sample}")

# one column is target, so we have :
number_feature -= 1
print(f"Number of features :{number_feature}\n")


# Missing values
missing_values = df.isnull().sum()
print(f"Missing values :\n{missing_values}")

# Class distribution
print(f"Class distribution:{df["Class"].value_counts()}\n")

# Class ratio
print("Class ratio:")
print(df["Class"].value_counts(normalize=True))
print()

# Duplicate analysis
duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows:{duplicate_count}")
print()

duplicate_ratio = duplicate_count / len(df)

print("Duplicate ratio:", duplicate_ratio)
print()


# Descriptive statistics

print("\n\n")
print(f"Descriptive statistics :\n{df.describe()}")



# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)
print()
print(f"the number of sample after removing duplicates :{df.shape[0]}")

# checking part
print("info :\n")
print(df.info)
print("\n\n")
target = df.columns.to_list()[-1]
features = df.columns.drop(target).to_list()

print(f"target :{target}")
print()
print(f"features :{features}")

print("\n\n")
print(f"""The data type :
{df.dtypes}""")



# train test split
X = df[features]
y = df[target]


X_train ,X_test ,y_train ,y.test = train_test_split(
    X ,
    y ,
    random_state = 42 ,
    stratify=y ,
    test_size=0.2
)


# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


