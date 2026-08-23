import pandas as pd


def load_data(path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


def data_quality_check(df):
    """Perform basic data quality checks."""

    print("========== Data Quality Checks ==========\n")

    # Dataset type
    print(f"Dataset type: {type(df)}\n")

    # Shape
    number_samples, number_columns = df.shape

    print(f"Number of samples: {number_samples}")

    # One column is the target
    number_features = number_columns - 1

    print(f"Number of features: {number_features}")
    print(f"Number of columns: {number_columns}\n")

    # Columns
    print("Columns:")
    print(df.columns.tolist())
    print()

    # Data types
    print("Data types:")
    print(df.dtypes)
    print()

    # Dataset information
    print("Dataset info:")
    df.info()
    print()


def missing_value_analysis(df):
    """Analyze missing values in the dataset."""

    print("========== Missing Value Analysis ==========\n")

    missing_values = df.isnull().sum()

    print("Missing values:")
    print(missing_values)

    print(f"\nTotal missing values: {missing_values.sum()}")
    print()


def duplicate_analysis(df):
    """Analyze duplicate rows."""

    print("========== Duplicate Analysis ==========\n")

    duplicate_count = df.duplicated().sum()

    print(f"Number of duplicate rows: {duplicate_count}")

    duplicate_ratio = duplicate_count / len(df)

    print(f"Duplicate ratio: {duplicate_ratio}")
    print()


def class_distribution(df):
    """Analyze the target class distribution."""

    print("========== Class Distribution ==========\n")

    print("Class distribution:")
    print(df["Class"].value_counts())
    print()

    print("Class ratio:")
    print(df["Class"].value_counts(normalize=True))
    print()

    print("Unique target values:")
    print(df["Class"].unique())
    print()


def descriptive_statistics(df):
    """Display descriptive statistics."""

    print("========== Descriptive Statistics ==========\n")

    print(df.describe())
    print()


def clean_data(df):
    """Remove duplicate rows from the dataset."""

    df = df.drop_duplicates().reset_index(drop=True)

    print("========== Number of samples after removing duplicates. ==========")
    print(f"Number of samples :{df.shape[0]}")
    print()

    return df


def get_features_and_target(df):
    """Separate features and target."""

    target = "Class"

    features = df.columns.drop(target).tolist()

    return features, target


def prepare_data(path):
    """
    Complete data preparation process.

    Returns:
        df: cleaned DataFrame
        features: list of feature names
        target: target column name
    """

    # Load
    df = load_data(path)

    # Data quality
    data_quality_check(df)

    # Missing values
    missing_value_analysis(df)

    # Duplicate analysis
    duplicate_analysis(df)

    # Clean data
    df = clean_data(df)

    # Class distribution
    class_distribution(df)

    # Descriptive statistics
    descriptive_statistics(df)

    # Features and target
    features, target = get_features_and_target(df)

    return df, features, target