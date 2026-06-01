"""
Data Preprocessing Pipeline
Author: Repuri Manohar
Description:
Performs data cleaning and preprocessing without model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# --------------------------------------------------
# Configure Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(file_path):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


# --------------------------------------------------
# Missing Values
# --------------------------------------------------

def handle_missing_values(df):
    """Fill missing values."""

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    logging.info("Missing values handled.")
    return df


# --------------------------------------------------
# Duplicate Records
# --------------------------------------------------

def remove_duplicates(df):
    """Remove duplicate rows."""

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    logging.info(
        f"Removed {before - after} duplicate rows."
    )

    return df


# --------------------------------------------------
# Outlier Handling
# --------------------------------------------------

def handle_outliers(df):
    """Cap outliers using IQR method."""

    numerical_cols = df.select_dtypes(include=np.number).columns

    for col in numerical_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.clip(df[col], lower, upper)

    logging.info("Outliers handled.")
    return df


# --------------------------------------------------
# Categorical Encoding
# --------------------------------------------------

def encode_categorical_features(df):
    """Apply One-Hot Encoding."""

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    logging.info("Categorical encoding completed.")
    return df


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

def scale_features(df):
    """Apply StandardScaler."""

    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    logging.info("Feature scaling completed.")
    return df


# --------------------------------------------------
# Skewness Handling
# --------------------------------------------------

def handle_skewness(df):
    """Apply log transformation."""

    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns

    skew_values = df[numerical_cols].skew()

    for col in skew_values.index:

        if abs(skew_values[col]) > 1:
            df[col] = np.log1p(
                df[col] - df[col].min() + 1
            )

    logging.info("Skewness handled.")
    return df


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    input_file = "data/raw/dataset.csv"
    output_file = "data/processed/cleaned_dataset.csv"

    df = load_data(input_file)

    logging.info(f"Dataset Shape: {df.shape}")

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_categorical_features(df)
    df = scale_features(df)
    df = handle_skewness(df)

    df.to_csv(output_file, index=False)

    logging.info(
        f"Cleaned dataset saved to: {output_file}"
    )

    logging.info(
        f"Final Dataset Shape: {df.shape}"
    )


if __name__ == "__main__":
    main()