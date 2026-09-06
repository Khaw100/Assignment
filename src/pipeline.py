import os
import pandas as pd
import numpy as np
from dotenv import get_key, load_dotenv

load_dotenv()

INPUT_FILE = os.getenv("INPUT_FILE")
OUTPUT_FILE = os.getenv("OUTPUT_FILE")

class DataPipeline:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.df_cleaned = None
        self.df_final = None
        self.categorical_cols = []
        self.numerical_cols = []

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.input_file)
        self.categorical_cols = self.df.select_dtypes(include="object").columns.tolist()
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        return self.df

    def write_data(self) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.df_final.to_csv(self.output_file, index=False)
        print(f"Data written to {self.output_file}")

    def data_inspection(self, df: pd.DataFrame, label: str = "Original DataFrame") -> None:
        print(f"{label}:")
        print("5 first rows of the DataFrame:")
        print(df.head())

        print("\nDataFrame Shape:")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        print("\nDataFrame Columns and Data Types:")
        print(df.dtypes)

        print("\nDataFrame Info:")
        print(df.info())

        print("\nMissing Values per Column:")
        print(df.isna().sum()[df.isna().sum() > 0])

        print("\nDuplicate Rows Count:")
        print(df.duplicated().sum())

        print("\nUnique Values in Categorical Columns:")
        catl_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in catl_cols:
            print(f"Column '{col}': {df[col].unique()[:]}") 

    def data_cleaning(self) -> pd.DataFrame:
        df = self.df.copy()

        missing_before = df.isna().sum().sum()
        rows_before = df.shape[0]

        # Normalize text columns
        text_cols = df.select_dtypes(include="object").columns.tolist()
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})

        # Convert transaction_date to datetime
        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(
                df["transaction_date"], format="mixed", errors="coerce"
            )

        # Remove duplicate rows based on all columns
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = rows_before - df.shape[0]
        print(f"\nDuplicate rows removed: {duplicates_removed}")

        # Handle missing values per column, imputation is chosen so that rows are not wasted unnecessarily while other columns in that row are valid.
        ## Categorical = Use mode 
        cat_fill_cols = [c for c in ["make", "num-of-doors", "horsepower-binned"] if c in df.columns]
        for col in cat_fill_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        ## Numerical = Use median (outlier resistant)
        num_fill_cols = [c for c in ["stroke", "horsepower", "price"] if c in df.columns]
        for col in num_fill_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Check for implausible values (negative or zero where it should not be possible)
        unwanted_value_cols = [c for c in ["price", "horsepower", "wheel-base", "engine-size", "curb-weight"] if c in df.columns]
        for col in unwanted_value_cols:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                print(f"Found {invalid_count} implausible (<=0) values in '{col}', replacing with median")
                df.loc[df[col] <= 0, col] = df[col].median()
            else:
                print(f"No implausible values found in '{col}'")

        self.df_cleaned = df
        self.categorical_cols = df.select_dtypes(include="object").columns.tolist()
        self.numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        missing_after = df.isna().sum().sum()
        print(f"n rows before cleaning : {rows_before}")
        print(f"n rows after cleaning : {df.shape[0]}")
        print(f"Total missing values before : {missing_before}")
        print(f"Total missing values after : {missing_after}")
        print(f"Columns changed/cleaned : {text_cols + num_fill_cols}")

        return self.df_cleaned

    def data_transformation(self) -> pd.DataFrame:
        df = self.df_cleaned.copy()

        # Ordinal encoding: horsepower-binned (Low/Medium/High) -> 0/1/2
        ordinal_map = {"low": 0, "medium": 1, "high": 2}
        df["horsepower_ordinal"] = df["horsepower-binned"].map(ordinal_map)

        # Convert text categories to numeric counts
        door_map = {"two": 2, "four": 4}
        df["num-of-doors"] = df["num-of-doors"].map(door_map)

        cylinder_map = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}
        df["num-of-cylinders"] = df["num-of-cylinders"].map(cylinder_map)

        # Min-Max Scaling for all wide-range numerical columns
        minmax_cols = [
            "symboling", "num-of-doors", "curb-weight", "num-of-cylinders",
            "engine-size", "horsepower", "peak-rpm", "city-mpg",
            "highway-mpg", "price",
        ]
        for col in minmax_cols:
            col_min, col_max = df[col].min(), df[col].max()
            df[col] = (df[col] - col_min) / (col_max - col_min)
        print(f"\nMin-Max Scaling applied to: {minmax_cols}")
        print(df[minmax_cols].head(3))

        # transaction_date and horsepower-binned are no longer needed
        df = df.drop(columns=["transaction_date", "horsepower-binned"])

        # One-hot encoding for all nominal categorical columns
        onehot_cols = ["body-style", "drive-wheels", "aspiration", "engine-type", "engine-location", "fuel-system"]
        before_cols = df.columns.tolist()
        df = pd.get_dummies(df, columns=onehot_cols)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)
        new_cols = [c for c in df.columns if c not in before_cols]
        print(f"\nNew columns from one-hot encoding {onehot_cols}: {new_cols}")

        # Frequency encoding for 'make' (too many categories for one-hot)
        freq = df["make"].value_counts(normalize=True)
        df["make_freq"] = df["make"].map(freq)
        print(f"\nFrequency encoding example for 'make' (before vs after):")
        print(df[["make", "make_freq"]].head(3))
        df = df.drop(columns=["make"])

        self.df_final = df
        return self.df_final

    def run(self) -> pd.DataFrame:
        self.load_data()
        self.data_inspection(self.df, label="Raw DataFrame (before cleaning)")
        self.data_cleaning()
        self.data_transformation()
        self.data_inspection(self.df_final, label="Final DataFrame (after cleaning & transformation)")
        self.write_data()
        return self.df_final

if __name__ == "__main__":
    dp = DataPipeline(INPUT_FILE, OUTPUT_FILE)
    dp.run()