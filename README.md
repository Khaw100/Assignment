# Data Pipeline Assignment

## Brief Dataset Description
This dataset contains automotive data from the Automobile dataset. The raw file used in this project is a training version that contains missing values, duplicate rows, unnormalized text, and categorical columns that are not yet ready for modeling.

## Dataset Source
The dataset is based on the Automobile dataset from the UCI Machine Learning Repository (dataset ID 10), originally sourced from Ward's Automotive Yearbook 1985.

Source: https://archive.ics.uci.edu/dataset/10/automobile

## Project Folder Structure
- `data/raw/automobileEDA_dirty_training.csv`: raw dataset
- `data/processed/automobileEDA_cleaned_training.csv`: processed dataset
- `data/Dataset_Sesi_3/`: supporting dataset files from the training session
- `documentation/data-flow-diagram.png`: data flow diagram
- `src/pipeline.py`: ETL script
- `requirements.txt`: project dependencies
- `.env`: input and output file configuration

## Initial Dataset Condition
- The initial dataset contains 205 rows and 31 columns.
- Duplicate rows are present and need to be removed.
- Missing values exist in categorical and numerical columns.
- Some text columns are inconsistent because of differences in whitespace and capitalization.
- The `transaction_date` column is still stored as a string.
- Some numerical columns contain invalid values that should be positive.

## Problems Found
- Duplicate data can bias analysis and modeling results.
- Missing values in important columns need to be handled to avoid losing too many rows.
- Categorical columns cannot be used directly by most machine learning models.
- Numerical features have very different scales and need to be standardized.
- The date column is not yet stored as a datetime value.
- Some columns that should not contain zero or negative values contain values `<= 0`.

## Cleaning Performed and Reasons
- Text was normalized with `strip()` and `lower()` so categorical values are consistent.
- `transaction_date` was converted to datetime so it can support time-based analysis.
- Duplicate rows were removed so records are not counted more than once.
- Mode imputation was applied to `make`, `num-of-doors`, and `horsepower-binned` to preserve the affected rows.
- Median imputation was applied to `stroke`, `horsepower`, and `price` because the median is more resistant to outliers.
- Values `<= 0` in `price`, `horsepower`, `wheel-base`, `engine-size`, and `curb-weight` were replaced with the corresponding column median because they are not realistic values.

## Transformations Performed
- Ordinal encoding was applied to `horsepower-binned`, producing `horsepower_ordinal`.
- `num-of-doors` and `num-of-cylinders` were mapped to numerical values.
- Min-max scaling was applied to the main numerical columns so the features use a consistent scale.
- One-hot encoding was applied to nominal categorical columns such as `body-style`, `drive-wheels`, `aspiration`, `engine-type`, `engine-location`, and `fuel-system`.
- Frequency encoding was applied to `make` because it contains many categories.
- `transaction_date` and `horsepower-binned` were removed after their transformations were completed.

## Example Before and After Transformation
| Column | Before | After |
| --- | --- | --- |
| `num-of-doors` | `two` | `0.0` |
| `num-of-cylinders` | `four` | `0.2` |
| `horsepower-binned` | `Medium` | `horsepower_ordinal = 1` |
| `body-style` | `convertible` | `body-style_convertible = 1` |
| `make` | `alfa-romero` | `make_freq = 0.014925373134328358` |

## Data Size Before and After Processing
- Before processing: 205 rows and 31 columns.
- After processing: 201 rows and 48 columns.
- The 4-row difference is caused by duplicate rows removed during cleaning.

## How to Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run `pipeline.py`
1. Make sure `.env` contains `INPUT_FILE` and `OUTPUT_FILE`.
2. Run the following command:
```bash
python src/pipeline.py
```
3. The processed dataset will be written to the location specified by `OUTPUT_FILE`.

## Brief ETL Flow Explanation
- Extract: read the raw dataset from the CSV file.
- Transform: perform inspection, cleaning, imputation, encoding, and scaling.
- Load: save the final result as the processed dataset.

## Processed Dataset Location
- `data/processed/automobileEDA_cleaned_training.csv`
