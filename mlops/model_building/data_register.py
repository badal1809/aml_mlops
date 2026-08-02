# Data Registration
# 1. Loads the raw tourism dataset from the repo's data folder
# 2. Validates that all expected columns are present (fails loudly if not)
# 3. Prints a summary (rows, columns, target balance)
import pandas as pd

RAW_PATH = "mlops/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Expected schema - the dataset is only "registered" if these columns exist
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print("Columns:", list(df.columns))
print("Target (ProdTaken) distribution:")
print(df["ProdTaken"].value_counts())
