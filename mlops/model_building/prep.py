# Data Preparation
# 1. Loads the registered dataset from the repo's data folder
# 2. Cleans it (drops non-predictive columns, fixes the Gender typo)
# 3. Splits into stratified train/test sets
# 4. Saves Xtrain/Xtest/ytrain/ytest as CSVs (passed to the next job as an artifact)
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("mlops/data/tourism.csv")
print("Dataset loaded successfully.")

target = "ProdTaken"

# Drop non-predictive columns: leftover index + unique identifier
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)

# Fix a known data-entry typo in Gender ("Fe Male" -> "Female")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Numeric predictors
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]

# Categorical predictors
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

X = df[numeric_features + categorical_features]
y = df[target]

# Stratified split preserves the (imbalanced) buyer ratio in both sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Train shape:", Xtrain.shape, "| Test shape:", Xtest.shape)
