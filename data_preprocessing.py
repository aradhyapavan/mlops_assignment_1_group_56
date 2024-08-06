import pandas as pd  # data processing
import matplotlib.pyplot as plt  # plot package
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns  # for data visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import zscore

# Load data
liverdisease_tbl = pd.read_csv('sampledatasets/liver_disease_1.csv')

# Printing first 2 rows of the dataset
# print(forest_tbl.iloc[:2])
liverdisease_tbl.head(2)
liverdisease_tbl.info()
liverdisease_tbl.describe(include="all", percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

# Data processing
# Your data processing code here

# Plot X & Y
liverdisease_tbl.hist(bins=50, figsize=(10, 8))

# Check for null values
liverdisease_tbl.isnull().sum()

# Feature engineering
liverdisease_tbl["X1-high_level"] = (
    liverdisease_tbl["Age"]
    + liverdisease_tbl["Total_Bilirubin"]
    + liverdisease_tbl["Alkaline_Phosphotase"]
    + liverdisease_tbl["Aspartate_Aminotransferase"]
)
print(liverdisease_tbl.columns.to_list())

# Combining important features for X1 and X2
liverdisease_tbl["X2-low_level"] = (
    liverdisease_tbl["Albumin"] + liverdisease_tbl["Total_Protiens"]
)

required_columns = ['Dataset', 'X1-high_level', 'X2-low_level']
liverdisease_df = liverdisease_tbl[required_columns]
print(liverdisease_df)

# Applying z-score normalization
liverdisease_df[['zscore_X1', 'zscore_X2']] = liverdisease_df[['X1-high_level', 'X2-low_level']].apply(zscore)
print(liverdisease_df)

# Define thresholds for outliers
min_X1_zscore_threshold = -0.9
max_X1_zcore_threshold = 4
min_X2_zscore_threshold = -3
max_X2_zcore_threshold = 2.4

# Identify outliers
outliers = liverdisease_df[
    (
        ((liverdisease_df['zscore_X1'] < min_X1_zscore_threshold) |
         (liverdisease_df['zscore_X1'] > max_X1_zcore_threshold))
        |
        ((liverdisease_df['zscore_X2'] < min_X2_zscore_threshold) |
         (liverdisease_df['zscore_X2'] > max_X2_zcore_threshold))
    )
]
print(outliers)
outlier_index = outliers.index
print(outlier_index)

# Drop outliers
liverdisease_df = liverdisease_df.drop(outlier_index)

# Reset the index and drop the old index
liverdisease_df = liverdisease_df.reset_index(drop=True)
liverdisease_df = liverdisease_df[['Dataset', 'X1-high_level', 'X2-low_level']]
print(liverdisease_df)

# Prepare features and target
X = liverdisease_df[['X1-high_level', 'X2-low_level']]
y = liverdisease_df['Dataset']
