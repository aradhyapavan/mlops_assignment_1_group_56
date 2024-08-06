import pandas as pd
from scipy.stats import zscore

# Load data
liverdisease_tbl = pd.read_csv('sampledatasets/liver_disease_1.csv')

# Data processing
liverdisease_tbl["X1-high_level"] = liverdisease_tbl["Age"] + liverdisease_tbl["Total_Bilirubin"] + liverdisease_tbl["Alkaline_Phosphotase"] + liverdisease_tbl["Aspartate_Aminotransferase"]
liverdisease_tbl["X2-low_level"] = liverdisease_tbl["Albumin"] + liverdisease_tbl["Total_Protiens"]

required_columns = ['Dataset', 'X1-high_level', 'X2-low_level']
liverdisease_df = liverdisease_tbl[required_columns]

# Apply z-score normalization
liverdisease_df[['zscore_X1', 'zscore_X2']] = liverdisease_df[['X1-high_level', 'X2-low_level']].apply(zscore)

# Define outlier thresholds
min_X1_zscore_threshold = -0.9
max_X1_zscore_threshold = 4
min_X2_zscore_threshold = -3
max_X2_zscore_threshold = 2.4

# Identify and remove outliers
outliers = liverdisease_df[
    ((liverdisease_df['zscore_X1'] < min_X1_zscore_threshold) | (liverdisease_df['zscore_X1'] > max_X1_zscore_threshold)) |
    ((liverdisease_df['zscore_X2'] < min_X2_zscore_threshold) | (liverdisease_df['zscore_X2'] > max_X2_zscore_threshold))
]
liverdisease_df = liverdisease_df.drop(outliers.index).reset_index(drop=True)

# Prepare features (X) and target (y)
X = liverdisease_df[['X1-high_level', 'X2-low_level']]
y = liverdisease_df['Dataset']
