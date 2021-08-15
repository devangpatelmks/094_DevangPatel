# CELL 0
# Import necessary libraries
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

# CELL 1
# Import data
dataset = pd.read_csv('/Exercise-CarData.csv', na_values = ['??', '????'])
original_data = dataset
print("\nData: \n", dataset)
print("\nData statistics: \n", dataset.describe())

# CELL 2
# Handling missing values
# Method 1: Remove the row with all null values
dataset.dropna(how = 'all', inplace = True)
print(dataset)

# CELL 3
# Handling Categorical Data
le = LabelEncoder()
X = dataset.iloc[:, 4] # FuelType column
Y = dataset.iloc[:, 9] # Door column
dataset.iloc[:, 4] = le.fit_transform(X.astype(str))
dataset.iloc[:, 9] = le.fit_transform(Y.astype(str))

# CELL 4
# Handling missing values
# Method 2: Replace null values with mean value of the corresponding attribute

# Using Imputer function to replace NaN values with mean of that parameter value 
imputer = SimpleImputer(missing_values = np.nan,strategy = "mean")

# Fitting the data, function learns the stats 
imputer = imputer.fit(dataset)

# fit_transform() will execute those stats on the input ie. dataset 
dataset = imputer.transform(dataset)

# filling the missing value with mean 
print("\n\nNew Input with Mean Value for NaN: \n", dataset)

# CELL 5
# Perform scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(dataset)
print("\n\nScaled X : \n", X_scaled)

# CELL 6
# Perform standardization
std = StandardScaler()
X_std = std.fit_transform(dataset)
print("\n\nStandardized X : \n", X_std)

# CELL 7
# Use dummy variables from pandas library to create one column for each FuelType
dummy = pd.get_dummies(original_data['FuelType']) # FuelType
new_dummy = original_data.drop(['FuelType', 'Age', 'KM', 'HP', 'MetColor', 'Automatic', 'CC', 'Doors', 'Weight'],axis=1)
new_dummy = pd.concat([dummy,new_dummy],axis=1)
print("\n\nFinal Data :\n", new_dummy)

# CELL 8
# Feature selection
dataset.head()

# Selecting features based on correlation
corr_data = dataset.iloc[:,:-1]
corr_data = corr_data.corr()
corr_data.head()

# CELL 9
# Generating the correlation heatmap
sns.heatmap(corr_data)

# CELL 10
# Compare the correlation between features and remove features that have a correlation higher than 0.9
columns = np.full((corr_data.shape[0],), True, dtype=bool)
for i in range(corr_data.shape[0]):
    for j in range(i+1, corr_data.shape[0]):
        if corr_data.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = corr_data.columns[columns]
selected_columns.shape

# CELL 11
# Print final columns
data = corr_data[selected_columns]
print(data)