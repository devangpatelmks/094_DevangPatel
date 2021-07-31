# CELL 1
# Import necessary library functions
import numpy as np
import pandas as pd

# CELL 2
# Create Two numpy array of size 3 X 2 and 2 X 3.
# Randomly Initalize that array.
narray1 = np.random.randint(100, size=(3,2))
narray2 = np.random.randint(100, size=(2,3))
print(narray1)
print()
print(narray2)

# CELL 3
# Perform matrix multiplication
mul_mat = np.matmul(narray1,narray2)
print(mul_mat)

# CELL 4
# Perform elementwise matrix multiplication
ew_matmul1 = narray1 * narray1
ew_matmul2 = narray2 * 5
print(ew_matmul1)
print()
print(ew_matmul2)

# CELL 5
# Find mean of first matrix
mean_narray1 = np.mean(narray1)
print(mean_narray1)

# CELL 6
mtcars = pd.read_csv('/mtcars.csv')
print(mtcars)

# CELL 7
# Convert Numeric entries(columns) of mtcars.csv to Mean Centered Version
num_col_list = ['mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
num_cols = pd.read_csv('/mtcars.csv', usecols=num_col_list)
meanByCols = np.mean(num_cols, axis=0)
print('Mean:')
print(meanByCols)
print()
print('Mean Centered Version')
for i in num_col_list:
  print(i + ':')
  for j,k in zip(num_cols[i], range(len(meanByCols))):
    print(j - meanByCols[i])
  print()