import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/drive/My Drive/Colab Notebooks/ML-Lab/L1/Data_for_Transformation.csv')

#Scatter plot of ‘Age’ vs ‘Salary’.
plt.scatter(data['Age'],data['Salary'])
plt.show()

# Plot a histogram to check the frequency distribution of the variable ‘Salary’.
plt.hist(data['Salary'],bins=5)
plt.show()

# Creating the bar plot for 'Country'.
plt.bar(data['Country'], color='red', height=10)
plt.show()