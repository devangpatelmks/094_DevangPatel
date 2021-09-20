# CELL 0

# Import necessary libraries
import io
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# CELL 1

# Load dataset
dataset = pd.read_csv('BuyComputer.csv')

# CELL 2

dataset.drop(columns = ['User ID',], axis=1, inplace = True)
dataset.head()

# CELL 3

# Declare label as last column
label = dataset['Purchased']
print(label)

# CELL 4

# Declaring X as every column except last
X = dataset[['Age', 'EstimatedSalary']]
print(X)

# CELL 5

# Splitting data for testing and training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size = 0.25, random_state = 94)
X_train, X_test, y_train, y_test

# Scaling data
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
X_train = std_sc.fit_transform(X_train)
X_test = std_sc.transform(X_test)

# Variabes to calculate sigmoid function
y_pred = []
len_x = len(X_train[0])
w = []
b = 0.2
print(len_x)

# CELL 6

entries = len(X_train[:, 0])
print(entries)

for weights in range(len_x):
    w.append(0)
print(w)

# CELL 7

# Sigmoid function
def sigmoid(z):
  return 1/(1+np.exp(-z))

def predict(inputs, w):
    z=np.dot(inputs, w)
    return sigmoid(z)

# CELL 8

# Loss function
def loss_func(features, labels, w):
    observations = len(labels)
    predictions = predict(features, w)
    class1_cost = -labels*np.log(predictions)
    class2_cost = (1-labels)*np.log(1-predictions)
    cost = class1_cost - class2_cost
    cost = cost.sum() / observations
    return cost

dw = []
db = 0
J = 0
alpha = 0.1
for x in range(len_x):
    dw.append(0)


# CELL 9

def update_weights(features, labels, weights, lr):    
    
    N = len(features)    
    predictions = predict(features, weights)    
    gradient = np.dot(features.T,  predictions - labels)    
    gradient /= N    
    gradient *= lr    
    weights -= gradient

    return weights

# CELL 10

# Repeating the process 3000 times
cost_history = []
for i in range(3000):
    w = update_weights(X, label, w, alpha)

    # Calculating error for auditing purposes
    cost = loss_func(X, label, w)
    cost_history.append(cost)

    # Log Progress
    if i % 1000 == 0:
        print("iter: " + str(i) + " cost: " +str(cost))

# CELL 11

# Print weight and bias
print(w)
print(b)

# CELL 12

# Predicting the label
predicted_labels = predict(X_test, w)

# Print actual and predicted values in a table
print(predicted_labels, label)

# CELL 13

# Calculating accuracy of prediction
diff = predicted_labels - y_test
arcy = 1.0 - (float(np.count_nonzero(diff)) / len(diff))
print(arcy)

# CELL 14

# Using sklearn LogisticRegression model

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 94)

# Fit
LR = LR.fit(X_train,y_train)
# Predicting the test label with LR. Predict always takes X as input
y_pred = LR.predict(X_test)
print(y_pred)