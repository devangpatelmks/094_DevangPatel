# CELL 0
# Import necessary libraries

import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

# CELL 1

dataset = pd.read_csv("Lab3/Dataset2.csv")

# CELL 2
# LabelEncoder object

label_enc = preprocessing.LabelEncoder()
Y_rows = None
for data_head in dataset:
    if data_head != "Class":
        print(f"\nHeading :- {data_head}")
        dummy = pd.get_dummies(dataset[data_head])
        dataset = dataset.drop([data_head], axis = 1)
        dataset = pd.concat([dataset, dummy], axis = 1)
    else:
        Y_rows = label_enc.fit_transform(dataset[data_head])
        dataset = dataset.drop([data_head], axis = 1)

# CELL 3
# Print Y_rows

print(dataset, Y_rows)

# CELL 4
# Split the dataset for training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(dataset, Y_rows, test_size = 0.25, random_state = 94)

# CELL 5
# Create model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Predict Y from X_text
Y_predicted = model.predict(X_test)
print(X_test)
print(Y_predicted)

# CELL 6
# Accuracy, Precision and Recall score

from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print(f"Accuracy:- {metrics.accuracy_score(Y_test, Y_predicted)}")
precision = precision_score(Y_test, Y_predicted)
recall = recall_score(Y_test, Y_predicted)

# Print precision and recall
print(f"precision :- {precision}")
print(f"recall :- {recall}")

# CEll 7
# Excersice Task 1

# Case 1
# 0,1,0 -> Rainy Outlook | 0,0,1 -> Mild Temperature | 1,0 -> False Wind | 0,0,1 -> Normal Humidity

# Case 2
# 0,0,1 -> Sunny Outlook | 1,0,0 -> Cool Temperature | 0,1 -> True Wind | 1,0,0 -> High Humidity
 
output1 = model.predict([[0,1,0,  0,0,1,  1,0, 0,0,1]])
output2 = model.predict([[0,0,1,  1,0,0,  0,1,  1,0,0]])
print(f"Final prediction(Case 1): {output1}")
print(f"Final prediction(Case 2): {output2}")