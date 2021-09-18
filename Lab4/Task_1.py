# CELL 0
# Import necessary libraries

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import metrics

# CELL 1
# Prepare dataset

# Predictor variables
Outlook = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
            'Rainy', 'Rainy', 'Sunny', 'Rainy','Overcast', 'Overcast', 'Sunny']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
Wind = ['False', 'True', 'False', 'False', 'False', 'True', 'True',
            'False', 'False', 'False', 'True', 'True', 'False', 'True']

# Class Label
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# CELL 2
# Digitize the dataset using encoding

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
Outlook_encoded = le.fit_transform(Outlook)
Outlook_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Outlook mapping: ",Outlook_name_mapping)

Temperature_encoded = le.fit_transform(Temperature)
Temperature_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Temperature mapping: ",Temperature_name_mapping)

Humidity_encoded = le.fit_transform(Humidity)
Humidity_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Humidity mapping: ",Humidity_name_mapping)

Wind_encoded = le.fit_transform(Wind)
Wind_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Wind mapping: ",Wind_name_mapping)

Play_encoded = le.fit_transform(Play)
Play_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Play mapping: ",Play_name_mapping)

print("\n")
print("Weather: ", Outlook_encoded)
print("Temerature: ", Temperature_encoded)
print("Humidity: ", Humidity_encoded)
print("Wind: ", Wind_encoded)
print("Play: ", Play_encoded)


# CELL 3
# Split the data

Weather = pd.DataFrame(Outlook_encoded, columns = ['Outlook'])
Weather['Temperature'] = pd.DataFrame(Temperature_encoded, columns = ['Temperature'])
Weather['Humidity'] = pd.DataFrame(Humidity_encoded, columns = ['Humidity'])
Weather['Wind'] = pd.DataFrame(Wind_encoded, columns = ['Wind'])
Weather['Play'] = pd.DataFrame(Play_encoded, columns = ['Play'])
print(f"\n{Weather}")

from sklearn.model_selection import train_test_split
X=Weather.values[:,0:4]
Y=Weather.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.25, random_state = 94)

# CELL 4
# Create a Decision Tree Classifier (using Gini)

clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 27, max_depth = 7, min_samples_leaf = 25)

clf_gini.fit(X_train, y_train)

# CELL 5
# Predict the classes of test data

y_pred = clf_gini.predict(X_test)
print("Predicted values: ")
print(y_pred)

# CELL 6
# Model accuracy

print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred))
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100)
print("Report: ", metrics.classification_report(y_test, y_pred))