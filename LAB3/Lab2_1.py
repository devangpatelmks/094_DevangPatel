# CELL 0
# Import necessary libraries
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# CELL 1
# Prepare dataset
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy',
           'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy',
           'Sunny', 'Overcast', 'Overcast', 'Rainy']

temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
        'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
        'Mild', 'Mild', 'Hot', 'Mild']

play = ['No', 'No', 'Yes', 'Yes', 'Yes',
      'No', 'Yes', 'No', 'Yes', 'Yes',
      'Yes', 'Yes', 'Yes', 'No']


# CELL 2
# Digitize the data set using encoding

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers
weather_encoded = le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print("Weather:" , weather_encoded)
print("Temp:", temp_encoded)
print("Play:", label)


# CELL 3
# Merge different features to prepare dataset

# Combinig weather and temp into single list of tuples
features=tuple(zip(weather_encoded, temp_encoded))
print("Features: ", features)

# CELL 4
# Train ’Naive Bayes Classifier’

# Create a Classifier
model = MultinomialNB()
# Train the model using the training sets
model.fit(features, label)

# CELL 5
# Predict Output for new data
predicted = model.predict([[2,1]]) # 2: Sunny, 1: Hot
print("Predicted Value:", predicted)

# CELL 6
# Exercise:
# Manually calculate output for the following cases and compare it with system’s output.

# (1) Will you play if the temperature is 'Hot' and weather is 'overcast'?
# (2) Will you play if the temperature is 'Mild' and weather is 'Sunny'?

pred1 = model.predict([[0, 0]]) # 0: Overcast, 1: Hot
pred2 = model.predict([[2, 2]]) # 2: Sunny, 2: Mild
print("Predicted value when,")
print("1) Temperature is Hot and Weather is Overcast: ", pred1)
print("2) Temperature is Mild and Weather is Sunny: ", pred2)

# CELL 7
# Manual Calculation

pyes = 9/14
pno = 5/14

# Case 1 (Hot, Overcast)
vnby1 = pyes * (4/9) * (2/9)
vnbn1 = pno * (0) * (2/5)

ny1 = vnby1/(vnby1+vnbn1)
nn1 = vnbn1/(vnbn1+vnby1)

# Case 2 (Mild, Sunny)
vnby2 = pyes * (2/9) * (4/9)
vnbn2 = pno * (3/5) * (2/5)

ny2 = vnby2/(vnby2+vnbn2)
nn2 = vnbn2/(vnbn2+vnby2)

print("The probability that I will play in case 1: ", ny1)
print("The probability that I will play in case 2: ", ny2)

# CELL 8
# Exercise conclusion

print("Upon comparing the values of manual calculation and system's calculation,")
print("it has been noted that system's calculated value for case 2 differs from")
print("manual calculation.")