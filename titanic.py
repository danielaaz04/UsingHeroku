#Importing libraries
import pandas as pd
import numpy as np

#Loading files
def load_file(file):
    return pd.read_csv(file)

print("Loading data")

train_df = load_file('data/train.csv')
test_df = load_file('data/test.csv')


#Delete irrelevant variables like name, passenger id and ticket number. Delete Cabin because it has a 77% of missing values
#from both train and test datasets.
#FOR THE PURPOSE OF DEVELOPING ON FLASK, I WILL ERASE ALSO CATEGORICAL VARIABLES TO AVOID ENCODING AND MAKE THE UNDERSTANDING OF FLASK MORE SIMPLE.
#Cleaning

train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin","Sex","Embarked"], axis= 1)
test_df = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin","Sex","Embarked"], axis= 1)

def replace_with_avg(df, col):
    average = df[col].mean(axis=0)
    print("The average is:" , average)
    df[col].replace(np.nan, average , inplace = True)
    print("Replacing missing values with average:", average)

replace_with_avg(train_df, "Age")

train_target = train_df["Survived"]
train_features = train_df.drop(["Survived"], axis = 1)

#Select algorithm
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
#Training
clf_rf.fit(train_features, train_target)

#Transformations to my test set
age_average = train_df['Age'].mean(axis=0)
print("The average is:" , age_average)

test_df['Age'].replace(np.nan, age_average , inplace = True)
print("Replacing missing values in test dataset with average:", age_average)

fare_average = train_df['Fare'].mean(axis=0)
print("The average is:" , fare_average)

test_df['Fare'].replace(np.nan, fare_average , inplace = True)
print("Replacing missing values in test dataset with average:", fare_average)


predictions = clf_rf.predict(test_df)


 #Save the model to disk

import pickle

filename = "titanic_model.pkl"

pickle.dump(clf_rf, open(filename, 'wb'))
