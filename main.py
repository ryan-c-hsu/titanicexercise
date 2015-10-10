import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# For .read_csv, always use header=0 when you know row 0 is the header row
data = pd.read_csv('Data/titanic_full.csv', header=0)

# Basic Cleaning
data['nSex'] = data['sex'].map({'female':0, 'male':1}).astype(int)
data['nEmbarked'] = data['embarked'].map({'S':0, 'C':1, 'Q':2})
data.loc[(data.nEmbarked.isnull())] = 0
data['nFamilySize'] = data['sibsp'] + data['parch']

# Getting Data for Cross Validation
ratioTable = pd.DataFrame({'actual': pd.Series([0,1,2], index=['overall','men','women'])})
ratioTable.actual[1] = float(data.survived.sum())/data.shape[0]
men = data[(data[''])]

# Cleaning for Model 1 (Cleaning Age and Fares)
median_ages = np.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = data[(data['nSex'] == i) & (data['pclass'] == j+1)]['age'].dropna().median()
data['nAge'] = data['age']
for i in range(0,2):
	for j in range(0,3):
		data.loc[(data.age.isnull()) & (data.nSex == i) & (data.pclass == j+1),'nAge'] = median_ages[i,j]

median_fare = np.zeros((1,3))
for i in range(0,1):
	for j in range(0,3):
		median_fare[i,j] = data[(data['pclass'] == j+1) & (data['fare'] != 0)]['age'].median()
data['nFare'] = data['fare']
for i in range(0,1):
	for j in range(0,3):
		data.loc[((data.fare.isnull()) | (data.fare == 0)) & (data.pclass == j+1),'nFare'] = median_fare[i,j]

# data cleaning for model 2 (Converted Ages, Fares into bins)

# data cleaning for model 3 (Added First letter of last name)

traindata = data[['survived','nSex','nEmbarked','nFamilySize','nAge','nFare','pclass']]
traindata = traindata.astype(int)
traindata = traindata.values

testdata = data[['nSex','nEmbarked','nFamilySize','nAge','nFare','pclass']]
testdata = testdata.astype(int)
testdata = testdata.values

forest = RandomForestClassifier(n_estimators = 1300)
forest = forest.fit(traindata[0::,1::],traindata[0::,0])
output = forest.predict(testdata)

output = pd.DataFrame(output)
output['Gender'] = data.nSex

print output[0].sum()
print output.shape[0]
ratio1 = float(output[0].sum())/float(output.shape[0])
print ratio
print ratio1

predictions_file = open("model1.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Survived"])
open_file_object.writerows(zip(output))
predictions_file.close()

