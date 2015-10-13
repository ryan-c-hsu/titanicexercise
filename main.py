import pandas as pd
import pylab as P
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
ratioTable = pd.DataFrame({'actual': pd.Series([float(0)], index=['overall','men','women','class 1','class 2','class 3','embarked S', 'embarked C', 'embarked Q'])})
ratioTable.actual[0] = float(data.survived.sum())/data.shape[0]
ratioTable.actual[1] = float(data[(data['nSex'] == 1)]['survived'].sum())/data.shape[0]
ratioTable.actual[2] = float(data[(data['nSex'] == 0)]['survived'].sum())/data.shape[0]
ratioTable.actual[3] = float(data[(data['pclass'] == 1)]['survived'].sum())/data.shape[0]
ratioTable.actual[4] = float(data[(data['pclass'] == 2)]['survived'].sum())/data.shape[0]
ratioTable.actual[5] = float(data[(data['pclass'] == 3)]['survived'].sum())/data.shape[0]
ratioTable.actual[6] = float(data[(data['nEmbarked'] == 0)]['survived'].sum())/data.shape[0]
ratioTable.actual[7] = float(data[(data['nEmbarked'] == 1)]['survived'].sum())/data.shape[0]
ratioTable.actual[8] = float(data[(data['nEmbarked'] == 2)]['survived'].sum())/data.shape[0]

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

model1traindata = data[['survived','nSex','nEmbarked','nFamilySize','nAge','nFare','pclass']]
model1traindata = model1traindata.astype(int)
model1traindata = model1traindata.values

model1testdata = data[['nSex','nEmbarked','nFamilySize','nAge','nFare','pclass']]
model1testdata = model1testdata.astype(int)
model1testdata = model1testdata.values

model1forest = RandomForestClassifier(n_estimators = 1300)
model1forest = model1forest.fit(model1traindata[0::,1::],model1traindata[0::,0])
model1output = pd.DataFrame(model1forest.predict(model1testdata))
model1output['nSex'] = data[['nSex']]
model1output['nEmbarked'] = data[['nEmbarked']]
model1output['nFamilySize'] = data[['nFamilySize']]
model1output['nAge'] = data[['nAge']]
model1output['nFare'] = data[['nFare']]
model1output['pclass'] = data[['pclass']]

ratioTable['model1'] = float(0)
ratioTable.model1[0] = float(model1output[0].sum())/model1output.shape[0]
ratioTable.model1[1] = float(model1output[(model1output['nSex'] == 1)][0].sum())/model1output.shape[0]
ratioTable.model1[2] = float(model1output[(model1output['nSex'] == 0)][0].sum())/model1output.shape[0]
ratioTable.model1[3] = float(model1output[(model1output['pclass'] == 1)][0].sum())/model1output.shape[0]
ratioTable.model1[4] = float(model1output[(model1output['pclass'] == 2)][0].sum())/model1output.shape[0]
ratioTable.model1[5] = float(model1output[(model1output['pclass'] == 3)][0].sum())/model1output.shape[0]
ratioTable.model1[6] = float(model1output[(model1output['nEmbarked'] == 0)][0].sum())/model1output.shape[0]
ratioTable.model1[7] = float(model1output[(model1output['nEmbarked'] == 1)][0].sum())/model1output.shape[0]
ratioTable.model1[8] = float(model1output[(model1output['nEmbarked'] == 2)][0].sum())/model1output.shape[0]

# data cleaning for model 2 (Converted Ages (every 5 years), Fares into bins (0-15,15-30,30+))
data['bAge'] = data['nAge']
data.loc[(data.nAge <= 10),'bAge'] = 1;
# data cleaning for model 3 (Added First letter of last name)

print model1output[0].sum()
print model1output.shape[0]
ratio1 = float(model1output[0].sum())/float(model1output.shape[0])
print ratio1
print ratioTable

data['nFare'].hist(bins=16, range=(0,80), alpha = .5)
#P.show()

predictions_file = open("model1.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Survived"])
open_file_object.writerows(zip(model1output))
predictions_file.close()

