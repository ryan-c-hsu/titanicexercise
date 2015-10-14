import pandas as pd
import pylab as P
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# For .read_csv, always use header=0 when you know row 0 is the header row
data = pd.read_csv('Data/titanic_full.csv', header=0)

# Basic Cleaning
data['nSex'] = data['sex'].map({'female':0, 'male':1}).astype(int)
data['nEmbarked'] = data['embarked'].map({'S':0, 'C':1, 'Q':2})
data.loc[(data.nEmbarked.isnull())] = 0
data['nFamilySize'] = data['sibsp'] + data['parch']

actual = data[['survived','nSex','nEmbarked','pclass']]

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

# Creating model 1 
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

# data cleaning for model 2 (Converted Ages (every 10 years), Fares into bins (0-15,15-30,30+), Family Size into bins (0-3,4-6,7-10))
data['bAge'] = data['nAge']
data.loc[(data.nAge <= 10),'bAge'] = 1;
data.loc[(data.nAge > 10) & (data.nAge <= 20),'bAge'] = 2;
data.loc[(data.nAge > 20) & (data.nAge <= 30),'bAge'] = 3;
data.loc[(data.nAge > 30) & (data.nAge <= 40),'bAge'] = 4;
data.loc[(data.nAge > 40) & (data.nAge <= 50),'bAge'] = 5;
data.loc[(data.nAge > 50),'bAge'] = 6;

data['bFare'] = data['nFare']
data.loc[(data.nFare <= 15),'bFare'] = 1;
data.loc[(data.nFare > 15) & (data.nFare <= 30),'bFare'] = 2;
data.loc[(data.nFare > 30),'bFare'] = 3;

data['bFamilySize'] = data['nFamilySize']
data.loc[(data.nFamilySize <= 3),'bFamilySize'] = 1;
data.loc[(data.nFamilySize > 3) & (data.nFamilySize <= 6),'bFamilySize'] = 2;
data.loc[(data.nFamilySize > 7),'bFamilySize'] = 3;

#creating model 2
model2traindata = data[['survived','nSex','nEmbarked','bFamilySize','bAge','bFare','pclass']]
model2traindata = model2traindata.astype(int)
model2traindata = model2traindata.values

model2testdata = data[['nSex','nEmbarked','bFamilySize','bAge','bFare','pclass']]
model2testdata = model2testdata.astype(int)
model2testdata = model2testdata.values

model2forest = RandomForestClassifier(n_estimators = 1300)
model2forest = model2forest.fit(model2traindata[0::,1::],model2traindata[0::,0])
model2output = pd.DataFrame(model2forest.predict(model2testdata))
model2output['nSex'] = data[['nSex']]
model2output['nEmbarked'] = data[['nEmbarked']]
model2output['pclass'] = data[['pclass']]

ratioTable['model2'] = float(0)
ratioTable.model2[0] = float(model2output[0].sum())/model2output.shape[0]
ratioTable.model2[1] = float(model2output[(model2output['nSex'] == 1)][0].sum())/model2output.shape[0]
ratioTable.model2[2] = float(model2output[(model2output['nSex'] == 0)][0].sum())/model2output.shape[0]
ratioTable.model2[3] = float(model2output[(model2output['pclass'] == 1)][0].sum())/model2output.shape[0]
ratioTable.model2[4] = float(model2output[(model2output['pclass'] == 2)][0].sum())/model2output.shape[0]
ratioTable.model2[5] = float(model2output[(model2output['pclass'] == 3)][0].sum())/model2output.shape[0]
ratioTable.model2[6] = float(model2output[(model2output['nEmbarked'] == 0)][0].sum())/model2output.shape[0]
ratioTable.model2[7] = float(model2output[(model2output['nEmbarked'] == 1)][0].sum())/model2output.shape[0]
ratioTable.model2[8] = float(model2output[(model2output['nEmbarked'] == 2)][0].sum())/model2output.shape[0]

# Creating model 3 (Separated Family size to Sibling/Spouses and Parents/Children)
model3traindata = data[['survived','nSex','nEmbarked','parch','sibsp','bAge','bFare','pclass']]
model3traindata = model3traindata.astype(int)
model3traindata = model3traindata.values

model3testdata = data[['nSex','nEmbarked','parch','sibsp','bAge','bFare','pclass']]
model3testdata = model3testdata.astype(int)
model3testdata = model3testdata.values

model3forest = RandomForestClassifier(n_estimators = 1300)
model3forest = model3forest.fit(model3traindata[0::,1::],model3traindata[0::,0])
model3output = pd.DataFrame(model3forest.predict(model3testdata))
model3output['nSex'] = data[['nSex']]
model3output['nEmbarked'] = data[['nEmbarked']]
model3output['pclass'] = data[['pclass']]

ratioTable['model3'] = float(0)
ratioTable.model3[0] = float(model3output[0].sum())/model3output.shape[0]
ratioTable.model3[1] = float(model3output[(model3output['nSex'] == 1)][0].sum())/model3output.shape[0]
ratioTable.model3[2] = float(model3output[(model3output['nSex'] == 0)][0].sum())/model3output.shape[0]
ratioTable.model3[3] = float(model3output[(model3output['pclass'] == 1)][0].sum())/model3output.shape[0]
ratioTable.model3[4] = float(model3output[(model3output['pclass'] == 2)][0].sum())/model3output.shape[0]
ratioTable.model3[5] = float(model3output[(model3output['pclass'] == 3)][0].sum())/model3output.shape[0]
ratioTable.model3[6] = float(model3output[(model3output['nEmbarked'] == 0)][0].sum())/model3output.shape[0]
ratioTable.model3[7] = float(model3output[(model3output['nEmbarked'] == 1)][0].sum())/model3output.shape[0]
ratioTable.model3[8] = float(model3output[(model3output['nEmbarked'] == 2)][0].sum())/model3output.shape[0]

print ratioTable

print 'Model 1 Confusion Matrix'
print confusion_matrix(actual['survived'], model1output[0])
print 'Model 2 Confusion Matrix'
print confusion_matrix(actual['survived'], model2output[0])
print 'Model 3 Confusion Matrix'
print confusion_matrix(actual['survived'], model3output[0])


predictions_file = open("model1.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Survived"])
open_file_object.writerows(zip(model1output))
predictions_file.close()

predictions_file = open("model2.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Survived"])
open_file_object.writerows(zip(model2output))
predictions_file.close()

predictions_file = open("model3.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Survived"])
open_file_object.writerows(zip(model3output))
predictions_file.close()

