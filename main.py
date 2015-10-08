import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
data = pd.read_csv('Data/titanic_full.csv', header=0)

data['nSex'] = data['sex'].map({'female':0, 'male':1}).astype(int)
data['nEmbarked'] = data['embarked'].map({'S':0, 'C':1, 'Q':2})
data['nFamilySize'] = data['sibsp'] + data['parch']


print data[nSex][0:5]
print data[nEmbarked][0:5]
print data[nFamilySize][0:5]