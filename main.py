import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
data = pd.read_csv('Data/titanic_full.csv', header=0)

df['nSex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
df['nEmbarked'] = df['embarked'].map({'S':0, 'C':1, 'Q':2})
df['nFamilySize'] = df['sibsp'] + df['parch']


