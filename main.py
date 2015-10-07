import csv as csv
import numpy as np

csv_file_object = csv.reader(open('Titanic_Data/train.csv', 'rb')) 	
header = csv_file_object.next() 						
data=[] 												

for row in csv_file_object: 							
    data.append(row[0:]) 								
data = np.array(data) 	