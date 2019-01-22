import numpy as np
import pandas as pd
from pprint import pprint

train_data = np.load('X_train_kaggle.npy')

all_id_classes = pd.read_csv('y_train_final_kaggle.csv')


# Checking the amount of data given
# Most likely not useful in the actual solution, just visualizing the data
# for the beginning of the project
class_amount = {}
for i in all_id_classes['Surface']:
	class_amount.setdefault(i, 0)
	class_amount[i] += 1

print('The amount of data for all classes:')
pprint(class_amount)


# Tagging the data with the ID
classified_data = {}
for i in range(len(train_data)):
	data_class = (all_id_classes['Surface'][i])
	classified_data.setdefault(data_class, {})
	for j in range(len(train_data[i])):
		classified_data[data_class].setdefault(j, [])
		classified_data[data_class][j].append(train_data[i][j])


# Calculating the class statistics
class_statistics = {}
for i in classified_data:
	class_statistics.setdefault(i, {})
	for j in classified_data[i]:
		class_statistics[i][j] = [np.mean(classified_data[i][j]),\
		 np.var(classified_data[i][j]), np.max(classified_data[i][j]),\
		 np.min(classified_data[i][j])]

pprint(class_statistics)
