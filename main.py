import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing

train_data = np.load('X_train_kaggle.npy')
all_id_classes = pd.read_csv('y_train_final_kaggle.csv')

le = preprocessing.LabelEncoder()
le.fit(all_id_classes['Surface'])
all_id_classes_transformed = le.transform(all_id_classes['Surface'])
all_id_classes['Surface'] = all_id_classes_transformed

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

plt_a1 = np.concatenate(classified_data[6][0])
plt_a2 = np.concatenate(classified_data[6][7])

plt_b1 = np.concatenate(classified_data[3][0])
plt_b2 = np.concatenate(classified_data[3][7])

plt.figure()
plt.plot(plt_b1, plt_b2, "b+")
plt.plot(plt_a1, plt_a2, "r+")
plt.show()


#plt(np.concatenate(classified_data["hard_tiles"][8]),\
#    np.concatenate(classified_data["hard_tiles"][9]), linestyle="None",\
#    marker="+", color="b")
