import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection

train_data = np.load('X_train_kaggle.npy')
all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
groups = pd.read_csv('groups.csv')

def create_data_from_groups(X, y):
	"""
	Transform a list containing groups of sample ID's to a list that contains consecutive samples with
	actual sample data. For example: if we have a list group_data = [[1, 2, 3], [4, 6, 9]], the first
	group (group_data[0]) will contain samples with ID's 1, 2 and 3, and the second group with ID's
	4, 6 and 9. However, for classification purposes, we only want to have the following format:
		X: [[SAMPLE_1_DATA], [SAMPLE_2_DATA], [SAMPLE_3_DATA], ..., [SAMPLE_N_DATA]]
		y: [[SAMPLE_1_CLASS], [SAMPLE_2_CLASS], [SAMPLE_3_CLASS], ..., [SAMPLE_N_CLASS]]

	This function will create that type of data with returning a list of sample data (X_filtered)
	along with the classes that represent the samples in the respective indices.

	Parameters
	----------
	x : list
		A list of lists; each inner list contains the sample ID's of samples in a group.
	y : list
		A list of lists; each inner list contains the classes of the respective samples in `x`.

	Returns
	-------
	X_filtered : list
		A consecutive list of samples (data).
	y_filtered : list
		The classes corresponding to the samples in X_filtered.
	"""
	X_filtered = []
	y_filtered = []
	for i in range(len(X)):
		n_samples_in_group = len(X[i])
		for j in range(n_samples_in_group):
			sample_id = X[i][j]
			sample_data = train_data[j].tolist()
			sample_surface = y[i][j]
			X_filtered.append(sample_data)
			y_filtered.append(sample_surface)

	return (X_filtered, y_filtered)


def get_groups():
	"""
	Get the data in "groups.csv". The groups are returned by indices [0-36] in a list, and each those
	indices contain a list of sample ID's belonging to that group in the return value "groups". The
	"groups_classes" return value contains a similar structure, but instead of sample ID's, it contains
	the classes for the sample ID's in "groups".

	The idea is that the return value of this function can be directly passed to create_data_from_groups()
	after doing a train/test split, which will replace the sample ID's with the actual data.

	Returns
	-------
	groups : list
		A list of lists; each inner list contains the sample ID's of samples in a group.
	groups_classes : list
		A list of lists; each inner list contains the classes of the respective samples in `groups`.
	"""
	all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
	groups_csv = pd.read_csv('groups.csv')

	le = preprocessing.LabelEncoder()
	le.fit(all_id_classes['Surface'])
	all_id_classes_transformed = le.transform(all_id_classes['Surface'])
	all_id_classes['Surface'] = all_id_classes_transformed

	# In blocks_list, we have the samples mapped to each group by e.g. groups_list[5][0] being the first
	# sample to belong to group 5. groups_classes_list contains the surface ID's for each sample, respectively.
	groups = [list() for _ in range(36)]
	groups_classes = [list() for _ in range(36)]
	for value in groups_csv.values:
		sample_id = value[0]
		group_id = value[1]
		surface_name = value[2]
		surface_id = le.transform([surface_name])[0]
		groups[group_id].append(sample_id)
		groups_classes[group_id].append(surface_id)

	return (groups, groups_classes)


le = preprocessing.LabelEncoder()
le.fit(all_id_classes['Surface'])
all_id_classes_transformed = le.transform(all_id_classes['Surface'])
all_id_classes['Surface'] = all_id_classes_transformed

blocks_list = [list() for _ in range(36)]
blocks_classes_list = [-1 for _ in range(36)]

for value in groups.values:
	id = value[0]
	group_id = value[1]
	surface = value[2]
	blocks_list[group_id].append(train_data[id])
	if blocks_classes_list[group_id] == -1:
		blocks_classes_list[group_id] = surface

X_train, X_validation, y_train_str, y_validation_str = model_selection.train_test_split(blocks_list, blocks_classes_list, test_size=0.2)
y_train = le.transform(y_train_str)
y_validation = le.transform(y_validation_str)

ss = model_selection.ShuffleSplit()
test_indexes = np.array(list(ss.split(X_train)))


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
