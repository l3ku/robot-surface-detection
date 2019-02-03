import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from extract_features import *

train_data = np.load('X_train_kaggle.npy')

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



if __name__ == '__main__':
	# Get the groups and the respective classes of the samples of those groups.
	groups, classes = get_groups()

	# Split the groups to training and validation data.
	X_train_groups, X_validation_groups, y_train_groups, y_validation_groups = model_selection.train_test_split(groups, classes, test_size=0.2)

	# FIXME: where to utilize this now that we already have a training/validation split?
	ss = model_selection.ShuffleSplit()
	test_indexes = np.array(list(ss.split(X_train_groups)))

	# Transform the data in such a way that the group information is no longer present: just concatenate
	# all values after one another. This also replaces the sample ID's with the actual sample data.
	X_train, y_train = create_data_from_groups(X_train_groups, y_train_groups)
	X_validation, y_validation = create_data_from_groups(X_validation_groups, y_validation_groups)

	# Initialize LDA
	lda = discriminant_analysis.LinearDiscriminantAnalysis()

	# a.) Use ravel features
	F_train = extract_ravel(X_train)
	F_validation = extract_ravel(X_validation)
	lda.fit(F_train, y_train)
	predicted = lda.predict(F_validation)
	print(f'a.) Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')

	# b.) Use mean features
	F_train = extract_mean(X_train)
	F_validation = extract_mean(X_validation)
	lda.fit(F_train, y_train)
	predicted = lda.predict(F_validation)
	print(f'a.) Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')
