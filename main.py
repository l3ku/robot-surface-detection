import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from extract_features import *
from sklearn.svm import SVC


if __name__ == '__main__':
	train_data = np.load('X_train_kaggle.npy')
	#all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
	all_id_classes = np.genfromtxt('y_train_final_kaggle.csv',delimiter=',',dtype='str')
	#groups_csv = pd.read_csv('groups.csv').values
	groups_csv = np.genfromtxt('groups.csv',delimiter=',',dtype='str')
	le = preprocessing.LabelEncoder()
	le.fit(all_id_classes[:,1])
	all_id_classes_transformed = le.transform(all_id_classes[:,1])
	classes_array = np.array(all_id_classes_transformed)

	# Split the groups to training and validation data.
	gss = model_selection.GroupShuffleSplit(n_splits=10, test_size=0.2)
	data_split = gss.split(groups_csv[:, 0], groups_csv[:, 2], groups_csv[:, 1])

	# Initialize LDA
	lda1 = discriminant_analysis.LinearDiscriminantAnalysis()
	lda2 = discriminant_analysis.LinearDiscriminantAnalysis()
	lda3 = discriminant_analysis.LinearDiscriminantAnalysis()

	SVC_list = [SVC(), SVC(), SVC(), SVC(kernel="linear"), SVC(kernel="linear"), SVC(kernel="linear")]

	# Feature data
	ravel_data = np.array(extract_ravel(train_data))
	mean_data = np.array(extract_mean(train_data))
	var_mean_data = np.array(extract_var_mean(train_data))
	
	round = 0
	for train, test in data_split:
		X_train = train_data[train]
		y_train = classes_array[train]
		X_validation = train_data[test]
		y_validation = classes_array[test]

		# a.) Use ravel features
		F_train = ravel_data[train]
		F_validation = ravel_data[test]
		lda1.fit(F_train, y_train)
		predicted = lda1.predict(F_validation)
		print(f'a.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')

		SVC_list[0].fit(F_train, y_train)
		predicted = SVC_list[0].predict(F_validation)
		print(f'aa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')

		SVC_list[3].fit(F_train, y_train)
		predicted = SVC_list[3].predict(F_validation)
		print(f'aaa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')

		LR1=LogisticRegression(random_state=0, solver='lbfgs',multi_class = 'multinomial').fit(F_train, y_train)
		predicted = LR1.predict(F_validation)
		print(f'aaaa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')

		RF1=RandomForestClassifier(n_estimators=100, max_depth=5,random_state = 0).fit(F_train, y_train)
		predicted = RF1.predict(F_validation)
		print(f'aaaaa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')



		# b.) Use mean features
		F_train = mean_data[train]
		F_validation = mean_data[test]
		F_train = np.array(F_train)
		F_validation = np.array(F_validation)

		lda2.fit(F_train, y_train)
		predicted = lda2.predict(F_validation)
		print(f'b.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')

		SVC_list[1].fit(F_train, y_train)
		predicted = SVC_list[1].predict(F_validation)
		print(f'bb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')

		SVC_list[4].fit(F_train, y_train)
		predicted = SVC_list[4].predict(F_validation)
		print(f'bbb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')

		LR1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(F_train, y_train)
		predicted = LR1.predict(F_validation)
		print(f'bbbb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')

		RF1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0).fit(F_train, y_train)
		predicted = RF1.predict(F_validation)
		print(f'bbbbb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')

		# c.) Use mean and variance features
		F_train = var_mean_data[train]
		F_validation = var_mean_data[test]
		F_train = np.array(F_train)
		F_validation = np.array(F_validation)

		lda3.fit(F_train, y_train)
		predicted = lda3.predict(F_validation)
		print(f'c.) Round {round}: Accuracy with mean and variance: {metrics.accuracy_score(y_validation, predicted)}')

		SVC_list[2].fit(F_train, y_train)
		predicted = SVC_list[2].predict(F_validation)
		print(f'cc.) Round {round}: Accuracy with mean and variance: {metrics.accuracy_score(y_validation, predicted)}')

		SVC_list[5].fit(F_train, y_train)
		predicted = SVC_list[5].predict(F_validation)
		print(f'ccc.) Round {round}: Accuracy with mean and variance: {metrics.accuracy_score(y_validation, predicted)}')

		LR1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(F_train, y_train)
		predicted = LR1.predict(F_validation)
		print(f'cccc.) Round {round}: Accuracy with mean and variance: {metrics.accuracy_score(y_validation, predicted)}')

		RF1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0).fit(F_train, y_train)
		predicted = RF1.predict(F_validation)
		print(f'ccccc.) Round {round}: Accuracy with mean and variance: {metrics.accuracy_score(y_validation, predicted)}')

		round += 1
