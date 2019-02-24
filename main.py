import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from extract_features import *
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical
from gene_testing import *
from tensorflow import set_random_seed


if __name__ == '__main__':
	train_data = np.load('X_train_kaggle.npy')
	all_id_classes = np.genfromtxt('y_train_final_kaggle.csv',delimiter=',',dtype='str')
	groups_csv = np.genfromtxt('groups.csv',delimiter=',',dtype='str')
	le = preprocessing.LabelEncoder()
	le.fit(all_id_classes[:,1])
	all_id_classes_transformed = le.transform(all_id_classes[:,1])
	classes_array = np.array(all_id_classes_transformed)

	"""
	# Split the groups to training and validation data.
	gss = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2)
	data_split = gss.split(groups_csv[:, 0], groups_csv[:, 2], groups_csv[:, 1])

	# Initialize LDA
	lda1 = discriminant_analysis.LinearDiscriminantAnalysis()
	lda2 = discriminant_analysis.LinearDiscriminantAnalysis()
	lda3 = discriminant_analysis.LinearDiscriminantAnalysis()

	LR1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
	LR2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
	LR3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

	RF1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
	RF2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
	RF3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

	clf_list = [lda1, lda2, lda3, SVC(), SVC(), SVC(), SVC(kernel="linear"), SVC(kernel="linear"), SVC(kernel="linear"),\
				LR1, LR2, LR3, RF1, RF2, RF3]

	# Defining the neural network
	n_filters = 10
	len_kernel = 7
	model = Sequential()
	model.add(Conv1D(n_filters, len_kernel, input_shape=(10,128), activation='relu', data_format='channels_first'))
	#model.add(Conv1D(n_filters, len_kernel, activation='relu', data_format='channels_first'))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Conv1D(int(n_filters/2), len_kernel, activation='relu', data_format='channels_first'))
	#model.add(Conv1D(int(n_filters/2), len_kernel, activation='relu', data_format='channels_first'))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dropout(rate=0.4))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(9, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	"""

	amount_of_inviduals = 7
	inv_list = [Genetic_invidual([10, 2, 0.4, 64, 64, 30, 1, 1], 0.3, 0.7)\
     for i in range(amount_of_inviduals)]

	rounds = 10000
	error = False

	gss = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2)
	data_split = gss.split(groups_csv[:, 0], groups_csv[:, 2], groups_csv[:, 1])
	for i in range(rounds):

		for tr, ts in data_split:
			train = tr
			test = ts

		for j in inv_list:
			n_filters = int(j.params[0]+1)
			len_kernel = int(j.params[1]*2+1)
			if len_kernel > 128:
				len_kernel = 128

			np.random.seed(42)
			set_random_seed(24)

			model = Sequential()
			model.add(Conv1D(n_filters, len_kernel, input_shape=(3,128), activation='relu', data_format='channels_first'))
			error = False
			for k in range(int(j.params[6])):
				model.add(Conv1D(int(n_filters), len_kernel, activation='relu', data_format='channels_first'))
			model.add(MaxPooling1D(pool_size=2))

			#model.add(Conv1D(n_filters, len_kernel, activation='relu', data_format='channels_first'))
			#model.add(MaxPooling1D(pool_size=2))
			#model.add(Conv1D(int(n_filters/2), len_kernel, activation='relu', data_format='channels_first'))
			#model.add(Conv1D(int(n_filters/2), len_kernel, activation='relu', data_format='channels_first'))
			#model.add(MaxPooling1D(pool_size=2))
			try:
				model.add(Flatten())
			except:
				print("Ja ny se flatteni veti persiilleen...")
				i -= 1
				error = True
				break
			model.add(Dropout(rate=j.params[2]))
			model.add(Dense(int(j.params[3]+1), activation='relu'))
			for k in range(int(j.params[7])):
				model.add(Dense(int(j.params[3]+1), activation='relu'))
			model.add(Dense(9, activation='softmax'))

			model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

			y_train = to_categorical(classes_array[train])
			y_validation = classes_array[test]
			F_train = np.array([i[7:] for i in train_data[train]])
			F_validation = np.array([i[7:] for i in train_data[test]])

			#print(np.array([i[7:] for i in train_data[train]]).shape)

			try:
				model.fit(F_train, y_train, batch_size=int(j.params[4]+1), epochs=int(j.params[5]+1), verbose=0)
				prediction = model.predict(F_validation)
				prediction = np.argmax(prediction, axis=1)
				j.fitness = metrics.accuracy_score(y_validation, prediction)
				print("Accuracy =", j.fitness, "with params:", j.params)
			except:
				print("Virhe parametreilla:", j.params)
				error = True
				i -= 1
				break
		
		if error == True:
			error = False
		else:
			print("Cycle:", i, "  The top fitness:", max([j.fitness for j in inv_list]))
			inv_list = survival_of_the_fittest(inv_list, 0.4)
	
	"""
	# Feature data
	ravel_data = np.array(extract_ravel(train_data))
	mean_data = np.array(extract_mean(train_data))
	var_mean_data = np.array(extract_var_mean(train_data))

	score_list = []
	
	round = 0
	for train, test in data_split:
		y_train = classes_array[train]
		y_validation = classes_array[test]

		# a.) Use ravel features
		F_train = ravel_data[train]
		F_validation = ravel_data[test]
		clf_list[0].fit(F_train, y_train)
		predicted = clf_list[0].predict(F_validation)
		print(f'a.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[3].fit(F_train, y_train)
		predicted = clf_list[3].predict(F_validation)
		print(f'aa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[6].fit(F_train, y_train)
		predicted = clf_list[6].predict(F_validation)
		print(f'aaa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[9].fit(F_train, y_train)
		predicted = clf_list[9].predict(F_validation)
		print(f'aaaa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[12].fit(F_train, y_train)
		predicted = clf_list[12].predict(F_validation)
		print(f'aaaaa.) Round {round}: Accuracy with np.ravel(): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))



		# b.) Use mean features
		F_train = mean_data[train]
		F_validation = mean_data[test]
		F_train = np.array(F_train)
		F_validation = np.array(F_validation)

		clf_list[1].fit(F_train, y_train)
		predicted = clf_list[1].predict(F_validation)
		print(f'b.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[4].fit(F_train, y_train)
		predicted = clf_list[4].predict(F_validation)
		print(f'bb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[7].fit(F_train, y_train)
		predicted = clf_list[7].predict(F_validation)
		print(f'bbb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[10].fit(F_train, y_train)
		predicted = clf_list[10].predict(F_validation)
		print(f'bbbb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[13].fit(F_train, y_train)
		predicted = clf_list[13].predict(F_validation)
		print(f'bbbbb.) Round {round}: Accuracy with np.mean(axis=1): {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		# c.) Use mean and variance features
		F_train = var_mean_data[train]
		F_validation = var_mean_data[test]
		F_train = np.array(F_train)
		F_validation = np.array(F_validation)

		clf_list[2].fit(F_train, y_train)
		predicted = clf_list[2].predict(F_validation)
		print(f'c.) Round {round}: Accuracy with mean and var: {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[5].fit(F_train, y_train)
		predicted = clf_list[5].predict(F_validation)
		print(f'cc.) Round {round}: Accuracy with mean and var: {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[8].fit(F_train, y_train)
		predicted = clf_list[8].predict(F_validation)
		print(f'ccc.) Round {round}: Accuracy with mean and var: {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[11].fit(F_train, y_train)
		predicted = clf_list[11].predict(F_validation)
		print(f'cccc.) Round {round}: Accuracy with mean and var: {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		clf_list[14].fit(F_train, y_train)
		predicted = clf_list[14].predict(F_validation)
		print(f'ccccc.) Round {round}: Accuracy with mean and var: {metrics.accuracy_score(y_validation, predicted)}')
		score_list.append(metrics.accuracy_score(y_validation, predicted))

		round += 1
		print("Best score:", max(score_list))
		m = max(score_list)
		print("Index for the classifier:", [i for i, j in enumerate(score_list) if j == m])
	
	

	F_train = var_mean_data
	y_train = classes_array
	clf_list[14].fit(F_train, y_train)

	final_data = np.load("X_test_kaggle.npy")
	var_mean_data = np.array(extract_var_mean(final_data))
	predicted = clf_list[14].predict(var_mean_data)

	labels = list(le.inverse_transform(predicted))

	with open("submission.csv", "w") as fp:
		fp.write("# Id,Surface\n")

		for i, label in enumerate(labels):
			fp.write("%d,%s\n" % (i, label))
	"""