import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from extract_features import *
from sklearn.svm import SVC, NuSVC, LinearSVC


if __name__ == '__main__':
	train_data = np.load('X_train_kaggle.npy')
	all_id_classes = np.genfromtxt('y_train_final_kaggle.csv',delimiter=',',dtype='str')
	groups_csv = np.genfromtxt('groups.csv',delimiter=',',dtype='str')
	le = preprocessing.LabelEncoder()
	le.fit(all_id_classes[:,1])
	all_id_classes_transformed = le.transform(all_id_classes[:,1])
	classes_array = np.array(all_id_classes_transformed)

	train_data = extract_statistical(train_data)
	final_data = np.load("X_test_kaggle.npy")
	y_train = classes_array

	parameters = {'gamma':np.logspace(-5,-1), 'C':np.logspace(0,1,10)}

	clf = model_selection.GridSearchCV(SVC(kernel='linear'), parameters, verbose=2)
	clf.fit(train_data, y_train)

	print(clf.best_score_, clf.best_params_)	

	"""
	labels = list(le.inverse_transform(prediction))

	with open("submission.csv", "w") as fp:
		fp.write("# Id,Surface\n")

		for i, label in enumerate(labels):
			fp.write("%d,%s\n" % (i, label))
	"""