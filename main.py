import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from sklearn.ensemble import RandomForestClassifier
from extract_features import *
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Model params
    n_estimators = 800
    max_depth = 5
    criterion = 'entropy'

    train_data = np.load('X_train_kaggle.npy')
    #all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
    all_id_classes = np.genfromtxt('y_train_final_kaggle.csv',delimiter=',',dtype='str')
    #groups_csv = pd.read_csv('groups.csv').values
    groups_csv = np.genfromtxt('groups.csv',delimiter=',',dtype='str')
    le = preprocessing.LabelEncoder()
    le.fit(all_id_classes[:,1])
    all_id_classes_transformed = le.transform(all_id_classes[:,1])
    classes_array = np.array(all_id_classes_transformed)

    # Feature data
    statistical_features = np.array(extract_statistical(train_data))

    ## Split the groups to training and testing data. The testing data should only
    # be used in the final evaluation of the model and thus never included in
    # training.
    number_of_splits = 50
    gss = model_selection.GroupShuffleSplit(n_splits=number_of_splits, test_size=0.2, random_state=0)
    data_split = gss.split(groups_csv[:, 0], le.transform(groups_csv[:, 2]), groups_csv[:, 1])

    clf_name_list = ['RandomForestClassifier()']
    score_list = [[] for i in range(len(clf_name_list))]

    round = 1
    for train, test in data_split:
        clf_list = [
            RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, criterion=criterion, bootstrap=False, n_jobs=-1)
        ]
        print(f'======== ROUND {round} =========')
        y_train = classes_array[train]
        y_validation = classes_array[test]
        F_train = statistical_features[train]
        F_validation = statistical_features[test]

        for i, clf in enumerate(clf_list):
            clf.fit(F_train, y_train)
            # clf = RFECV(classifier, step=1, cv=5, n_jobs=-1, verbose=2)
            # clf = clf.fit(F_train, y_train)
            # print(clf.support_)
            predicted = clf.predict(F_validation)
            score = metrics.accuracy_score(y_validation, predicted)
            print(f'{clf_name_list[i]} scored {score}...')
            score_list[i].append(score)

        round += 1

    average_scores = np.mean(score_list, axis=1)
    best_classifier = np.argmax(average_scores)
    best_score = average_scores[best_classifier]
    print(f'{clf_name_list[best_classifier]} got the best average score with {best_score}')
    print(f'{clf_name_list[best_classifier]} variance: {np.var(score_list)}')
    # for i in range(len(score_list)):
    #     plt.plot(list(range(len(score_list[i]))), score_list[i], label=clf_name_list[i])
    # plt.xlabel('Split')
    # plt.ylabel('Accuracy')
    # plt.title(f'{clf_name_list[best_classifier]} got the best average score with {best_score}')
    # plt.show()

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, criterion=criterion, bootstrap=False, n_jobs=-1)
    clf.fit(statistical_features, classes_array)

    final_data = np.load("X_test_kaggle.npy")
    statistical_features = np.array(extract_statistical(final_data))
    predicted = clf.predict(statistical_features)

    labels = list(le.inverse_transform(predicted))

    with open("submission.csv", "w") as fp:
        fp.write("# Id,Surface\n")

        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
