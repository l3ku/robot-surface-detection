"""
Extracts data in 3 ways:
extract_ravel just ravels the full data and returns it.
extract_mean calculates mean and returns the means instead of the full data.
extract_var_mean calculates both variances and means and returns them.

Meant to be used with just the X_data.
------------
TODO:
Maybe cleaning?
"""

import numpy as np

def extract_ravel(data):
    """
    Ravels the data and returns it

    Parameters
    ----------
    data : list
      The X_data from the train_test_split function made from the blocks

    Returns
    -------
    raveled_data : list
      The same structure as the X_data, just the single sample layer raveled
      so that the 10 sensors' data is in one long vector
    """
    raveled_data = []
    for i in range(len(data)):
        raveled_data.append(np.ravel(data[i]))

    return raveled_data


def extract_mean(data):
    """
    Calculates mean from the data and returns it

    Parameters
    ----------
    data : list
      The X_data from the train_test_split function made from the blocks

    Returns
    -------
    mean_data : list
      The same structure as the X_data, just the single sample layer full sensor
      data replaced with the means
    """
    mean_data = []
    for i in range(len(data)):
        mean_data.append(np.mean(data[i], axis=1))
    return mean_data


def extract_var_mean(data):
    """
    Calculates variance and mean from the data and returns it

    Parameters
    ----------
    data : list
      The X_data from the train_test_split function made from the blocks

    Returns
    -------
    var_mean_data : list
      The same structure as the X_data, just the single sample layer full sensor
      data replaced with the variances and means
    """
    var_mean_data = []
    for i in range(len(data)):
        var_mean_data.append((np.var(data[i]), np.mean(data[i])))

    return var_mean_data


def extract_chanel_var_mean(data):
    """
    Calculates variance and mean from the data for each chanel and returns it

    Parameters
    ----------
    data : list
      The X_data from the train_test_split function made from the blocks

    Returns
    -------
    var_mean_data : list
      The same structure as the X_data, just the single sample layer full sensor
      data replaced with the variances and means of chanel
    """
    chanel_var_mean = []
    for i in range(0,len(data)):
        chanel_var_mean.append([])

        for j in range(0,len(data[0])):
            chanel_var_mean[i].append((np.var(data[i][j]), np.mean(data[i][j])))

    return chanel_var_mean



if __name__ == '__main__':
    import pandas as pd
    from pprint import pprint
    import matplotlib.pyplot as plt
    from sklearn import preprocessing, model_selection

    train_data = np.load('X_train_kaggle.npy')
    all_id_classes = pd.read_csv('y_train_final_kaggle.csv')
    groups = pd.read_csv('groups.csv')

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

    X_train, X_validation, y_train_str, y_validation_str =\
     model_selection.train_test_split(blocks_list, blocks_classes_list, test_size=0.2)
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

    pprint(extract_var_mean(X_train))
