import numpy as np
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras import regularizers
from extract_features import *


if __name__ == '__main__':

    #Load Data
    train_data = np.load('X_train_kaggle.npy')
    all_id_classes = np.genfromtxt('y_train_final_kaggle.csv', delimiter=',', dtype='str')
    groups_csv = np.genfromtxt('groups.csv', delimiter=',', dtype='str')


    le = preprocessing.LabelEncoder()
    le.fit(all_id_classes[:, 1])
    all_id_classes_transformed = le.transform(all_id_classes[:, 1])
    classes_array = np.array(all_id_classes_transformed)

    #Transform labels to n x 9 vectors
    target_classes = to_categorical(classes_array)


    # Split the groups to training and validation data.
    gss = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2)
    data_split = gss.split(groups_csv[:, 0], groups_csv[:, 2], groups_csv[:, 1])


    #Feature Data
    ravel_data = np.array(extract_ravel(train_data))
    mean_data = np.array(extract_mean(train_data))
    var_mean_data = np.array(extract_var_mean(train_data))
    chanel_var_mean = np.array(extract_chanel_var_mean(train_data))


    #Reshape mean data from (1703, 10) to (1703, 10, 1)
    mean_data = mean_data.reshape([int(len(mean_data)),10,1])
    var_mean_data = var_mean_data.reshape(int(len(var_mean_data)),2,1)

    weight_l1 = 0.001

    #LSTM Structure for raw training data
    #input Shape
    model = Sequential()
    model.add(LSTM(64, return_sequences= True, input_shape=train_data.shape[1:],activity_regularizer= regularizers.l1(weight_l1)))
    model.add(LSTM(64, activity_regularizer= regularizers.l1(weight_l1)))
    model.add(Dense(target_classes.shape[1], activation='softmax', activity_regularizer= regularizers.l1(weight_l1)))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #LSTM Structure for mean data
    model2 = Sequential()
    model2.add(LSTM(64, return_sequences= True, input_shape=mean_data.shape[1:], activity_regularizer= regularizers.l1(weight_l1)))
    model2.add(LSTM(64, activity_regularizer= regularizers.l1(weight_l1)))
    model2.add(Dense(target_classes.shape[1], activation='softmax', activity_regularizer= regularizers.l1(weight_l1)))
    model2.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #LSTM Structure for mean and var data
    model3 = Sequential()
    model3.add(LSTM(64, return_sequences= True, input_shape=var_mean_data.shape[1:], activity_regularizer= regularizers.l1(weight_l1)))
    model3.add(LSTM(64, activity_regularizer= regularizers.l1(weight_l1)))
    model3.add(Dense(target_classes.shape[1], activation='softmax', activity_regularizer= regularizers.l1(weight_l1)))
    model3.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    #LSTM Structure for chanel mean and var data

    model4 = Sequential()
    model4.add(LSTM(64, return_sequences= True, input_shape=chanel_var_mean.shape[1:], activity_regularizer= regularizers.l1(weight_l1)))
    model4.add(LSTM(64, activity_regularizer= regularizers.l1(weight_l1)))
    model4.add(Dense(target_classes.shape[1], activation='softmax', activity_regularizer= regularizers.l1(weight_l1)))
    model4.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    round = 0
    scores = []
    scores2 = []
    scores3 = []
    scores4 = []

    epochs = 20
    for train, test in data_split:

        #RAW DATA

        #Get raw training data and labels with correct indices
        F_train = train_data[train]
        y_train = target_classes[train]

        #Split test into test and validation data
        test_input, validation_input, test_target, validation_target = train_test_split(train_data[test],
                                                                                        target_classes[test],
                                                                                        test_size=0.33)

        model.fit(F_train,y_train, validation_data=(validation_input,validation_target), epochs=epochs)

        score = model.evaluate(test_input,test_target)
        scores.append(score[1])


        #MEAN DATA

        F_train = mean_data[train]
        y_train = target_classes[train]

        test_input, validation_input, test_target, validation_target = train_test_split(mean_data[test],
                                                                                        target_classes[test],
                                                                                        test_size=0.33)

        model2.fit(F_train, y_train, validation_data=(validation_input, validation_target), epochs=epochs)

        score2 = model2.evaluate(test_input, test_target)
        scores2.append(score2[1])


        #VAR AND MEAN DATA

        F_train = var_mean_data[train]
        y_train = target_classes[train]

        test_input, validation_input, test_target, validation_target = train_test_split(var_mean_data[test],
                                                                                        target_classes[test],
                                                                                        test_size=0.33)

        model3.fit(F_train, y_train, validation_data=(validation_input, validation_target), epochs=epochs)

        score3 = model3.evaluate(test_input, test_target)
        scores3.append(score3[1])


        #CHANEL VAR AND MEAN DATA

        F_train = chanel_var_mean[train]
        y_train = target_classes[train]

        test_input, validation_input, test_target, validation_target = train_test_split(chanel_var_mean[test],
                                                                                        target_classes[test],
                                                                                        test_size=0.33)

        model4.fit(F_train, y_train, validation_data=(validation_input, validation_target), epochs=epochs)

        score4 = model4.evaluate(test_input, test_target)
        scores4.append(score4[1])

    print(scores)
    print(scores2)
    print(scores3)
    print(scores4)