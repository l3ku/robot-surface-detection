from random import randint
import numpy as np
from sklearn import preprocessing, model_selection, discriminant_analysis, metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GaussianDropout, GlobalAveragePooling2D
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from extract_features import *

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

if __name__ == '__main__':
    """
    I have now tested with:
        - Xception with 3D image features => 0.64
        - Deep LSTM model with raw features => 0.61
        - Shallow LSTM model with raw features => BAD
        - Xception model with low epochs => BAD
    """
    # Model tuning params
    num_splits = 10
    epochs = 50
    show_images = 0

    #Load Data
    train_data = np.load('X_train_kaggle.npy')#[:, 6:, :]
    all_id_classes = np.genfromtxt('y_train_final_kaggle.csv', delimiter=',', dtype='str')
    groups_csv = np.genfromtxt('groups.csv', delimiter=',', dtype='str')

    # Encode labels
    le = preprocessing.LabelEncoder()
    le.fit(all_id_classes[:, 1])
    all_id_classes_transformed = le.transform(all_id_classes[:, 1])
    classes_array = np.array(all_id_classes_transformed)

    # Normalize samples
    for i in range(train_data.shape[1]):
        transformer = preprocessing.Normalizer().fit(train_data[:, i, :])
        transformer.transform(train_data[:, i, :])

    # Transform labels to n x 9 vectors
    target_classes = to_categorical(classes_array)

    # Feature Data
    img_3d_features = np.array(extract_as_3d_image(train_data))
    for i in range(0, show_images):
        image_index = randint(0, img_3d_features.shape[0])
        plt.imshow(img_3d_features[image_index])
        plt.title(f'Class: {le.inverse_transform([classes_array[image_index]])[0]}')
        plt.show()
    #img_3d_features = train_data

    # Split the groups to training and testing data. The testing data should only
    # be used in the final evaluation of the model and thus never included in
    # training.
    scores = []
    gss = model_selection.GroupShuffleSplit(n_splits=num_splits, test_size=0.1)
    split = gss.split(groups_csv[:, 0], le.transform(groups_csv[:, 2]), groups_csv[:, 1])
    round = 0
    for tr, ev in split:
        print("\n==============================================================")
        print(f"=======================   SPLIT {round+1}/{num_splits}   =======================")
        print("==============================================================\n")
        F_train = img_3d_features[tr]
        y_train = target_classes[tr]
        training_groups = np.array(groups_csv[:, 1])[tr]
        F_test = np.array(img_3d_features[ev])
        y_test = target_classes[ev]

        print(f'Using a total of {len(tr)} groups for training, and {len(ev)} for final evaluation after training the model...')

        #LSTM Structure for raw training data
        # model = Sequential()
        # model.add(LSTM(512, input_shape=train_data.shape[1:], go_backwards=True))
        # model.add(Dense(512))
        # model.add(GaussianDropout(0.2))
        # model.add(Dense(100))
        # model.add(Dense(target_classes.shape[1], activation='sigmoid'))
        # model.summary()

        # # Test model that is based on XCeption
        base_model = Xception(include_top=False, weights='imagenet', input_shape=img_3d_features.shape[1:])

        # Specify only the layers after 100 as trainable
        for layer in base_model.layers:
            layer.trainable = True
        w = base_model.output
        w = GlobalAveragePooling2D()(w)
        w = Dense(500)(w)
        w = GaussianDropout(0.1)(w)
        w = Dense(100)(w)
        output = Dense(target_classes.shape[1], activation='sigmoid')(w)
        model = Model(inputs=[base_model.input], outputs=[output])

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        # callbacks = [
        #     EarlyStopping(monitor='val_acc', patience=30),
        #     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
        # ]
        model.fit(F_train, y_train, validation_data=(F_test, y_test),
                      epochs=epochs, verbose=2)

        score = model.evaluate(F_test, y_test)
        scores.append(score[1])
        print(f"Accuracy of current round: {score[1]}")
        round += 1

    # Show results
    #plt.plot(list(range(len(scores))), scores)
    #plt.xlabel('Split')
    #plt.ylabel('Accuracy')
    #plt.title(f'Mean accuracy for all measurements: {np.mean(scores)}')
    #plt.legend(loc = 'best')
    #plt.show()
    print(f'Accuracies: {scores}')
    print(f'Mean accuracy: {np.mean(scores)}')
    final_data = np.load("X_test_kaggle.npy")
    img_3d_final_data = np.array(extract_as_3d_image(final_data))
    #img_3d_final_data = final_data
    predicted_categorical = model.predict(img_3d_final_data)
    predicted = np.argmax(predicted_categorical, axis=1)
    labels = list(le.inverse_transform(predicted))

    with open("submission3.csv", "w+") as fp:
        fp.write("# Id,Surface\n")

        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
