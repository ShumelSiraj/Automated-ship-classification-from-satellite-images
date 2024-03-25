import numpy as np
import os
#hide warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to hide all warning messages
import cv2
import pickle
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from sklearn.neighbors import KNeighborsClassifier
from model import VGG16, VGG19, InceptionModel, ResNet50, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV

# image dataset path
code_dir = os.getcwd()
os.chdir("..") # Change to the parent directory
project_dir = os.getcwd()
image_dataset_path = project_dir + os.path.sep + 'Data'

#define the categories
CATEGORIES = ['coast', 'coast_ship', "detail", "land", "multi", "ship", "water"]
num_classes = 7
CHANNELS = 3  # set number of channels to 3 for RGB images

data = []
Image_Size = 100
n_epochs = 50
batch_size = 32
learning_rate = 0.001


def preprocess_data(x, y, force_preprocessing=True):
    # check if preprocessed data exists
    already_preprocessed = os.path.exists('x_train.pkl') and os.path.exists('y_train.pkl') and os.path.exists(
        'x_test.pkl') and os.path.exists('y_test.pkl') and os.path.exists(
        'x_val.pkl') and os.path.exists('y_val.pkl')

    if not already_preprocessed or force_preprocessing:
        # apply image augmentation
        datagen = ImageDataGenerator(rotation_range=45,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True)

        # read and preprocess the images
        for category in CATEGORIES:
            path = os.path.join(image_dataset_path, category)  # path of dataset
            for img in os.listdir(path):
                try:
                    img_path = os.path.join(path, img)  # Getting the image path
                    label = CATEGORIES.index(category)  # Assigning label to image
                    arr = cv2.imread(img_path)  # RGB image
                    new_arr = cv2.resize(arr, (Image_Size, Image_Size))  # Resize image
                    new_arr = datagen.random_transform(new_arr)  # apply image augmentation
                    data.append([new_arr, label])  # appending image and label in list
                except Exception as e:
                    print(str(e))

        for features, label in data:
            x.append(features)  # Storing Images all images in X
            y.append(label)  # Storing al image label in y

        x = np.array(x)  # Converting it into Numpy Array
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)  # 0.125 = 0.1 / 0.8

        # load
        # normalize images
        x_train = x_train / 255
        x_val = x_val / 255
        x_test = x_test / 255

        # reshape images for CNN
        x_train = x_train.reshape(-1, Image_Size, Image_Size, CHANNELS)
        x_val = x_val.reshape(-1, Image_Size, Image_Size, CHANNELS)
        x_test = x_test.reshape(-1, Image_Size, Image_Size, CHANNELS)

        # save preprocessed data to pickle files
        with open('x_train.pkl', 'wb') as f:
            pickle.dump(x_train, f)

        with open('y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)

        with open('x_val.pkl', 'wb') as f:
            pickle.dump(x_val, f)

        with open('y_val.pkl', 'wb') as f:
            pickle.dump(y_val, f)

        with open('x_test.pkl', 'wb') as f:
            pickle.dump(x_test, f)

        with open('y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)

    else:
        # load preprocessed data from pickle files
        x_train = pickle.load(open('x_train.pkl', 'rb'))
        y_train = pickle.load(open('y_train.pkl', 'rb'))
        x_val = pickle.load(open('x_val.pkl', 'rb'))
        y_val = pickle.load(open('y_val.pkl', 'rb'))
        x_test = pickle.load(open('x_test.pkl', 'rb'))
        y_test = pickle.load(open('y_test.pkl', 'rb'))

    return x_train, y_train, x_val, y_val, x_test, y_test

def model_definition(num_classes, learning_rate):
    # define the CNNmodel
    cnn_model = Sequential()
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(7, activation='softmax'))

    return cnn_model

def train_model(model, x_train, y_train, x_val, y_val, n_epochs, batch_size):

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Define early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')

    # train the CNN model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_val, y_val), class_weight=class_weight_dict, callbacks=[early_stop])

    print(model.summary())
    return model

def evaluate(model, x_train, y_train, x_val, y_val, x_test, y_test):
    # evaluate the CNN model
    train_loss, train_acc = model.evaluate(x_train, y_train)  # evaluate train data
    print("Train - Loss:", train_loss, "Accuracy:", train_acc)
    val_loss, val_acc = model.evaluate(x_val, y_val)  # evaluate validation data
    print("Validation - Loss:", val_loss, "Accuracy:", val_acc)
    test_loss, test_acc = model.evaluate(x_test, y_test)  # evaluate test data
    print("Test - Loss:", test_loss, "Accuracy:", test_acc)

    # extract features(Neural Code) from the CNN model
    train_features = model.predict(x_train)
    val_features = model.predict(x_val)
    test_features = model.predict(x_test)

    # define parameter grid for KNN hyperparameters
    param_grid = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    # define randomized search to find best hyperparameters for KNN
    knn_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_iter=10, n_jobs=-1)
    knn_search.fit(train_features, y_train)

    # train and evaluate KNN model with best hyperparameters
    neigh = knn_search.best_estimator_
    neigh.fit(train_features, y_train)

    # evaluate the KNN model using f1 score, recall, and precision
    train_pred = neigh.predict(train_features)
    train_f1 = f1_score(y_train, train_pred, average='weighted')
    train_recall = recall_score(y_train, train_pred, average='weighted')
    train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=1)
    train_acc = accuracy_score(y_train, train_pred)
    print("Evaluation report on train data for {} - F1 score:".format(model.name), train_f1, "Recall:", train_recall, "Precision:", train_precision, "Accuracy:", train_acc)

    val_pred = neigh.predict(val_features)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    val_recall = recall_score(y_val, val_pred, average='weighted')
    val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=1)
    val_accuracy = accuracy_score(y_val, val_pred)
    print("Evaluation report on validation data for {} - F1 score:".format(model.name), val_f1, "Recall:", val_recall, "Precision:", val_precision, "Accuracy:", val_accuracy)

    test_pred = neigh.predict(test_features)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    test_recall = recall_score(y_test, test_pred, average='weighted')
    test_precision = precision_score(y_test, test_pred, average='weighted')
    test_acc = accuracy_score(y_test, test_pred)
    print("Evaluation report on test data for {} - F1 score:".format(model.name), test_f1, "Recall:", test_recall, "Precision:", test_precision, "Accuracy:", test_acc)

if __name__ == "__main__":

    # split data into train and test sets
    x = []
    y = []

    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(x, y)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='VGG16, VGG19, Inception, Resnet, Xception, CNNmodel')
    args = parser.parse_args()
    if args.model == 'VGG16':
        vgg16 = VGG16(num_classes, learning_rate)
        train_model(vgg16, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(vgg16, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'VGG19':
        vgg19 = VGG19(num_classes, learning_rate)
        train_model(vgg19, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(vgg19, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'Inception':
        inception = InceptionModel(num_classes, learning_rate)
        train_model(inception, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(inception, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'Resnet':
        resnet = ResNet50(num_classes, learning_rate)
        train_model(resnet, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(resnet, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'Xception':
        xception = Xception(num_classes, learning_rate)
        train_model(xception, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(xception, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'CNNmodel':
        model = model_definition(num_classes, learning_rate)
        train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(model, X_train, Y_train, X_val, Y_val, X_test, Y_test)