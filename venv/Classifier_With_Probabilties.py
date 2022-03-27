import matplotlib.pyplot as plt
import numpy as np
from Get_Dense_Output import Get_Dense_Output
import os
import cv2
import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models , layers
from tensorflow.keras.layers import Dropout
from keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd
from IPython.display import display
from keras.models import model_from_json
from Test_Set_Display import Test_Set_Display
from Save_And_Load_Models import Load_Latest_Model
from Save_And_Load_Models import Save_New_Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import csv
import get_dataset

INIT_LR = 1e-4
BATCH_SIZE = 8
IMAGE_SIZE  = 256

dataset_transmission = tf.keras.utils.image_dataset_from_directory(
    "../data_transmission",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)
class_names = dataset_transmission.class_names
file_path = dataset_transmission.file_paths
print("FILEPATH", file_path)
train_trans, val_trans, test_trans = get_dataset.get_dataset_partitions(dataset_transmission)

dataset_reflectance = tf.keras.utils.image_dataset_from_directory(
    "../data_reflectance",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)

train_ref, val_ref, test_ref = get_dataset.get_dataset_partitions(dataset_reflectance)

dataset_polar = tf.keras.utils.image_dataset_from_directory(
    "../data_polar",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)

train_polar, val_polar, test_polar = get_dataset.get_dataset_partitions(dataset_reflectance)

dataset_fluo = tf.keras.utils.image_dataset_from_directory(
    "../data_Fluorescence",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)

train_fluo, val_fluo, test_fluo = get_dataset.get_dataset_partitions(dataset_fluo)

dataset_fluo_2 = tf.keras.utils.image_dataset_from_directory(
    "../data_Fluorescence_2",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)

train_fluo_2, val_fluo_2, test_fluo_2 = get_dataset.get_dataset_partitions(dataset_fluo_2)


def Get_Predictions():

    dataset_transmission = tf.keras.utils.image_dataset_from_directory(
        "../Test_Dataset/Image_Dataset/Transmission",
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )
    class_names = dataset_transmission.class_names
    file_path = dataset_transmission.file_paths
    print("FILEPATH", file_path)

    dataset_reflectance = tf.keras.utils.image_dataset_from_directory(
        "../Test_Dataset/Image_Dataset/Reflectance",
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    dataset_polar = tf.keras.utils.image_dataset_from_directory(
        "../Test_Dataset/Image_Dataset/Polar",
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    dataset_fluo = tf.keras.utils.image_dataset_from_directory(
        "../Test_Dataset/Image_Dataset/Fluorescence",
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    dataset_fluo_2 = tf.keras.utils.image_dataset_from_directory(
        "../Test_Dataset/Image_Dataset/Fluorescence_2",
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )
    model_a = Load_Latest_Model(9,"transmission")
    proba_a = model_a.predict(dataset_transmission)
    print("LENGTH",len(proba_a))
    test_prob = proba_a[2]

    model_b = Load_Latest_Model(6,"reflectance")
    proba_b = model_b.predict(dataset_reflectance)
    test_prob = np.append(test_prob,proba_b[2])

    model_c = Load_Latest_Model(3,"fluorescence")
    proba_c = model_b.predict(dataset_fluo)
    test_prob = np.append(test_prob, proba_c[2])

    model_d = Load_Latest_Model(2,"fluorescence_2")
    proba_d = model_b.predict(dataset_fluo_2)
    test_prob = np.append(test_prob, proba_d[2])

    model_e = Load_Latest_Model(4,"polar")
    proba_e = model_b.predict(dataset_polar)
    test_prob = np.append(test_prob, proba_e[2])

    model_Classifier = Load_Latest_Model(2,"probabilities")
    # print(model_Classifier.predict(np.expand_dims(test_prob,0)))


    return


Get_Predictions()
def Load_Predictions():
    model_trans = Load_Latest_Model(9, "transmission")
    model_reflectance = Load_Latest_Model(6,"reflectance")
    model_fluo = Load_Latest_Model(3,"fluorescence")
    model_fluo_2 = Load_Latest_Model( 2,"fluorescence_2")

    model_polar = Load_Latest_Model(4,"polar")

    a = model_trans.predict(dataset_transmission)
    b = model_reflectance.predict(dataset_reflectance)
    c = model_fluo.predict(dataset_fluo)
    d = model_fluo_2.predict(dataset_fluo_2)
    e = model_polar.predict(dataset_polar)

    return a,b,c,d,e

def Create_CSV(trans_pred, ref_pred , fluo_pred , fluo_2_pred , polar_pred,dataset_transmission):

    header_csv = ['trans1','trans2','trans3','trans4','trans5','trans6','trans7','trans8','ref1','ref2','ref3','ref4','ref5','ref6','ref6','ref7','ref8','fluo1','fluo2','fluo3','fluo4','fluo5','fluo6','fluo7','fluo8','fluo_2_1','fluo_2_2','fluo_2_3','fluo_2_4','fluo_2_5','fluo_2_6','fluo_2_7','fluo_2_8','polar1','polar2','polar3','polar4','polar5','polar6','polar7','polar8','label']
    with open('train.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_csv)
        f.close()


    counter=0
    all_images=np.array([])
    for images,labels in dataset_transmission:
            for i in range(8):
                if i < len(labels):
                    # print("EEEEE",i,counter,len(labels),np.array(labels[i]).astype('int'))
                    csv_line = trans_pred[counter]
                    csv_line = np.append(csv_line,ref_pred[counter])
                    csv_line = np.append(csv_line, fluo_pred[counter])
                    csv_line = np.append(csv_line, fluo_2_pred[counter])
                    csv_line = np.append(csv_line, polar_pred[counter])
                    csv_line = np.append(csv_line,[np.array(labels[i]).astype('int')])
                    if counter ==0:
                        all_images = np.expand_dims(images[i],0)
                    else :
                        all_images = np.append(all_images,np.expand_dims(images[i],0),axis=0)
                    counter+=1
                    print("CSV_LINe",all_images.shape,all_images[0])

                with open('train.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_line)
                    f.close()
    return

def Classifier_With_Probabilities():
    trans_pred, ref_pred, fluo_pred, fluo_2_pred, polar_pred = Load_Predictions()
    # Create_CSV(trans_pred, ref_pred, fluo_pred, fluo_2_pred, polar_pred, dataset_transmission)

    df = pd.read_csv('train.csv')
    dataset = df.values

    X = dataset[:,0:40]
    Y = dataset[:,40]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    # print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

    number_of_classes = 8
    model = Sequential([
        Dense(128, activation='relu', input_shape=(40,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax'),
    ])

    opt = keras.optimizers.Adam(learning_rate=INIT_LR,epsilon=0.1)
    # opt= keras.optimizers.RMSprop(lr=0.001, momentum=0.9,epsilon=0.1)
    model.compile(
        optimizer=opt,
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    hist = model.fit(X_train, Y_train,
                     batch_size=BATCH_SIZE, epochs=300,
                     validation_data=(X_val, Y_val))
    model.evaluate(X_test, Y_test)
    Save_New_Model(model,"probabilities")
    # loaded_model = Load_Latest_Model(1,"probabilities")
    # Test_Set_Display((X_test,Y_test),loaded_model)
    return


Classifier_With_Probabilities()

