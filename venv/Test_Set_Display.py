import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import get_dataset
import cv2
import config
from Save_And_Load_Models import Load_Latest_Model

BATCH_SIZE=8
IMAGE_SIZE=256


dataset_2 = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset_reflectance",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
)
class_names = dataset_2.class_names

train_ds , val_ds , test_ds = get_dataset.get_dataset_partitions(dataset_2)

def predict(model,img):
    img_array = (img)
    img_array = tf.expand_dims(img_array, 0 ) #CREATE A BATCH

    predictions = model.predict(img_array)
    # print("IMGARRAYPREDICT",img,img.dtype,img_array,img_array.dtype)
    # print("BATCH_ID",i)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return  predicted_class , confidence


def Test_Set_Display(test_ds,model):
    figure_no = 0
    print("TETETE",(test_ds))
    for images , labels in test_ds.take(15):
        figure_no+=1
        plt.figure(figure_no,figsize=(18,18))
        for i in range(8):

            # print("IMAGESISISI",images[i].numpy().dtype,"PWSEINAITOIMAGE",images[i].numpy(),"POSAEINAIKSEREGW",len(images[i].numpy()),"SHAPEREGAMW",images[i].numpy().shape)
            ax = plt.subplot(3,3 , i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # print("IMAGGGERNTO",images.numpy().shape)
            predicted_class ,confidence  = predict(model,images[i].numpy())
            actual_class = class_names[labels[i]]

            plt.title(f"Actual:{actual_class}\n Predict: {predicted_class},  Conf: {confidence}%")


            plt.axis("off")
        plt.show()





# Test_Set_Display(test_ds,config.loaded_model)
# Get_Predictions()