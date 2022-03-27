import config
import tensorflow as tf
from Test_Set_Display import Test_Set_Display
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models , layers
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import model_from_json
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import get_dataset
import cv2

INIT_LR = 1e-4
EPOCHS = 80
BATCH_SIZE = 8
IMAGE_SIZE=256


print("[INFO] loading images...")

dataset_2 = tf.keras.preprocessing.image_dataset_from_directory(
    "../data_Fluorescence_2",
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
)
class_names = dataset_2.class_names

train_ds , val_ds , test_ds = get_dataset.get_dataset_partitions(dataset_2)
print("TRAIN LENGTH",len(train_ds) , len(val_ds),len(test_ds))

train_ex = train_ds.take(256)
train_X=np.empty([257*8,256,256,3])
train_Y = np.empty([258*8])
po=0
po_2=0
for idx,sample in enumerate(train_ex):
        # print("IIII",idx,sample)
        for po in range(len(sample[0])):
            po_2+=1
            train_X[po_2] =  np.array(sample[0][po]).copy()
            train_Y[po_2] = np.array(sample[1][po]).copy()
            po+=1
        po=0
    # plt.imshow(image[:, :, 0].astype(np.uint8), cmap=plt.get_cmap("gray"))
    # plt.title(label)
    # plt.show()

print("IMAGE",len(train_X),train_X.shape)

resize_and_rescale_layer = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)

])

data_augmentation_layer=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),


])




tf.random.set_seed(0)
n_classes=8
input_shape_model= (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3)
# # #
model = models.Sequential([
    keras.Input(shape=(256,256,3)),

    data_augmentation_layer,
    # layers.Conv2D(128, (3,3),activation='relu',input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3)),
    layers.Conv2D(128, (3, 3), input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    # layers.Conv2D(64, (3,3),activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE) ),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    # layers.Conv2D(32, (3,3),activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE) ),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    # layers.Conv2D(64, (3,3),activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE) ),
    layers.Conv2D(128, (3,3)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(512,activation='relu',name='dense_map_fluo_2'),
    layers.Dropout(0.4),
    layers.Dense(n_classes, activation='softmax'),

])

model.build(input_shape=input_shape_model)

print("[INFO] compiling model...")
opt = keras.optimizers.Adam(learning_rate=INIT_LR)
model.compile(
    optimizer=opt,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# train the head of the network
print("[INFO] training head...")
history = model.fit(
    (train_ds),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)
model.summary()

# serialize model to JSON for models with DropOut and BatchNOrmalization

model_version = len( os.listdir("models_saved_json_fluorescence_2"))+1
os.makedirs("../venv/models_saved_json_fluorescence_2/model_{}".format(model_version))
model_json = model.to_json()
with open("../venv/models_saved_json_fluorescence_2/model_{}/model.json".format(model_version),"w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../venv/models_saved_json_fluorescence_2/model_{}/model_weights.h5".format(model_version),overwrite=True)
print("Saved model to disk")

model_version = len( os.listdir("../venv/models_saved_json_fluorescence_2"))
print("MODELVERSION",model_version)
# load json and create model
json_file = open('../venv/models_saved_json_fluorescence_2/model_{}/model.json'.format(model_version), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../venv/models_saved_json_fluorescence_2/model_{}/model_weights.h5".format(model_version))
print("Loaded model from disk")

def predict(model,img):
    img_array = (images[i].numpy())
    img_array = tf.expand_dims(img_array, 0 ) #CREATE A BATCH

    predictions = model.predict(img_array)
    # print("IMGARRAYPREDICT",img,img.dtype,img_array,img_array.dtype)
    # print("BATCH_ID",i)
    Batch_ID=i
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return  predicted_class , confidence , Batch_ID
# print("EEE",test_ds)


Test_Set_Display(test_ds,loaded_model)


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("../venv/models_saved_json_fluorescence_2/model_{}/plot.png".format(model_version))
# plt.imshow()



