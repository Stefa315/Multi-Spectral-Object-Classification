import os
import tensorflow as tf
from tensorflow.keras import models
from os import listdir
from keras.models import model_from_json

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories

#LOAD MODEL_JSON

json_file = open('../venv/models_saved_json_classifier/model_7/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../venv/models_saved_json_classifier/model_7/model_weights.h5")




img_labels_str = os.listdir("../dataset_transmission")

BASE_PATH_new = "../dataset_3"



BLOOD_PATH_3 =  os.path.sep.join([BASE_PATH_new, "../dataset_2/Blood"])
GLASS_PATH_3 = os.path.sep.join([BASE_PATH_new, "../dataset_2/Glass"])
SAND_PATH_3 = os.path.sep.join([BASE_PATH_new, "../dataset_2/Sand"])
FIBER_PATH_3 = os.path.sep.join([BASE_PATH_new, "../dataset_2/Fiber"])
HAIR_PATH_3 = os.path.sep.join([BASE_PATH_new, "../dataset_3/Hair"])

# print("AAAA",img_labels_str)
ORIG_BASE_PATH_3 = "../items_3"
ORIG_IMAGES_3 = os.path.sep.join([ORIG_BASE_PATH_3, "../items_3/all_images_3"])
ORIG_ANNOTS_3 = os.path.sep.join([ORIG_BASE_PATH_3, "../items_3/annotations_3"])

ORIG_BASE_PATH = "../items"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "../all_images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "../Annotations"])
# originaltarxidiamoy = os.listdir("../all_images")
# print("EAISIXTIR",ORIG_IMAGES,originaltarxidiamoy)
# model_curr = models.load_model("../saved_models/6")
ENCODER_PATH = "label_encoder.pickle"

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "../dataset_1"
BLOOD_PATH =  os.path.sep.join([BASE_PATH, "../dataset_1/Blood"])
GLASS_PATH = os.path.sep.join([BASE_PATH, "../dataset_1/Glass"])
SAND_PATH = os.path.sep.join([BASE_PATH, "../dataset_1/Sand"])
FIBER_PATH = os.path.sep.join([BASE_PATH, "../dataset_1/Fiber"])
HAIR_PATH = os.path.sep.join([BASE_PATH, "../dataset_1/Hair"])



ORIG_BASE_PATH_SVM = "../items_SVM"
ORIG_IMAGES_SVM = os.path.sep.join([ORIG_BASE_PATH_3, "../items_SVM/all_images"])
ORIG_ANNOTS_SVM = os.path.sep.join([ORIG_BASE_PATH_3, "../items_SVM/annotations"])
NO_OBJ_PATH_SVM = os.path.sep.join([BASE_PATH, "../SVM_dataset/No_Object"])




POSITIVE_PATH = os.path.sep.join([BASE_PATH, "../dataset_1/object_path"])
NO_OBJECT_PATH = os.path.sep.join([BASE_PATH, "../dataset_1/no_object_path"])
# print("FFFFFF",POSITIVE_PATH)

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 500
MAX_PROPOSALS_INFER = 300

MAX_POSITIVE = 500
MAX_NEGATIVE = 30
MAX_BLOOD = 1000
MAX_FIBER = 1000
MAX_SAND = 1000
MAX_HAIR = 1000
MAX_GLASS=1000

# initialize the input dimensions to the network
INPUT_DIMS = (256, 256)
INPUT_DIMS_3 = (707,592)

# define the path to the output model and label binarizer
MODEL_ = tf.keras.models.load_model("../saved_models/1")

ENCODER_PATH = "label_encoder.pickle"
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99

BASE_PATH_SVM = "../SVM_dataset"


