import os
import cv2





def Rename_Files_Sequentially(path):



    for subdir, dirs, files in os.walk(path):
        y=0
        for i in files:
            os.rename(os.path.join(subdir,i),"{}/".format(subdir)+os.path.basename(subdir)+"_{}".format(y)+".jpg")
            y+=1

# Rename_Files_Sequentially("../dataset_1")