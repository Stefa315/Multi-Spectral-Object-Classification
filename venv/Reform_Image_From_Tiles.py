import config
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import matplotlib.pyplot as plt
import Find_Objects
import numpy as np
import os
import cv2
import copy
from Image_Resize import image_resize
import IOU
import csv

# model_current = config.loaded_model
# print("[INFO] loading images...")


def Build_Original_Image_For_Any(path):
	images_zoom_counter=0
	img_Hconc = np.empty((45, 672)).tolist()
	np_img_Hconc = np.empty((0, 1080 * 24))
	np_img_Hconc_transmission = np.empty((0, 1080 * 24))

	np_Vconc = np.empty((23, 45))
	trans_counter = 0 #Transmission Counter
	reflectance_counter = 2  # Reflectance Counter TO KANW 2 GIA NA SYGXRONIZONTAI TA IMREAD
	fluores_counter=7
	polar_counter=3
	fluores_2nd_counter=5
	ii = -1
	y = 0

	# Search In Panel
	for subdir, dirs, files in os.walk(path):
		trans_counter += 1  ## GIA KATHE CUBE +1
		reflectance_counter += 1
		fluores_counter+=1
		fluores_2nd_counter+=1
		polar_counter+=1

		# print("PATHLIST",subdir , files)

		if y <= 43:  # Last Row
			# Find the right subfolder with mod10
			if fluores_counter % 10 == 0:
				# print("PATHLISTFLUOR",subdir, files)
				image_fluores = cv2.imread(os.path.join(subdir, files[0]))
				# cv2.imshow("FLUORES", image_fluores)
				# cv2.waitKey(0)

			if fluores_2nd_counter % 10 == 0:
				# print("PATHLISTFLUO_@",subdir, files)
				image_fluores_2 = cv2.imread(os.path.join(subdir, files[0]))
				# cv2.imshow("FLUORES_2",image_fluores_2)
				# cv2.waitKey(0)
			if polar_counter % 10 == 0:
				# print("PATHLISTPOLAR",subdir, files)
				image_polar = cv2.imread(os.path.join(subdir, files[0]))
				image_polar = image_resize(image_polar, width=1415, height=1184)
				# cv2.imshow("Polar", image_polar)
				# cv2.waitKey(0)
			if (reflectance_counter % 10 == 0):
				# img_copy = image_cont.copy()
				# print("PATHLISTREFLEC",subdir, files)
				image_reflect = cv2.imread(os.path.join(subdir, files[0]))
				image_reflect_copy = image_reflect.copy()
			if (trans_counter % 10 == 0):

				ii += 1  ##GIA KATHE COLUMN +1
				print("ROW {} AND COLUMN {}".format(y, ii), subdir)
				image_trans = cv2.imread(os.path.join(subdir, files[0]))

				image ,Contour_Info= Find_Objects.Object_Segmentation_For_Any_Other(image_trans, image_reflect,image_fluores,image_fluores_2,image_polar,images_zoom_counter)
				print("Corrrrr",Contour_Info,Contour_Info.shape)
				if Contour_Info.size!=0:
					for yy in range(Contour_Info.shape[0]):
						print("CCCCC",Contour_Info[yy])
						with open('Contourinho.csv', 'a', encoding='UTF8', newline='') as f:
							writer = csv.writer(f)
							writer.writerow(Contour_Info[yy])
							f.close()


				if Contour_Info.size!=0:
					if images_zoom_counter==0 :
						Contour_Array = Contour_Info
					else:
						Contour_Array = np.append(Contour_Array,Contour_Info,axis=0)
						print("CONTOUR_ARRAY",Contour_Array)
				images_zoom_counter+=1

				if (trans_counter == 10 or ii == 0):  ##First Image Make a copy so we can concatenate
					image_copy = image.copy()
					np_img_Hconc_transmission = np.array(image_trans).copy()
					# img_Hconc = image.copy()
					np_img_Hconc = image_copy
				# Not first element of a row
				else:
					# Every second row images are feeded backward by default
					if (y % 2 == 1):
						np_img_Hconc = np.concatenate((image, np_img_Hconc), axis=1)
						np_img_Hconc_transmission = np.concatenate((np.array(image_trans), np_img_Hconc_transmission), axis=1)
						print("NP_HCONC", np_img_Hconc.shape)
					else:
						np_img_Hconc = np.concatenate((np_img_Hconc, image), axis=1)
						np_img_Hconc_transmission = np.concatenate((np_img_Hconc_transmission,np.array(image_trans)),axis=1)
						print("SHAPES OF CONCAT",np_img_Hconc.shape, np.array(image).shape, np_img_Hconc_transmission.shape,np.array(image_trans).shape)

						print("NP_HCONC", np_img_Hconc.shape)
					if (ii - 23 == 0):  ## EAN VRISKOMASTE TO TELOS MIAS SEIRAS
						if (y == 0):  ## Y = ROWS EAN VRISKOMASTE STHN PRWTH SEIRA
							np_img_Vconc = np.array(np_img_Hconc).copy()
							np_img_Vconc_transmission = np.array(np_img_Hconc_transmission).copy()
						else:  ## SE OPOIADHPOTE ALLH SEIRA EKTOS THS PRWTHS
							np_img_Vconc = np.array(np.concatenate((np_img_Vconc, np_img_Hconc), axis=0))
							np_img_Vconc_transmission = np.array(np.concatenate((np_img_Vconc_transmission, np_img_Hconc_transmission), axis=0))

						ii = -1
						y += 1
		else:
			break

	# cv2.imshow("Full Image{}||{}".format(ii, y), img_Vconc)
	# cv2.waitKey(0)

	cv2.imwrite("Whole_Image.jpg", np_img_Vconc)




Build_Original_Image_For_Any("../panel_213/area_213")






