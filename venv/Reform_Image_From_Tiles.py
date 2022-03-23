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



def Build_Original_Image(path,model_current):

	img_Hconc = np.empty((45,672)).tolist()
	np_img_Hconc = np.empty((0, 1080*24))
	np_Vconc = np.empty((23,45))
	i=0
	ii = -1
	y = 0

	#Search In Panel
	for subdir, dirs, files in os.walk(path):
		i+=1  ## GIA KATHE CUBE +
		if y<=43: #Last Row
			#Find the right subfolder with mod10
			if (i%10 ==0):
				ii += 1   ##GIA KATHE COLUMN +1
				print("ROW {} AND COLUMN {}".format( y, ii) , subdir)
				image = cv2.imread(os.path.join(subdir, files[0]))
				image = Find_Objects.Object_Segmentation(image)
				if (i==10 or ii==0):  ##First Image Make a copy so we can concatenate
					image_copy = image.copy()
					# img_Hconc = image.copy()
					np_img_Hconc = image_copy
					# img_Hconc[y][ii] = image.copy() #111
				#Not first element of a row
				else :
					#Every second row images are feeded backward by default
					if (y%2==1):
						# img_Hconc[y][ii] = cv2.hconcat([image,img_Hconc[y][ii - 1]])#111
						np_img_Hconc = np.concatenate((image,np_img_Hconc),axis=1)
						print("NP_HCONC",np_img_Hconc.shape)
						# cv2.imshow("IKETEUW",np_img_Hconc)
						# cv2.waitKey(0)
					else :
						# img_Hconc[y][ii]  =  cv2.hconcat([img_Hconc[y][ii-1],image])##111
						np_img_Hconc = np.concatenate((np_img_Hconc,image),axis=1)
					if (ii-23==0): ## EAN VRISKOMASTE TO TELOS MIAS SEIRAS
						if (y==0) :  ## Y = ROWS EAN VRISKOMASTE STHN PRWTH SEIRA
							# img_Vconc = img_Hconc[y][ii].copy() #111
							np_img_Vconc = np.array(np_img_Hconc).copy()
							# cv2.imshow("IKETEUW_2", np.array(img_Hconc[y][ii]))
							# cv2.waitKey(0)
						else :   ## SE OPOIADHPOTE ALLH SEIRA EKTOS THS PRWTHS
							np_img_Vconc = np.array(np.concatenate((np_img_Vconc,np_img_Hconc),axis=0))
							# img_Vconc = cv2.vconcat([img_Vconc, img_Hconc[y][ii]])#111
							# cv2.imshow("IKETEUW_2", np_img_Vconc)
							# cv2.waitKey(0)
						ii=-1
						y+=1
		else :
			break
	# cv2.imshow("Full Image{}||{}".format(ii, y), img_Vconc)
	# cv2.waitKey(0)
	cv2.imwrite("Whole_Image.jpg",np_img_Vconc)


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
				# cv2.imshow("DD",image)
				# cv2.waitKey(0)

				# color = (255, 0, 0)
				# thickness=2
				# if np.array(Contour_Info).size != 0 and Contour_Info.shape[0]>1:
				# 	print("MIN__X_R_D1")
				# 	for cont_num in range(Contour_Info.shape[0]-1):
				# 		whole_box = np.array([[0, 4]])
				# 		for cont_num_2 in range(0,Contour_Info.shape[0]):
				# 			if cont_num==cont_num_2:
				# 				continue
				# 			print("MIN__X_R_D2",Contour_Info.shape[0]-1)
				# 			boxA = np.array([Contour_Info[cont_num][0], Contour_Info[cont_num][1],Contour_Info[cont_num][0]+ Contour_Info[cont_num][2],Contour_Info[cont_num][1]+Contour_Info[cont_num][3]])
				# 			boxB = np.array([Contour_Info[cont_num_2][0], Contour_Info[cont_num_2][1],Contour_Info[cont_num_2][0]+Contour_Info[cont_num_2][2],Contour_Info[cont_num_2][1] + Contour_Info[cont_num_2][3]])
				# 			print("BOXAAAA",boxB,boxA)
				# 			iou_r = IOU.compute_iou(boxA,boxB)
				#
				# 			print("MIN__X_R_D23",iou_r,cont_num,cont_num_2,whole_box.shape[0])
				# 			if whole_box.shape[0]==1:
				# 				whole_box = boxA.copy()
				# 				print("MPHKAME PRWTH",whole_box)
				# 			if Contour_Info[cont_num][4]==Contour_Info[cont_num_2][4] and iou_r>0:
				# 				print("MPHKAME PRWTeee", whole_box.shape, whole_box , boxB)
				# 				# whole_box = np.append(whole_box,boxB,axis=0)
				# 				whole_box = np.vstack((whole_box,boxB))
				# 				print("MPHKAME PRWTH2", whole_box.shape,whole_box.shape[0],whole_box,whole_box[:,0],np.where(whole_box[:,0] == np.amin(whole_box[:,0])))
				# 				cont_num+=1
				# 			if (whole_box.size>4) :
				# 				resu = np.amin(whole_box[:, 0])
				# 				result = np.where(whole_box[:,0] == np.amin(whole_box[:,0]))
				# 				print("RER", result, resu, whole_box)



							# else:
							# 	print("WHGEEE",whole_box.shape[0],whole_box.shape,whole_box.size)
							# 	if whole_box.size==4:
							# 		image_trans = cv2.rectangle(image_trans,(boxA[0],boxA[1]),(boxA[0]+boxA[2],boxA[1]+boxA[3]),color,thickness)
							# 		cv2.imshow("DD",image_trans)
							# 		cv2.waitKey(0)
							# 	else:
							# 		po=0
							# 		min_x_coord = np.argmin((whole_box[:,0]))
							# 		# min_x_coord = np.where(whole_box[:,0] == np.amin(whole_box[:,0]))
							#
							# 		print("MIN__XAREDE",whole_box,whole_box.shape[0])
				# print("CONTOUR_INFO",Contour_Info)


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






