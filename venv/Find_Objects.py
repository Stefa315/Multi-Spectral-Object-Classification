import os
import cv2
import config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models , layers
from collections import Counter
import Get_Tiles_256x256
import math
# model_current = config.loaded_model
# model_current = tf.keras.models.load_model("../saved_models/10")

def sliding_windows(image, x, y, w ,h, step,cnt_count):
    # slide a window across the image
    rois = []
    # o = 0
    for yy in range(y, (y+h)-5 , step):
        o=+1
        p=0
        for xx in range(x, (x+w) -5, step):
            p+=1
            # yield the current window
            if ( yy+step > y+h ) and ( xx+step > x+w ):
                yy = yy - ((yy + step) - (y + h))
                xx = xx - ((xx + step) - (x + w))
                print("EIKONA {} |{} |{}".format(cnt_count,o, p))
                print("2APO X: {} EWS {} MEXRI {}   Y:{} EWS {}  MEXRI {}".format(xx, xx + step , x+w , yy, yy+step , y + h))
                # cv2.imshow("AHOY {} |{} |{}".format(cnt_count,o, p), image[yy:yy + step, xx:xx + step])
                # cv2.waitKey(0)
            elif ( yy+step > y+h ):
                # print("YYY",yy , h , y)
                yy =yy -  ((yy+step) - (y+h))
                print("EIKONA {} |{} |{}".format(cnt_count,o, p))
                print("APO X: {} EWS {} MEXRI {} Y:{} EWS {} MEXRI {} ".format(xx, xx + step , x+w , yy, yy+step , y + h))
                # cv2.imshow("AHOY{} |{} |{}".format(cnt_count,o, p), image[yy:yy + step, xx:xx + step])
                # cv2.waitKey(0)
            elif ( xx+step > x+w ):
                xx = xx - ((xx+step) - ( x + w ))
                print("EIKONA {} |{} |{}".format(cnt_count,o, p))
                print("1APO X: {} EWS {} MEXRI {}  Y:{} EWS {} MEXRI {}".format(xx, xx + step , x+w , yy, yy+step , y + h))

                # cv2.imshow("AHOY{} |{} |{}".format(cnt_count,o, p), image[yy:yy + step, xx:xx + step])
                # cv2.waitKey(0)
            # else
            #     print("EIKONA {} |{} |{}".format(cnt_count,o,p))
            #     print("3APO X: {} EWS {} MEXRI {} Y:{} EWS {} MEXRI {}".format(xx, xx + step , x+w , yy, yy+step , y + h))
                # cv2.imshow("AHOY {} |{} |{}".format(cnt_count,o,p),image[yy:yy + step, xx:xx + step])
                # cv2.waitKey(0)
            crop= image[yy:yy + step, xx:xx + step]
            # print("CROP SHAPE",crop.shape,crop.dtype)
            yield(crop)



def Object_Segmentation(imag):

    Label_List_of_All_Tiles = np.empty((20,0)).tolist()    ## DHMIOURGW ENAN ADEIO PIANAKA DIASTASEWN (10,0) kai ton kanw convert se list
    predictions_main = np.empty((15,5)).tolist()
    most_common_Label = np.empty((5,2))
    # image = cv2.imread('../dataset_3/Hair/Hair_63.jpg')
    # image = cv2.imread(imag)
    img_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 185 , 255, cv2.THRESH_BINARY)

    ##TA COUNTOURS EMPERIKLYOUN TA ANTIKEIMENA POY EINAI KONTA STO LEYKO XRWMA ,
    ##EMEIS THELOYME TO ANTITHETO , ETSI ANTISTREFW TA XRWMATA LEYKO->MAYRO , MAURO-> LEYKO
    for i in range(imag.shape[0]):
        for y in range(imag.shape[1]):
            if (thresh[i][y]==255):
                thresh[i][y]=0
            else:
                thresh[i][y]=255
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    image_copy = imag.copy()

    image_copy_2 = imag.copy()
    cv2.drawContours(image=imag, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                     lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(255,0,0)
    k=0
    for contour in contours:

        if cv2.contourArea(contour)>(0.0017*((imag.shape[0]*imag.shape[1])/2)):

            # print("OBJECTS NEED TO BE BIGGER THAN :", 0.03 * ((image.shape[0] * image.shape[1]) / 2),
            #       cv2.contourArea(contour))
            x,y,w,h = cv2.boundingRect(contour)
            origin=(x+10,y+23)
            cv2.rectangle(image_copy_2,(x,y),(x+w,y+h),(255,255,0),thickness=10)   #DRAWS THE RECTANGLE
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            crop_img = imag[y:y+h , x: x+w] ##MONO TO ORTHOGWNIO APO THNS SYNOLIKH EIKONA
            # print("MHKH PLATOUS",image.shape[0]-x-(image.shape[0]-(x+w)),image.shape[1]-y-(image.shape[1]-(y+h)))
            # print("XYWH",x,y,w,h)
            predictions_main = get_256x256_tiles(image_copy_2,x,y,w,h,k)
            # print("RPEDICTIONS",predictions_main.shape,len(predictions_main),k)

            for indx , predict in enumerate(predictions_main): ##ANALYSING LABELS AND CONFIDENCE OF CLASSIFICATION

                if (np.max(predict)>0.7) :
                    Label_for_Each_Tile = config.img_labels_str[np.argmax(predict)] ##K = ARITHMOS CONTOUR
                    Label_List_of_All_Tiles[k].append(Label_for_Each_Tile)  ##STHN SEIRA K PROSTHESE TO PREDICT APO KATHE TILE
                    counter = Counter(Label_List_of_All_Tiles[k])
                    most_common_Label = np.array(counter.most_common(5)).copy()

            # print("LABEL_LIST",most_common_Label[0][0],most_common_Label,np.array(most_common_Label).shape,len(np.array(most_common_Label))>2)
            if (most_common_Label[0][0]=='No_Object' and (len(np.array(most_common_Label)))>=2): # EAN TO PRWTO STH LISTA MOST COMMON EINAI NO_OBJECT
                image_copy_2 = cv2.putText(image_copy_2, '{}'.format(most_common_Label[1][0]), org=origin,
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA,
                                         bottomLeftOrigin=False)
            else :
                image_copy_2 = cv2.putText(image_copy_2, '{}'.format(most_common_Label[0][0]), org=origin,
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA,
                                         bottomLeftOrigin=False)
            k+=1 #Contour Count
    return image_copy_2


def Object_Segmentation_For_Any_Other(image_trans,image_refl,image_fluo,image_fluo_2,image_polar,image_zoom_counter):
    Contour_Info = np.array([[]])
    Label_List_of_All_Tiles = np.empty((20,0)).tolist()    ## DHMIOURGW ENAN ADEIO PIANAKA DIASTASEWN (10,0) kai ton kanw convert se list
    predictions_main = np.zeros((15,8)).tolist()
    most_common_Label = np.empty((8,2))
    img_gray = cv2.cvtColor(image_trans, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 185 , 255, cv2.THRESH_BINARY)
    ##TA COUNTOURS EMPERIKLYOUN TA ANTIKEIMENA POY EINAI KONTA STO LEYKO XRWMA ,
    ##EMEIS THELOYME TO ANTITHETO , ETSI ANTISTREFW TA XRWMATA LEYKO->MAYRO , MAURO-> LEYKO
    for i in range(image_trans.shape[0]):
        for y in range(image_trans.shape[1]):
            if (thresh[i][y]==255):
                thresh[i][y]=0
            else:
                thresh[i][y]=255
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    print("Contours",contours)
    # draw contours on the original image
    image_copy = image_trans.copy()
    image_3rd_copy = image_trans.copy()
    image_2nd_copy = image_trans.copy()
    # image_copy_2 = image_refl.copy()
    cv2.drawContours(image=image_2nd_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(255,0,0)
    k=0
    # print("OBJECTS NEED TO BE BIGGER THAN :", 0.05*((image.shape[0]*image.shape[1])/2),   cv2.contourArea(contour))
    for contour in contours:

        if cv2.contourArea(contour)>(0.0016*((image_trans.shape[0]*image_trans.shape[1])/2)):
            # Find bounding rectangles
            # print("OBJECTS NEED TO BE BIGGER THAN :", 0.03 * ((image.shape[0] * image.shape[1]) / 2),cv2.contourArea(contour))
            x,y,w,h = cv2.boundingRect(contour)
            origin=(x+10,y+23)
            cv2.rectangle(image_copy,(x,y),(x+w,y+h),(255,255,0),thickness=10)   #DRAWS THE RECTANGLE
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print("MHKH PLATOUS",image.shape[0]-x-(image.shape[0]-(x+w)),image.shape[1]-y-(image.shape[1]-(y+h)))
            # print("XYWH",x,y,w,h)
            predictions_main = Get_Tiles_256x256.Get_256x256_Tiles_Of_All(image_trans,image_refl,image_fluo,image_fluo_2,image_polar,x,y,w,h)
            # print("PREDICTIONS",predictions_main)   #config.img_labels_str[np.argmax(predictions_main)]
            for indx , predict in enumerate(predictions_main): ##ANALYSING LABELS AND CONFIDENCE OF CLASSIFICATION
                print("INDEX_PROBABI",indx , predict)
                # print("TA PANTA OLA",indx , predict)
                if (np.max(predict)>0.6) :
                    Label_for_Each_Tile = config.img_labels_str[np.argmax(predict)] ##K = ARITHMOS CONTOUR
                    Label_List_of_All_Tiles[k].append(Label_for_Each_Tile)  ##STHN SEIRA K PROSTHESE TO PREDICT APO KATHE TILE
                    counter = Counter(Label_List_of_All_Tiles[k])
                    most_common_Label = np.array(counter.most_common(5)).copy()
                    # print("OK",Label_List_of_All_Tiles,most_common_Label)
            print("LABEL_LIST",most_common_Label[0][0],most_common_Label,np.array(most_common_Label).shape)
            if (most_common_Label[0][0]=='No_Object' and (len(np.array(most_common_Label)))>=2): # EAN TO PRWTO STH LISTA MOST COMMON EINAI NO_OBJECT
                text_label = most_common_Label[1][0]
                image_copy = cv2.putText(image_copy, '{}'.format(text_label), org=origin,
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA,
                                        bottomLeftOrigin=False)
            elif (most_common_Label[0][0]) not in config.img_labels_str:
                image_copy = image_3rd_copy
                text_label = 'Empty'
            else :
                text_label = most_common_Label[0][0]
                image_copy = cv2.putText(image_copy, '{}'.format(text_label), org=origin,
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA,
                                         bottomLeftOrigin=False)
            if np.array(contour).size!=0:
                if k==0:
                    Contour_Info = np.array([[x,y,w,h ,text_label,image_zoom_counter]],dtype=object)
                else:
                    temp_info = np.array([[x,y,w,h ,text_label,image_zoom_counter]],dtype=object)
                    Contour_Info = np.append(Contour_Info,temp_info,0)
                k+=1 #Contour Count
            # print("CONTOUR_INFO",Contour_Info,Contour_Info.shape,Contour_Info[0][4])


    # for i in range(Contour_Info.shape[0]-1): ##ENWNW TA CONTOURS ME TA IDIA LABELS

    # cv2.imshow('Final_Decision', image_copy_2)
    # cv2.waitKey(0)
    return image_copy,Contour_Info



def get_256x256_tiles(image,x,y,w,h,cnt_count,model_current):
    rois = []
    ws = (256, 256)
    img_x = image.shape[1]
    img_y = image.shape[0]


    if (w<=ws[0]) and (h>ws[1]):    ##TO WIDTH EINAI MIKROTERO APO 256
        w = ws[0]
        # print("Width < 256  | Height >256")
        if x+w > image.shape[1]:
            w = img_x - x
            x = x - (ws[0] - w )
            w = ws[0]
        for ROI in sliding_windows(image ,x , y ,w ,h ,256,cnt_count):
            rois.append(ROI)
            predictions = model_current.predict(np.asarray(rois),batch_size=2)
    elif (w>ws[0]) and (h<=ws[1]): ##TO HEIGHT EINAI MIKROTERO APO 256
        h = ws[1]
        # print("Width > 256 | Height <256")
        if y+h > image.shape[0]:
            h = img_y - y
            y = y - (ws[1] - h)
            h = ws[1]
        for ROI in sliding_windows(image, x, y, w, h, 256, cnt_count):
            rois.append(ROI)
            predictions = model_current.predict(np.array(rois),batch_size=2)

    elif (w<=ws[0]) and (h<=ws[1]):  ##KAI TA DYO EINAI MIKROTERA APO 256
        w = ws[0]
        h = ws[1]
        # print("Width < 256  | Height < 256")
        if x+w > image.shape[1]:
            w = img_x - x
            x = x - (ws[0] - w )
            w = ws[0]
        if y+h > image.shape[0]:
            h = img_y - y
            y = y - (ws[1] - h)
            h = ws[1]
        for ROI in sliding_windows(image, x, y, w, h, 256, cnt_count):
            rois.append(ROI)
            predictions = model_current.predict(np.array(rois),batch_size=2)

    elif (w>ws[0]) and (h>ws[1]) :
        # print("Width,Height > 256")
        for ROI in sliding_windows(image, x, y, w, h, 256, cnt_count):
            rois.append(ROI)
            # print("ROIS SHAPE_4 ", np.array(rois).shape)
            predictions = model_current.predict(np.array(rois),batch_size=2)
    return predictions

# image = cv2.imread('../panel_213/area_213/frame_25397/cube_180527/-1.jpg')
# Object_Segmentation(image)