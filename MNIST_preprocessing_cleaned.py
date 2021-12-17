# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
# volume_path = 'C:\\Users\\IBK\\Desktop\\new'
volume_path = 'C:\\Users\\IBK\\Desktop\\all'

# volume_path = './data'

ImgPathList =[]

for dirName, subdirList, fileList in os.walk(volume_path):
    for filename in fileList:
        if not filename.startswith('.'):
            print(filename)
            ImgPathList.append(os.path.join(dirName , filename))



# +
from sklearn.model_selection import train_test_split

def data_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=2600, stratify = np.array(y),  random_state=42)
    X_val, X_test, y_val, y_test = train_test_split( X_test, y_test, test_size=100, stratify = np.array(y_test),  random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def mnist_split(X,y):
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size = 10000, stratify = np.array(y),  random_state=42)
    return X_train, y_train, X_val, y_val

def plot_image(image):
    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap="gray")
    plt.axis("off") # 이미지를 출력
#     print('image.shape : ',image.shape) # 차원을 확인 (해상도)
#     print(image.min(), image.max())

###테두리 자르는 코드

def slice_sq(img):
    min = 380
    max = 0
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)

    square_r = []
    square_c = []
    for i in range(380):

        if(img[i,img[i] < 200].shape[0] >100):
            square_r.append(i)        
            square_c = np.where(img[i] < 200)

            if(len(square_c[0]> 100)):
                for j in range(1,50):
                    if(square_c[0][j]+4 == square_c[0][j+1]+3 == square_c[0][j+2]+2 == square_c[0][j+3]+1 == square_c[0][j+4]):
                        if min > square_c[0][j]:
                            min = square_c[0][j]

                    if(square_c[0][-j-4]+4 == square_c[0][-j-3]+3 == square_c[0][-j-2]+2 == square_c[0][-j-1]+1 == square_c[0][-j]):
                        if max < square_c[0][-j]:
                            max = square_c[0][-j]

    up = square_r[0]
    down = square_r[-1]

    img_s = img[up : down , min : max]

    img_rs = cv2.resize(img_s,(340,340))

    return img_rs    

# def slice_sq(img):
#     img = img[10:375,20:360]
#     min = 380
#     max = 0
    
#     square_r = []
#     square_c = []
#     for i in range(365):

#         if(img[i,img[i] != 255].shape[0] >50):
#             square_r.append(i)        
#             square_c = np.where(img[i] != 255)
#             if(len(square_c[0]> 50)):
#                 if min > square_c[0][0]:
#                     min = square_c[0][0]
#                 if max < square_c[0][-1]:
#                     max = square_c[0][-1]

#     up = square_r[0]
#     down = square_r[-1]

#     img_s = img[up : down , min : max]
    
#     img_rs = cv2.resize(img_s,(340,340))
#     return img_rs


# +
##이미지 불러와서 테두리 자른 후 10X10으로 자름
arr = np.zeros((1,28,28))
y = []
for ImgPath in ImgPathList:
    img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 17)
#     plot_image(255-img)
#     plot_image(img)
    
    img = slice_sq(img)
    pix = 28
#     plot_image(img)
#     plot_image(255-img)
    CF_margin = 4
    RF_margin = 3
    C_margin = 6
    R_margin = 6
    
    RF_index = RF_margin
    RL_index = RF_margin+pix


    for i in range(10):
        CF_index = CF_margin
        CL_index = CF_margin + pix

        for j in range(10):
            label = j
            arr = np.append(arr, np.array([img[RF_index : RL_index, CF_index : CL_index]]), axis = 0)
            CF_index = CL_index + C_margin
            CL_index += pix + C_margin
            y.append(label)
        
        RF_index = RL_index + R_margin
        RL_index += pix + R_margin



arr = arr[1:arr.shape[0]]    
print(arr.shape)

arr = 255 - arr
y_ma = y

# np.save('./arr_all_yj_5c', boxed)
# np.save('./y_ma_all_yj_5c', y_boxed)

# +
# ##이미지 불러와서 테두리 자른 후 10X10으로 자름
# arr = np.zeros((1,28,28))
# y = []
# for ImgPath in ImgPathList[:1]:
#     img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 17)

#     plot_image(img)
#     plot_image(255-img)
#     img = slice_sq(img)
#     pix = 28
# #     plot_image(img)
#     plot_image(255-img)
#     CF_margin = 4
#     RF_margin = 3
#     C_margin = 6
#     R_margin = 6
    
#     RF_index = RF_margin
#     RL_index = RF_margin+pix


#     for i in range(10):
#         CF_index = CF_margin
#         CL_index = CF_margin + pix

#         for j in range(10):
#             label = j
#             arr = np.append(arr, np.array([img[RF_index : RL_index, CF_index : CL_index]]), axis = 0)
#             CF_index = CL_index + C_margin
#             CL_index += pix + C_margin
#             y.append(label)
        
#         RF_index = RL_index + R_margin
#         RL_index += pix + R_margin



# arr = arr[1:arr.shape[0]]    
# print(arr.shape)

# arr = 255 - arr
# y_ma = y

# # np.save('./arr_all_yj_5c', boxed)
# # np.save('./y_ma_all_yj_5c', y_boxed)
# -

for i in range(100):
    plot_image(arr[i])

# +
#자른 100개 마스크

boxed = np.zeros((1,28,28))

y_boxed = []

for y_in in range(len(y_ma)):

    binary = arr[y_in].astype(np.uint8)

    contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    digit_arr = [] 
    digit_arr2 = [] 
    count = 0

    box = []    
        #검출한 외곽선에 사각형을 그려서 배열에 추가 
    larWH = 50
    for i in range(len(contours)) : 
        bin_tmp = binary.copy() 
        x,y,w,h = cv2.boundingRect(contours[i]) 
        if w*h > larWH:
            bR_arr = [[x,y,w,h]]
    
# 
#     bR_arr = sorted(bR_arr, key=lambda num : num[0], reverse = False)

    for x,y,w,h in bR_arr :
        tmp_y = bin_tmp[y-2:y+h+2,x-2:x+w+2].shape[0]
        tmp_x = bin_tmp[y-2:y+h+2,x-2:x+w+2].shape[1]

        if tmp_x and tmp_y > 10 :
#             cv2.rectangle(color,(x-2,y-2),(x+w+2,y+h+2),(0,0,225),1)
#             print(np.array(bin_tmp[y-2:y+h+2,x-2:x+w+2]).shape)
            digit_arr = np.array(bin_tmp[y-2:y+h+2,x-2:x+w+2])


#             digit_arr = np.array(digit_arr).reshape(height,width)
            digit_arr = cv2.resize(digit_arr,(28,28))
            boxed = np.append(boxed, np.array([digit_arr]), axis = 0)
#             boxed_ag = np.append(boxed_ag,np.array([arr[y_in]]), axis = 0)
        
        else:
            boxed = np.append(boxed, np.array([arr[y_in]]), axis = 0)
        
        y_boxed.append(y_ma[y_in])
        

boxed = boxed[1:]

print(boxed.shape)
print(len(y_boxed))

# np.save('./X_boxed_all_adTh1_la_>10', boxed)
# np.save('./y_boxed_all_adTh1_la_>10', y_boxed)
# -

for i in range(100):
    plot_image(boxed[i])
