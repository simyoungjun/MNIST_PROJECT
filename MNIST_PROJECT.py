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
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import cv2
volume_path = 'C:\\Users\\IBK\\Desktop\\data'
ImgPathList =[]

for dirName, subdirList, fileList in os.walk(volume_path):
    for filename in fileList:
        print(filename)
        ImgPathList.append(os.path.join(dirName , filename))


# +

# y=[]
# n=0
# ImgPathList =[]
# for dirName, subdirList, fileList in os.walk(volume_path):
#     if fileList != []:
#         for filename in fileList:
#             ImgPathList.append(os.path.join(dirName , filename))
# #             print(dirName)
# #             print(int(filename))
# #             print(n)
#             y.append(int(dirName[-1]))

        

# +

# import os

# volume_path = 'C:\\Users\\IBK\\Desktop\\data'

# X = np.zeros((28,28))
# y=[0]

# for dirName, subdirList, fileList in os.walk(volume_path):
#     if fileList != []:
#         for filename in fileList:
#             ImgPath = os.path.join(dirName, filename)
#             img  = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
#             X = np.stack([X, img]) 
#             y.append(int(dirName[-1]))

# X = X[1:X.shape[0]]
# Y = Y[1:X.shape[0]]
            
        
# -

print(ImgPathList)


def plot_image(image):
    plt.figure(figsize=(4,4))
    plt.imshow(image, cmap="gray")
    plt.axis("off") # 이미지를 출력
#     print('image.shape : ',image.shape) # 차원을 확인 (해상도)
#     print(image.min(), image.max())


ImgPathList[0]

# +
arr = np.zeros((1,28,28))
y = []
for ImgPath in ImgPathList:
# for ImgPath in ImgPathList[0:2]:
    img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 17)
    
#     print(img.shape)
#     img = 255- img
    plot_image(img)
    img[379,379]


    pix = 28

    CF_margin = 28
    # CL_margin = 10
    RF_margin = 33
    # RL_margin = 10
    C_margin = 6
    R_margin = 6
    
    RF_index = RF_margin
    RL_index = RF_margin+pix
    # CF_index = 0
    # CL_index = 0+pix


    for i in range(10):
        CF_index = CF_margin
        CL_index = CF_margin + pix

        for j in range(10):
    #         print('CF ', CF_index, ', CL ', CL_index, ', RF ',RF_index, 'RL ', RL_index)
            label = j
            arr = np.append(arr, np.array([img[RF_index : RL_index, CF_index : CL_index]]), axis = 0)
    #         print(img[CF_index : CL_index, RF_index : RL_index].shape)
    #         print('arr.shape ', arr.shape)
            CF_index = CL_index + C_margin
            CL_index += pix + C_margin
            y.append(label)
    #         print('CF ', CF_index, ', CL ', CL_index, ', RF ',RF_index, 'RL ', RL_index)

        RF_index = RL_index + R_margin
        RL_index += pix + R_margin



arr = arr[1:arr.shape[0]]    
print(arr.shape)

for i in range(20):
    plot_image(arr[i*20])
print(arr.shape)
y_ma = y
# -



for i in range(2700):
    arr[i] = 255-arr[i]
    plot_image(arr[i])
print(arr.shape)

n =0
for i in range(10):
    plot_image(arr[i*10 + 1])


arr.shape
y = y_m
len(y)

X_m = arr
y_m = y_n
len(y_m)
print(y_m)

for i in range(arr.shape[0]//100):
    plot_image(arr[i])

data.shape
print(len(y))
print(data.shape)
print(y)

# +
from sklearn.model_selection import train_test_split

def data_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=700, stratify = np.array(y),  random_state=42)
    X_val, X_test, y_val, y_test = train_test_split( X_test, y_test, test_size=400, stratify = np.array(y_test),  random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def mnist_split(X,y):
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=10000, stratify = np.array(y),  random_state=42)
    return X_train, y_train, X_val, y_val


# +
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]
X= np.array(X)
y= np.array(y)
# -

X.shape
X = X.reshape(-1,28,28)
print(X.shape)

# +

X_ = X
y_ = y

X_trainm, y_trainm, X_valm, y_valm = mnist_split(X,y)
X_train, y_train, X_val, y_val, X_test, y_test = data_split(arr,y_m)

# print(X_trainm.shape, X_train.shape)
# X_trainm = np.expand_dims(X_trainm, axis=0)
# X_train = np.expand_dims(X_train, axis=0)
X_train = np.concatenate((X_trainm, X_train),axis = 0)
X_val = np.concatenate((X_valm, X_val),axis = 0)

# y_trainm = np.expand_dims(X_trainm, axis=0)
# y_train = np.expand_dims(X_train, axis=0)
y_train = np.concatenate((y_trainm, y_train),axis = 0)
y_val = np.concatenate((y_valm, y_val),axis = 0)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(np.array(X_test).shape, np.array(y_test).shape)


# -

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))

# +
from sklearn.model_selection import train_test_split

def data_split(X,y,):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify = np.array(y),  random_state=42)
    X_val, X_test, y_val, y_test = train_test_split( X_test, y_test, test_size=0.3, stratify = np.array(y_test),  random_state=42)
    


# +
n =0
for i in range(10):
    plot_image(arr[i*10])

# plot_image(data[100])

# +


plot_image(data[0])

# +
### 이미지 파일 배열로 저장
y = np.array(y)

N = len(y)

X = cv2.resize(cv2.imread(ImgPathList[0], cv2.IMREAD_GRAYSCALE),(100,100)) # 흑백 이미지로 로드
# X = cv2.resize(X, (100, 100))
X = np.reshape(X, (1,-1))
print(X.shape)

for i in range(1,N):
#     print(i)
    img  = cv2.imread(ImgPathList[i], cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
    img = cv2.resize(img, (100, 100))
    img = np.reshape(img, (1,-1))
    
    X = np.append(X, np.array(img), axis = 0)
    
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify = np.array(y),  random_state=42)
print(y_train.sort())

# +
np.sort(y_train)
print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))


# -

img.shape

# +
###전처리 코드

import cv2
cv2.__version__
image = cv2.imread(ImgPathList[1], cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
# image = cv2.resize(image, (50, 50))

image_enhanced = cv2.equalizeHist(image) # 이미지 대비를 향상시킵니다.


def img_binarize(image):
    max_output_value = 255
    neighborhood_size = 99
    subtract_from_mean = 10
    image_binarized = cv2.adaptiveThreshold(image_enhanced, max_output_value,
                                             cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             neighborhood_size, subtract_from_mean) # 적응적 임계처리를 적용
    
    return image_binarized
image_binarized = img_binarize(image_enhanced)


def plot_image(image):
    plt.figure(figsize=(4,4))
    plt.imshow(image, cmap="gray")
    plt.axis("off") # 이미지를 출력
    print('image.shape : ',image.shape) # 차원을 확인 (해상도)
    print(image.min(), image.max())

blur = cv2.GaussianBlur(image, ksize=(7,7), sigmaX=0)
    
plot_image(image)
plot_image(blur)
plot_image(image_enhanced)
plot_image(image_binarized)


# plt.figure(figsize=(1,1))
# plt.imshow(image_enhanced, cmap="gray"), plt.axis("off") # 이미지를 출력

# plt.imshow(image_reshaped, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
type(image) # 데이터 타입을 확인
image # 이미지 데이터를 확인
print('image.shape : ',image.shape) # 차원을 확인 (해상도)
print(image.min(), image.max())
# -

img = Image.open(ImgPathList[7])
img = img.resize((50,50))
plot_image(img)


help(cv2.fastNlMeansDenoisingColored)

# +
##전처리 코드

image = cv2.imread(ImgPathList[8], cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
image = cv2.resize(image, (100, 100))
thr3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 17)
blur = cv2.GaussianBlur(thr3, ksize=(35,35), sigmaX=9)


plot_image(thr3)

# -

for i in range(28):
    image = cv2.imread(ImgPathList[i], cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
    image = cv2.resize(image, (50, 50))
    thr3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 17)

    plot_image(thr3)


# +
img = cv2. imread(ImgPathList[1], cv2.IMREAD_GRAYSCALE)
# img = img.astype(np.uint8)
print(image.shape)
thr = cv2.resize(thr,(100,100))
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# thr = cv2.adaptiveThreshold(image,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

denoised_img = cv2.fastNlMeansDenoising(thr,None, 1, 10, 7, 21)
plot_image(thr)



# +

mask = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)[1]



mask = 255 - mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
result = image.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
# result[:, :, 3] = mask
plot_image(mask)
plot_image(result)

result1 = img_binarize(result)
plot_image(result1)

# -

src = image
height, width = src.shape
dst = cv2.pyrUp(src, dstsize=(width * 2, height * 2), borderType=cv2.BORDER_DEFAULT)
plot_image(dst)

# +
blur = cv2.GaussianBlur(image_binarized, ksize=(5,5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(blur, 10, 250)
cv2.imshow('Edged', edged)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

contours_xy = np.array(contours)
contours_xy.shape
plot_image(blur)
plot_image(thresh1)
edged = cv2.resize(edged,(50,50))
plot_image(edged)




# +
def image_boxing(image):
    img_idx = np.where(image_binarized == 255 )
    idx_raw_min = min(img_idx[0])
    idx_raw_max = max(img_idx[0])
    idx_col_min = min(img_idx[1])
    idx_col_max = max(img_idx[1])
    
    print(idx_raw_min,idx_raw_max,idx_col_min,idx_col_max)
    padding = 10

#     image_cropped = image[idx_raw_min-padding:idx_raw_max+padding,idx_col_min-padding:idx_col_max+padding]
    image_cropped = image[idx_raw_min:idx_raw_max,idx_col_min:idx_col_max]
    
    return image_cropped

cropped = image_boxing(image_binarized)
print(image_binarized[10,10])

# +


img_res = cv2.resize(image_binarized, (28, 28), 5,5, cv2.INTER_CUBIC)

plot_image(cropped)
plot_image(image_binarized)
plot_image(img_res)

# +
src = cv2.imread(ImgPathList[0]) # 흑백 이미지로 로드

src = cv2.selectROI(src) # 초기 위치 지정하고 모서리 좌표 4개를 튜플값으로 반환
mask = np.zeros(src.shape[:2], np.unit8) # 마스크는 검정색으로 채워져있고 입력 영상과 동일한 크기

# 결과를 계속 업데이트 하고 싶으면 bgd, fgd 입력
cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)

# grabCut 자료에서 0,2는 배경, 1,3은 전경입니다.
# mask == 0 or mask == 2를 만족하면 0으로 설정 아니면 1로 설정합니다
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('unit8')

# np.newaxis로 차원 확장
dst = src * mask2[:, :, np.newaxis]

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()


# -

def boxing(image):
    
