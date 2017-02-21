# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:17:32 2017

@author: lin
"""

from PIL import Image 
import cv2
import numpy as np
import PCA        
import matplotlib.pyplot as plt
import random

#读取原始图片并转化为numpy.ndarray，将灰度值由0～256转换到0～1  
# read the image data from olivettifaces.gif using the package PIL
# U need to dowmload the package PIL firstly
img = Image.open('olivettifaces.gif')

# the type of img is numpy.ndarray
# this image olivettifaces has the size 1190*942, and it has totally 20*20 images of 20 people
# so every image has the size （1190/20）*（942/20）= 57*47=2679 
# we store all the 400 images by a array, which has size 400 * 2679
# and the first dimension is about image, and the second is the dimension of image of face


img_ndarray = numpy.asarray(img,dtype=float)/256                        
olivettifaces=numpy.empty((400,2679))
#here, we  only store the image of gray and we change the value from [0,255] in int to value [0,1] in float



for row in range(20):
	for column in range(20):
		olivettifaces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

olivettifaces_label=numpy.empty(400)
for label in range(40):
	olivettifaces_label[label*10:label*10+10]=label
olivettifaces_label=olivettifaces_label.astype(numpy.int)

# all above is about the process to read image and store it in the array, which u dont need to understand




# the below is more about the preprocess and the process of PCA 
faces= olivettifaces
label = olivettifaces_label

train_data=numpy.empty((320,2679))
train_label=numpy.empty(320)

valid_data=numpy.empty((40,2679))
valid_label=numpy.empty(40,dtype = int)

test_data=numpy.empty((40,2679))
test_label=numpy.empty(40, dtype = int)

#for each person, we have 10 image, and we take 8 of them as train_data, which need to be learned
#and 1 of them as a model, which means a reference 
# and the last one, we look it as a test image, that we need to judge whether it's similary to the model or not

for i in range(40):
	train_data[i*8:i*8+8]=faces[i*10:i*10+8]
	train_label[i*8:i*8+8]=label[i*10:i*10+8]
    
	valid_data[i]=faces[i*10+8]
	valid_label[i]=label[i*10+8]
    
	test_data[i]=faces[i*10+9]
	test_label[i]=label[i*10+9]
    
      
## from now on, we start to used PCA   
## I just consider the first person here, and I take the 8 image of the first person as exemple  
img_exemple = train_data[0:8]  
  

##We need to have a good understanding of the algo PCA
#In PCA, we can get a lot of information, for exemple:
# the eigenvecteur, which is named "n_eigVect" in the code PCA
# the eigenvalue                    "eigVals"
# the coordinates of the 8 images exemples within the eigenvecteur, "lowDDataMat"
# and also the projection of the 8 images exemples in the eigenvecteur space, "img_result_reconmat"
# we just get what we want from the code PCA, which means u need to understand it


#[img_result_lowdatamat,img_result_reconmat,mean] = PCA.pca(img_exemple)
[mean,eigenvecteurs] =PCA.pca(img_exemple)
# mean is the mean of the 8 images exemples; in mathmatical view, the original point


valid_image = np.asmatrix(valid_data[0]) 
test_image_1 = np.asmatrix(test_data[0])
#get a random person except the first person
person2 =random.randint(1,20)
test_image_2 =np.asmatrix(test_data[person2])
# here valid_image is the model for the first person
#test_image_1 is the test image of the first person and the test_image_2 is the other person


valid_image_mean = valid_image-mean
test_image_1_mean =test_image_1-mean 
test_image_2_mean = test_image_2-mean 

value_valid = valid_image_mean * eigenvecteurs.T
value_imgae_1 =test_image_1_mean*eigenvecteurs.T
value_imgae_2 =test_image_2_mean*eigenvecteurs.T

##Firstly, we subtract the mean value from those images
## and second, we represent them in those eigenvecteurs
# so for exemple, the value_valid = [-1.94036955+0.j,  0.88698075+0.j, -3.29796039+0.j, -1.61098387+0.j,-1.51514546+0.j,  0.02646873+0.j,  0.03309258+0.j]
##                which means the coordinate of the valid image within  those eigenvecteurs



distance_1 = (value_valid - value_imgae_1)
value_1 = (distance_1*distance_1.T)[0,0].real
      
distance_2 = (value_valid - value_imgae_2)
value_2 = (distance_2*distance_2.T)[0,0].real
  
# value_1 is the distance between the valid image and the test_image_1        
#value_2 is the distance between the valid image and the test_image_2
# if value_1 < value_2, it means   test_image_1 is much closed to the valid image 
#   than test_iamge_2,  within the coordinate system created by those eigenvecteurs

[value_1,value_2]        
##[17.95142701884696, 39.07412441133593]
print("value of person 0:\n",value_1)
print("value of person ",person2,":\n",value_2)
# the below is not important












#plt.imshow(img_result_reconmat,cmap = 'gray')
#plt.imshow(img_result_lowdatamat,cmap = 'gray')
#img_t= img_result_reconmat[0,:]
#img_tt = np.asarray(img_t)
#img_ttt = numpy.ndarray.flatten(img_tt)
#img_tttt = [round(min(i*256,255)) for i in img_ttt]
#img_tttt = img_tttt.reshape(57,47)
#plt.imshow(img_tttt,cmap = 'gray')
#









