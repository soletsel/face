# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:17:32 2017

@author: lin
"""
#读取人脸库olivettifaces，并存储为pkl文件  

from PIL import Image 
import cv2
#import cPickle
import numpy
import PCA
#读取原始图片并转化为numpy.ndarray，将灰度值由0～256转换到0～1  
#opencv 不能读取gif
img = Image.open('olivettifaces.gif')
#图片大小时1190*942，一共20*20个人脸图，故每张人脸图大小为（1190/20）*（942/20）即57*47=2679  
#将全部400个样本存储为一个400*2679的数组，每一行即代表一个人脸图，并且第0～9、10～19、20～29...行分别属于同个人脸  
#另外，用olivettifaces_label表示每一个样本的类别，它是400维的向量，有0～39共40类，代表40个不同的人。  
#
img_ndarray = numpy.asarray(img,dtype=float)/256
                         
olivettifaces=numpy.empty((400,2679))

for row in range(20):
	for column in range(20):
		olivettifaces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])
##建一个&lt;span style=&quot;font-family: SimSun;&quot;&gt;olivettifaces_label&lt;/span&gt;

olivettifaces_label=numpy.empty(400)
for label in range(40):
	olivettifaces_label[label*10:label*10+10]=label
olivettifaces_label=olivettifaces_label.astype(numpy.int)
# 记录 第[i] 张照片 是第j个人的

#保存olivettifaces以及olivettifaces_label到olivettifaces.pkl文件
#write_file=open(&#39;/home/wepon/olivettifaces.pkl&#39;,&#39;wb&#39;)  
#cPickle.dump(olivettifaces,write_file,-1)  
#cPickle.dump(olivettifaces_label,write_file,-1)  
#write_file.close() </pre><br>

#图片种类数量有400张，40个人，每人10张
#img1 = olivettifaces[1]              
#img1 = img1.reshape(57,47)
#plt.imshow(img1,cmap = 'gray')

#读取olivettifaces.pkl文件，分为训练集（40*8个样本），验证集（40*1个样本），测试集（40*1个样本）

faces= olivettifaces
label = olivettifaces_label

train_data=numpy.empty((320,2679))
train_label=numpy.empty(320)

valid_data=numpy.empty((40,2679))
valid_label=numpy.empty(40,dtype = int)

test_data=numpy.empty((40,2679))
test_label=numpy.empty(40, dtype = int)

for i in range(40):
	train_data[i*8:i*8+8]=faces[i*10:i*10+8]
	train_label[i*8:i*8+8]=label[i*10:i*10+8]
    
	valid_data[i]=faces[i*10+8]
	valid_label[i]=label[i*10+8]
    
	test_data[i]=faces[i*10+9]
	test_label[i]=label[i*10+9]
    
img_exemple = test_data[1]              
img_exemple = img_exemple.reshape(57,47)
plt.imshow(img_exemple,cmap = 'gray')

[img_result_lowdatamat,img_result_reconmat] = PCA.pca(img_exemple)


plt.imshow(img_result_reconmat,cmap = 'gray')




















