import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

# 灰度化
def backlight(file_name):
    image = cv2.imread(file_name,0)
    
    if len(image.shape) == 3:
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayImg = image.copy()
        
    # 效果展示
    plt.hist(grayImg.ravel(), 256)
    plt.show()
    dark_part = cv2.inRange(grayImg, 0, 60)
    bright_part = cv2.inRange(grayImg, 225, 256)
    dark_pixel = np.sum(dark_part > 0)  # 是一个true和false的数组
    bright_pixel = np.sum(bright_part > 0)
    # 总的像素值
    total_pixel = np.size(grayImg)
    # 占比
    percentage = dark_pixel / total_pixel + bright_pixel / total_pixel
    
    return percentage

def get_fm(file_name):
	image = cv2.imread(file_name,0)
	hist = cv2.calcHist([image],[0],None,[256],[0,256])
 
	nums=image.shape[0]*image.shape[1]
	hist=hist/nums
 
	# 总体均值
	u=0
	for i in range(256):
		u+=hist[i]*i
	# print(u)
 
	# 总体方差
	a=0
	for i in range(256):
		a+=hist[i]*pow((i-u),2)
	# print(a)
 
	# 逆光度
	dbl=np.sqrt(a)
	
	return dbl
 
 
 
# file_name='data2/img/blur (8).jpg'
# dbl=get_fm(file_name)
# print(dbl)
if __name__ == '__main__':
	path='../weather_recognition/dataset/dataset/'
	value_list = []
	name_list = []
	for file in os.listdir(path):
		file_name = path + file
		# dbl = get_fm(file_name)
		# print(file, dbl)
		blk = backlight(file_name)
		print(file, blk)

		if blk > 0.5:
			value_list.append(blk)
			name_list.append(file)
			dataframe = pd.DataFrame({'name':name_list,'value': value_list})
			dataframe.to_csv("../weather_recognition/dataset.csv", index=False)