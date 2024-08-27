# -*- coding: UTF-8 -*- 
#!/usr/bin/env python
 
from PIL import Image
 
im_num = []
for line in open("../weather_recognition/CoCo.txt", "r"):
    im_num.append(line)
#print(im_num)
 
for a in im_num:
  im_name = '../weather_recognition/train2017/{}'.format(a[:-1])
  print(im_name)
  im = Image.open(im_name)#打开指定路径下的图像
 
  tar_name = '../weather_recognition/data/coco/{}'.format(a[:-1])
  print(tar_name)
  im.save(tar_name) #另存
  im.close()