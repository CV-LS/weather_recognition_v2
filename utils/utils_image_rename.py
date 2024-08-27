#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os

class BatchRename():
    def __init__(self):
        self.path = '../weather_recognition/trainsets/ew/indoor'  #表示需要命名处理的文件夹目录，复制地址后注意反斜杠

    def rename(self):  
        filelist = os.listdir(self.path)   #获取文件路径
        #给文件中的图片按从小到大进行排序
        filelist.sort(key=lambda x: str(x.split('.')[0]))
        print(filelist)
        total_num = len(filelist)  #获取文件长度（文件夹下图片个数）
        i = 3000  #表示文件的命名是从1开始的
        for item in filelist:  
            if item.endswith('.jpg') or item.endswith('.JPG') or item.endswith('.png') or item.endswith('.jpeg'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可，我习惯转成.jpg）
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), "indoor_" + format(str(i)) + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                # 这种情况下的命名格式为000xxxx.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
    

