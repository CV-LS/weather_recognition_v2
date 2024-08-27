import os
import random
import shutil

def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

# 把下面改成你的参数设置
org_img_folder = '../weather_recognition/mul_data/backlight'  # 待处理源文件夹路径
tar_img_folder = '../weather_recognition/mul_dataloader/train'     # 移动到新文件夹路径
picknumber = 700  # 需要从源文件夹中抽取的图片数量
img_format = 'jpg' # 需要处理的图片后缀
i = 1  # 选取后的图片从1开始命名

# 检索源文件夹并随机选择图片
imglist = getFileList(org_img_folder, [], img_format)  # 获取源文件夹及其子文件夹中图片列表
samplelist = random.sample(imglist, picknumber)  # 获取随机抽样后的图片列表

print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
print('本次共随机抽取 ' + str(len(samplelist)) + ' 张图像\n')

# 复制选取好的图片到新文件夹中，并重新命名
new_img_folder = tar_img_folder
for imgpath in samplelist:
    # name = str(i).zfill(5)  # 设置图片名为5位数，即从00001开始重新命名
    # new_img_folder = os.path.join(tar_img_folder, name + '.' + img_format)
    # i = i + 1 
    # 如果不需要重命名就把上面三行注释掉
    shutil.copy(imgpath, new_img_folder)  #  复制图片到新文件夹

