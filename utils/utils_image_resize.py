from PIL import Image
import os

def resize_images(source_folder, target_folder, width, height):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        # 检查是否为图片文件
        if os.path.isfile(file_path) and (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 调整图片大小
                    resized_img = img.resize((width, height), Image.ANTIALIAS)
                    
                    # 构建新的文件路径并保存
                    new_file_path = os.path.join(target_folder, filename)
                    resized_img.save(new_file_path)
                    
                    print(f"{filename} 已经被调整大小并保存。")
            except IOError:
                print(f"{filename} 无法打开或处理。")


if __name__ == '__main__':
    source_folder = "../weather_recognition/trainsets/ew/sunny"  # 源文件夹路径
    target_folder = "../weather_recognition/trainsets/data_final/resize_ew/sunny"  # 目标文件夹路径
    resize_images(source_folder, target_folder, 448, 256)