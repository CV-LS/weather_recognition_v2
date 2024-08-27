import cv2
import numpy as np

# 计算雾化图像的暗通道

def DarkChannel(img, size=15):
    """
    暗通道的计算主要分成两个步骤:
    1.获取BGR三个通道的最小值
    2.以一个窗口做MinFilter
    ps.这里窗口大小一般为15（radius为7）
    获取BGR三个通道的最小值就是遍历整个图像，取最小值即可
    """
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    """
    :param saturation_lvl: 2**14-1 is a common value. Not all images
                           have the same value.
    """
    return (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0))


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    # b = np.array([-0.76391272,  0.85105759, -0.23677116])
    # r = b * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return aug_img

def gamma(image):
    image = linearize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32))
    image = np.clip(image, 0, 1)
    rgb = np.clip(image, 0, 1)**(1.2)
    image = (rgb*255).astype(np.uint8) 
    return image

def gamma_transform_s(img, gamma):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 1] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 1] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def gamma_transform_v(img, gamma):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def gamma_transform_sv(img, gamma1,gamma2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma1)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)

    illum = hsv[..., 1] / 255.
    illum = np.power(illum, gamma2)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 1] = v.astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == '__main__':
    test_imgfog_path = "../weather_recognition/test/fog/fog_4.jpg"
    test_imgsmow_path = "../weather_recognition/test/snow/snow_1.jpg"

    test_imgfog = cv2.imread(test_imgfog_path)
    test_imgsnow = cv2.imread(test_imgsmow_path)

    # reslut = augment_hsv(test_img)
    # reslut = gamma(test_img)
    reslutfog = gamma_transform_sv(test_imgfog,1.2,1.1)
    reslutsonw = gamma_transform_sv(test_imgsnow,1.2,1.1)
    
    cv2.imwrite("../weather_recognition/test/reslut/test_fog4.jpg",reslutfog)
    cv2.imwrite("../weather_recognition/test/reslut/test_snow.jpg",reslutsonw)
    

