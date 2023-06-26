# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os

def add_gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)[:, :, ::-1]
    return out

def add_haze(image, t=0.6, A=1):
    '''
        添加雾霾
        t : 透视率 0~1
        A : 大气光照
    '''
    img_h = image / 255.0
    A = np.random.uniform(0.6, 0.95)
    img_h = img_h * t + A * (1 - t)
    return img_h[:, :, ::-1]

def adjust_image(image, cont=1, bright=0):
    '''
        调整对比度与亮度
        cont : 对比度，调节对比度应该与亮度同时调节
        bright : 亮度
    '''
    out = np.uint8(np.clip((cont * image + bright), 0, 255))[:, :, ::-1]
    # tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
    return out

def adjust_image_hsv(image, h=1, s=1, v=0.8):
    '''
        调整HSV通道，调整V通道以调整亮度
        各通道系数
    '''
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    H2 = np.uint8(H * h)
    S2 = np.uint8(S * s)
    V2 = np.uint8(V * v)
    hsv_image = cv2.merge([H2, S2, V2])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)[:, :, ::-1]
    return out

def adjust_jpeg_quality(image, q=80, save_path=None):
    '''
        调整图像JPG压缩失真程度
        q : 压缩质量 0~100
    '''
    if save_path is None:
        #cv2.imwrite("jpg_tmp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        out = cv2.imread('jpg_tmp.jpg')[:, :, ::-1]
        return out
    else:
        cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), q])

def add_gasuss_blur(image, kernel_size=(7, 7), sigma=0.8):
    '''
        添加高斯模糊
        kernel_size : 模糊核大小
        sigma : 标准差
    '''
    out = cv2.GaussianBlur(image, kernel_size, sigma)[:, :, ::-1]
    return out

def add_peppersalt_noise(image, n=10000):
    result = image.copy()
    # 测量图片的长和宽
    w, h = image.shape[:2]
    # 生成n个椒盐噪声
    for i in range(n):
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result



def test_methods():
    img = cv2.imread('D:\\desk\\tid2013\\reference_images\\I04.BMP')
    out1 = add_haze(img)
    #cv2.imwrite("D:\\desk\\loss\\add_haze.jpg", out)
    out2 = add_gasuss_noise(img)
    #cv2.imwrite("D:\\desk\\loss\\add_gasuss_noise.jpg", out)
    out3 = add_gasuss_blur(img)
    #cv2.imwrite("D:\\desk\\loss\\add_gasuss_blur.jpg", out)
    out4 = adjust_image(img)
    #cv2.imwrite("D:\\desk\\loss\\ajust_image.jpg", out)
    out5 = adjust_image_hsv(img)
    #cv2.imwrite("D:\\desk\\loss\\ajust_image_hsv.jpg", out)
    out6 = img[:, :, ::-1]

    fig = plt.figure(figsize=(10, 8))  # 创建新的figure
    # 绘制2x3两行三列共六个图，编号从1开始
    sub1 = fig.add_subplot(221)
    plt.imshow(out2)
    plt.title('添加高斯噪声', fontsize=20)
    sub2 = fig.add_subplot(222)  # 通过add_subplot()创建一个或多个绘图
    plt.imshow(out1)  # imshow()函数实现绘图
    plt.title('添加雾霾', fontsize=20)
    sub3 = fig.add_subplot(223)
    plt.imshow(out3)
    plt.title('添加高斯模糊', fontsize=20)
    # sub4 = fig.add_subplot(234)
    # plt.imshow(out4)
    # plt.title('调整对比度与亮度')
    # sub5 = fig.add_subplot(235)
    # plt.imshow(out5)
    # plt.title('调整HSV通道')
    sub4 = fig.add_subplot(224)
    plt.imshow(out6)
    plt.title('调整图像JPEG压缩失真程度', fontsize=20)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 手动选择字体，显示中文标签。SimHei 中文黑体 Kaiti 中文楷体 FangSong 中文仿宋
    plt.rcParams['font.size'] = 15  # 设置字体大小
    fig.suptitle("图像降质处理")
    plt.show()  # 图片的显示

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
# image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
#                               "data_set", "Medical image", "Brain Tumor MRI Dataset", "Training", "glioma", "origin", "Tr-gl_0010.jpg")
# img = cv2.imread(image_path)
# out = adjust_image_hsv(img)
# cv2.imshow('img', img)
# cv2.imshow('hsv', out)
# cv2.waitKey()
# cv2.destroyAllWindows()

# test_methods()
# 添加椒盐噪声1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_peppersalt_noise(image=img, n=1)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_peppersalt", "1", filename)
        cv2.imwrite(save_path, out)

# 添加椒盐噪声2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_peppersalt_noise(image=img, n=10)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_peppersalt", "2", filename)
        cv2.imwrite(save_path, out)

# 添加椒盐噪声3
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_peppersalt_noise(image=img, n=100)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_peppersalt", "3", filename)
        cv2.imwrite(save_path, out)

# 添加椒盐噪声4
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_peppersalt_noise(image=img, n=1000)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_peppersalt", "4", filename)
        cv2.imwrite(save_path, out)

# 添加椒盐噪声5
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_peppersalt_noise(image=img, n=10000)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_peppersalt", "5", filename)
        cv2.imwrite(save_path, out)

# #添加高斯噪声1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_noise(image=img, var=0.002)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_noise", "1", filename)
        cv2.imwrite(save_path, out)

# #添加高斯噪声2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_noise(image=img, var=0.004)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_noise", "2", filename)
        cv2.imwrite(save_path, out)

# #添加高斯噪声3
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_noise(image=img, var=0.006)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_noise", "3", filename)
        cv2.imwrite(save_path, out)

# #添加高斯噪声4
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_noise(image=img, var=0.008)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_noise", "4", filename)
        cv2.imwrite(save_path, out)

# #添加高斯噪声5
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_noise(image=img, var=0.01)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_noise", "5", filename)
        cv2.imwrite(save_path, out)
#
# # 添加高斯模糊1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_blur(image=img, sigma=0.1)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_blur", "1", filename)
        cv2.imwrite(save_path, out)

# # 添加高斯模糊2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_blur(image=img, sigma=0.3)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_blur", "2", filename)
        cv2.imwrite(save_path, out)

# # 添加高斯模糊3
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_blur(image=img, sigma=0.5)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_blur", "3", filename)
        cv2.imwrite(save_path, out)

# # 添加高斯模糊4
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_blur(image=img, sigma=0.7)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_blur", "4", filename)
        cv2.imwrite(save_path, out)

# # 添加高斯模糊5
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_gasuss_blur(image=img, sigma=0.9)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_gasuss_blur", "5", filename)
        cv2.imwrite(save_path, out)

# # 添加雾霾1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_haze(image=img, t=0.4)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_haze", "1", filename)
        cv2.imwrite(save_path, out)

# # 添加雾霾2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_haze(image=img, t=0.5)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_haze", "2", filename)
        cv2.imwrite(save_path, out)

# # 添加雾霾3
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_haze(image=img, t=0.6)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_haze", "3", filename)
        cv2.imwrite(save_path, out)

# # 添加雾霾4
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_haze(image=img, t=0.7)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_haze", "4", filename)
        cv2.imwrite(save_path, out)

# # 添加雾霾5
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = add_haze(image=img, t=0.8)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "add_haze", "5", filename)
        cv2.imwrite(save_path, out)

# # 调整压缩程度1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                 "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item,
                                 "add_jpg_quality", "1", filename)
        adjust_jpeg_quality(image=img, q=80, save_path=save_path)

# # 调整压缩程度2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                 "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item,
                                 "add_jpg_quality", "2", filename)
        adjust_jpeg_quality(image=img, q=70, save_path=save_path)

# # 调整压缩程度3
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                 "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item,
                                 "add_jpg_quality", "3", filename)
        adjust_jpeg_quality(image=img, q=60, save_path=save_path)

# # 调整压缩程度4
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                 "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item,
                                 "add_jpg_quality", "4", filename)
        adjust_jpeg_quality(image=img, q=50, save_path=save_path)

# # 调整压缩程度5
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                 "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item,
                                 "add_jpg_quality", "5", filename)
        adjust_jpeg_quality(image=img, q=40, save_path=save_path)

# # 调整过曝光1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                              "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = adjust_image_hsv(image=img, v=1.1)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "over_exposure", "1", filename)
        cv2.imwrite(save_path, out)

# # 调整过曝光2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                          "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = adjust_image_hsv(image=img, v=1.2)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                          "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "over_exposure", "2", filename)
        cv2.imwrite(save_path, out)

# 调整欠曝光1
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                      "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = adjust_image_hsv(image=img, v=0.7)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                      "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "under_exposure", "1", filename)
        cv2.imwrite(save_path, out)

# 调整欠曝光2
folder_list = ["glioma", "meningioma", "notumor", "pituitary"]
for item in folder_list:
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                  "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item, "origin")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, filename))
        out = adjust_image_hsv(image=img, v=0.8)
        save_path = os.path.join(data_root, "home", "MZ2109123", "deep-learning-for-image-processing-master",
                                     "data_set", "Medical image", "Brain Tumor MRI Dataset", "Testing", item,
                                     "under_exposure", "2", filename)
        cv2.imwrite(save_path, out)

