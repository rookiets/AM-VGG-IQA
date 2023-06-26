import math
from itertools import combinations
import PIL.Image as Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat


#Pearson algorithm 皮尔森系数
def pearson(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))

#Spearman algorithm 斯皮尔曼系数
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))

#Kendall algorithm 肯德尔系数
def kendall(x, y):
    assert len(x) == len(y) > 0
    c = 0 #concordant count
    d = 0 #discordant count
    t = 0 #tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)

#读取图片并转换为矩阵形式
def image_to_matrix(file_name):
    # 读取图片
    image = Image.open(file_name)
    #image = image.resize((32, 32))
    # 显示图片
    #image.show()
    width, height = image.size
    # 灰度化
    image_grey = image.convert("L")
    data = image_grey.getdata()
    data = np.matrix(data, dtype="float")
    new_data = np.reshape(data, (height, width))
    return new_data

#循环读取文件
#for file in os.listdir(''):
#    data = image_to_matrix(file)
#将图片转换为列表形式，计算相关系数
# arr1 = image_to_matrix('D://desk//3.jpeg')
# arr2 = image_to_matrix('D://desk//3.jpeg')
# arr1_list = arr1.tolist()
# arr2_list = arr2.tolist()
# arr1_list = list(np.array(arr1_list).flatten())
# arr2_list = list(np.array(arr2_list).flatten())
#kendall_test = kendall(GSM, LGC)
#pearson_test = pearson(arr1_list, arr2_list)
#spearman_test = spearman(arr1_list, arr2_list)


score_list1 = []
score_list2 = []
#读取txt文件
with open('D:\\desk\\tid2013\\mos.txt', encoding='utf-8') as file:
    content = file.readlines()
#逐行读取数据，生成txt文件
for item in content:
    score_list1.append(float(item.rstrip()))
#读取txt文件
with open('D:\\desk\\tid2013\\score.txt', encoding='utf-8') as file:
    content = file.readlines()
#逐行读取数据，生成txt文件
for item in content:
    score_list2.append(float(item.rstrip()))

#read in file
GSM = score_list1
LGC = score_list2
kendall_test = kendall(GSM, LGC)
pearson_test = pearson(GSM, LGC)
spearman_test = spearman(GSM, LGC)

print("肯德尔系数：", kendall_test)
print("皮尔逊系数：", pearson_test)
print("斯皮尔曼系数：", spearman_test)

