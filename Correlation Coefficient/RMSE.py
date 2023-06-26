import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def evaluation(y_test, y_predict):
    #mae = mean_absolute_error(y_test, y_predict)
    #mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    #mape = (abs(y_predict - y_test) / y_test).mean()
    #r_2 = r2_score(y_test, y_predict)
    return rmse

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
sz1 = np.array(score_list1)
sz2 = np.array(score_list2)
result = evaluation(sz1, sz2)
print(result)
