# coding=utf-8
import os


def gen_txt(txt_path, img_dir, image_path):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('jpg'):  # 若不是png文件，跳过
                    continue
                # label = (img_list[i].split('.')[0] == 'cat')? 0 : 1
                label = img_list[i].split('.')[0]
                # 将字符类别转为整型类型表示
                if label == 'cat':
                    label = '0'
                else:
                    label = '1'
                img_path = os.path.join(image_path, sub_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
    f.close()

def get_txt(txt_path, mos_path, img_dir):
    f = open(txt_path, 'a')
    list1 = []
    for file in os.listdir(img_dir):
        list1.append(file)
    fh = open(mos_path, 'r')
    mos = []
    for line in fh:
        mos.append(line)
    for index1, i in enumerate(list1):
        for index2, j in enumerate(mos):
            if index1 == index2:
                pic_path = os.path.join(img_dir, i)
                address = pic_path + ',' + mos[index2]
                f.write(address)
    f.close()

if __name__ == '__main__':
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    train_txt_path = os.path.join(data_root,
                              "data_set", "tid2013", "train_label.txt")
    train_dir = os.path.join(data_root,
                              "data_set", "tid2013", "distorted_images")
    # valid_txt_path = os.path.join("data", "catVSdog", "test.txt")
    # valid_dir = os.path.join("data", "catVSdog", "test_data")
    # gen_txt(train_txt_path, train_dir, os.path.join(data_root, "CNN_PyTorch_Beginner-main", "VGGNet", "data", "catVSdog", "train_data"))
    # gen_txt(valid_txt_path, valid_dir, os.path.join(data_root, "CNN_PyTorch_Beginner-main", "VGGNet", "data", "catVSdog", "test_data"))
    mos_path = os.path.join(data_root,
                              "data_set", "tid2013", "mos.txt")
    get_txt(train_txt_path, mos_path, train_dir)