# 解析xml工具包
import xml.etree.ElementTree as ET
import os
import pickle
from os import listdir, getcwd
from os.path import join

sets = []
name_class = ['WhitePlate', 'license_plate', 'ColorPlate']

def convert_coordinate(size, box):  # size：原圖的 w 與 h, box: xmin, ymin, xmax, ymax
    # print('size:', size, '\nbox:', box)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    # print(dw, dh)
    x = (box[0] + box[2]) / 2.0  # box的x中心
    y = (box[1] + box[3]) / 2.0  # box的y中心
    w = box[2] - box[0]  # box的w寬度
    h = box[3] - box[1]  # box的h高度
    # print('xywh', x, y, w, h)
    # 轉換為比例座標
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    # print(x, y, w, h)
    return (x, y, w, h)

def convert_annotation(input_path,output_path):
    # 傳入xml資料夾路徑
    # 轉換成[]格式把每一個檔案路徑建立在name_path
    name_path = []
    name = os.listdir(input_path)
    name.sort()
    for i in name:
        if i == ".DS_Store":
            print(i)
            name.remove(i)
        else:
            name_path.append(os.path.join(input_path, i))
    for i in range(len(name_path)):
        name_jpg = name[i].split('.')[0]
        tree = ET.parse(name_path[i])
        root = tree.getroot()
        # 讀取 原圖的 size：w, h. box: xmin, ymin, xmax, ymax
        size = root.find("size")
        W = int(size.find("width").text)
        H = int(size.find("height").text)
        # print(W, H)
        for obj in root.iter("object"):
            clasname = str(obj.find("name").text)
            bndbox = obj.find('bndbox')
            box = (float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                   float(bndbox.find('xmax').text), float(bndbox.find('ymax').text))
            yolosize = convert_coordinate((W, H), box)
            print(yolosize)
            class_index = name_class.index(clasname)
            yolosize = (class_index, format(yolosize[0], '.12f'), format(yolosize[1], '.12f'),
                        format(yolosize[2], '.12f'), format(yolosize[3], '.12f'))
            print(yolosize)
            with open(os.path.join(output_path, '%s.txt' %(name_jpg)), 'a') as f:
                for i in yolosize:
                    f.write(str(i)+" ")
                f.write('\n')
    with open(os.path.join(output_path, 'classes.txt'), 'a') as f:
        for i in name_class:
            f.write(i + '\n')
    # print(name_jpg)


if __name__ == '__main__':
    xml_path = "/Users/anthony/Desktop/xmltest"
    txt_path = "/Users/anthony/Desktop/txttest"
    convert_annotation(xml_path, txt_path)
    print("Convert xml to txt file complete")
