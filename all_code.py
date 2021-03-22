# -*- coding: utf-8 -*-
import sys
import os
import pickle
import momepy
import geopandas
import shapefile
import numpy as np
from shapely.geometry import Polygon
import math
import cv2
import pandas as pd
import json
import shutil
import csv

def minus(a):
    a = np.array([-x for x in a])
    return a

def distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

def mid(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

def fuc_1(a,b): # a是上传地块
    return (1-abs(float(a)-float(b))/float(a)) * 100

def detected(a):
    if a < 0:
        a = 0
    return round(a,2)

def picture_show(landuse_path):

    a = 1024
    b = 512
    landuse = shapefile.Reader(landuse_path, encoding='gbk')

    k = 0
    img = np.zeros((a, a, 3))
    img.fill(255)

    l_shape = landuse.shape(k)  #######要改
    l_convex = Polygon(l_shape.points).convex_hull
    x_c = l_convex.centroid.xy[0][0] #l_convex.centroid.xy:(array('d', [12945692.760656377]), array('d', [4861576.219346005]))
    y_c = l_convex.centroid.xy[1][0]

    l_dot = np.array(l_shape.points)

    l_nom_x = np.array(list(map(int, l_dot[:, 0]))) - int(x_c)
    l_nom_y = np.array(list(map(int, l_dot[:, 1]))) - int(y_c)
    l_inter = np.concatenate((l_nom_x[:, np.newaxis] + b, minus(l_nom_y)[:, np.newaxis] + b),1)  # nom_x[:, np.newaxis]新增一个维度
    cv2.polylines(img, [np.asarray(l_inter)], True, (0,0,255), 1) # cv2.polylines(画布，点坐标列表，封闭，颜色，宽度)polylines点坐标不能出现浮点型，需要是整型

    cv2.imwrite('./picture.jpg', img)

    landuse.close()
    return 'OK'



def compute_DFT(Shape):  # Shape = sf.shape(i)  ############改
    convex = Polygon(Shape.points).convex_hull
    x_c = convex.centroid.xy[0][0]
    y_c = convex.centroid.xy[1][0]  # convex.centroid.xy:(array('d', [12945692.760656377]), array('d', [4861576.219346005]))
    dot = np.array(Shape.points)
    nom_x = np.array(list(map(int, dot[:, 0]))) - int(x_c)
    nom_y = np.array(list(map(int, dot[:, 1]))) - int(y_c)

    final_x = []
    final_y = []
    for i in range(len(dot) - 1):
        ## 修改间隔，下面也要同步修改
        num = int(distance((nom_x[i], nom_y[i]), (nom_x[i + 1], nom_y[i + 1])) / 5)  # 整数
        final_x.append(float(nom_x[i]))
        final_y.append(float(nom_y[i]))
        if num != 0:
            tmp_x = (nom_x[i + 1] - nom_x[i]) / num
            tmp_y = (nom_y[i + 1] - nom_y[i]) / num
            for j in range(1, num):
                new_x = nom_x[i] + tmp_x * j
                new_y = nom_y[i] + tmp_y * j
                final_x.append(float(new_x))
                final_y.append(float(new_y))
    final_x.append(float(nom_x[-1]))
    final_y.append(float(nom_y[-1]))
    ##### 修改间隔，同上
    f_num = int(distance((nom_x[0], nom_y[0]), (nom_x[-1], nom_y[-1])) / 5)  # 第一个点和最后一个点之间计算
    if f_num != 0:
        f_tmp_x = (nom_x[0] - nom_x[-1]) / f_num  # 此时0是i+1，-1是i
        f_tmp_y = (nom_y[0] - nom_y[-1]) / f_num
        for t in range(1, f_num):
            f_new_x = nom_x[-1] + f_tmp_x * t
            f_new_y = nom_y[-1] + f_tmp_y * t
            final_x.append(float(f_new_x))
            final_y.append(float(f_new_y))
    dft = []
    l = len(final_x)
    for u in range(l):
        sum_k = 0
        for p in range(l):
            sum_k += complex(final_x[p], final_y[p]) * complex(math.cos(-2 * math.pi * u * p / l),
                                                               math.sin(-2 * math.pi * u * p / l))  # -2
        dft.append(abs(sum_k))
    fd = []
    for i in range(l):
        fd.append(dft[i] / dft[0])
    return fd, nom_x, nom_y

def compute_ori(Shape):
    cnt = np.array(Shape.points)
    convex = Polygon(Shape.points).convex_hull
    x_c = convex.centroid.xy[0][0]
    y_c = convex.centroid.xy[1][0]
    cnt = cnt - np.array([x_c, y_c])
    cnt = [list(map(int, item)) for item in cnt]
    rect = cv2.minAreaRect(np.asarray(cnt))  # rect[0]中心点坐标，rect[1]长和宽（但注意有别于正常的长短，是第一条碰到的边，rect[2]旋转角度）
    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
    box = np.int0(box)
    if distance(box[1], box[0]) < distance(box[3], box[0]):
        x1, y1 = mid(box[0], box[1])
        x2, y2 = mid(box[2], box[3])
        length = distance(box[3], box[0])
    else:
        x1, y1 = mid(box[0], box[3])
        x2, y2 = mid(box[2], box[1])
        length = distance(box[1], box[0])
    dy = abs(y2 - y1)
    c = dy / length
    if c > 1:
        c = 1
    if c < -1:
        c = -1
    angle = round(math.asin(c) / math.pi * 180)  # 反正弦出来的是弧度，弧度*180/pi变成角度
    if x2 < x1:
        angle = 180 - angle  # 范围0-180(大于0，到小于等于180，让所有为0的都变成180）
    if angle == 0:
        angle = 180
    return angle

def retrieval_similar_cases(data_path, cases_path, threshold):

    threshold = json.loads(threshold)['threshold']
    data = geopandas.read_file(data_path)  # print(data.shape) #(1, 46)
    cases = geopandas.read_file(cases_path)  # print(len(cases)) #1055
    sf_cases = shapefile.Reader(cases_path, encoding='gbk')
    sf_data = shapefile.Reader(data_path, encoding='gbk')
    change_list = ['bank', 'hospital', 'mall', 'school', 'subway']
    for l in change_list:
        cases.loc[cases[l].notnull(), l] = 1
        cases.loc[cases[l].isnull(), l] = 0
    cases = cases.fillna(value='nan')

    data['area'] = momepy.Area(data).series
    data['length'] = momepy.Perimeter(data).series
    data['ccd_means'] = momepy.CentroidCorners(data).mean
    data['ccd_std_stdev'] = momepy.CentroidCorners(data).std
    data['circ_comp'] = momepy.CircularCompactness(data).series  # 周长紧凑度
    data['cwa'] = momepy.CompactnessWeightedAxis(data).series  # 紧凑度加权轴
    data['convexity'] = momepy.Convexity(data).series  # 凸度
    data['corners'] = momepy.Corners(data).series  # 角数
    data['elongation'] = momepy.Elongation(data).series  # 伸长率
    data['eri'] = momepy.EquivalentRectangularIndex(data).series  # 等校矩形指数
    data['fractal'] = momepy.FractalDimension(data).series  # 分形维数
    data['rectangularity'] = momepy.Rectangularity(data).series  # 矩形度
    data['squ_comp'] = momepy.SquareCompactness(data).series  # 紧凑度指数
    data['long_ax'] = momepy.LongestAxisLength(data).series  # 最长轴的长度值
    data['shape_index'] = momepy.ShapeIndex(data, longest_axis='long_ax').series  # 形状索引

    cases['area'] = momepy.Area(cases).series
    cases['length'] = momepy.Perimeter(cases).series
    cases['ccd_means'] = momepy.CentroidCorners(cases).mean
    cases['ccd_std_stdev'] = momepy.CentroidCorners(cases).std
    cases['circ_comp'] = momepy.CircularCompactness(cases).series  # 周长紧凑度
    cases['cwa'] = momepy.CompactnessWeightedAxis(cases).series  # 紧凑度加权轴
    cases['convexity'] = momepy.Convexity(cases).series  # 凸度
    cases['corners'] = momepy.Corners(cases).series  # 角数
    cases['elongation'] = momepy.Elongation(cases).series  # 伸长率
    cases['eri'] = momepy.EquivalentRectangularIndex(cases).series  # 等校矩形指数
    cases['fractal'] = momepy.FractalDimension(cases).series  # 分形维数
    cases['rectangularity'] = momepy.Rectangularity(cases).series  # 矩形度
    cases['squ_comp'] = momepy.SquareCompactness(cases).series  # 紧凑度指数
    cases['long_ax'] = momepy.LongestAxisLength(cases).series  # 最长轴的长度值
    cases['shape_index'] = momepy.ShapeIndex(cases, longest_axis='long_ax').series  # 形状索引

    test_x = cases.iloc[:, 18:].sub(data.iloc[0, 2:], axis=1).abs().astype('float')

    ori = []
    dft = []
    him = []
    for i in range(len(cases)):
        # Ori
        ori.append(abs(compute_ori(sf_cases.shape(i)) - compute_ori(sf_data.shape(0))))
        # DFT
        fd_c, final_x_c, final_y_c = compute_DFT(sf_cases.shape(i))
        fd_d, final_x_d, final_y_d = compute_DFT(sf_data.shape(0))
        tmp = 0
        for k in range(20):
            tmp += math.pow((fd_c[k] - fd_d[k]), 2)
        dft.append(math.sqrt(tmp))
        # him
        inter = np.concatenate((final_x_c[:, np.newaxis], final_y_c[:, np.newaxis]),
                               1)  # nom_x[:, np.newaxis]新增一个维度
        inter = inter.reshape(len(final_x_c), 1, 2)  # !!!OpenCV找轮廓后，返回的ndarray的维数是(100, 1, 2)！！！而不是我们认为的(100, 2)。
        inter_d = np.concatenate((final_x_d[:, np.newaxis], final_y_d[:, np.newaxis]),
                                 1)  # nom_x[:, np.newaxis]新增一个维度
        inter_d = inter_d.reshape(len(final_x_d), 1,
                                  2)  # !!!OpenCV找轮廓后，返回的ndarray的维数是(100, 1, 2)！！！而不是我们认为的(100, 2)。
        him.append(cv2.matchShapes(inter, inter_d, 1, 0))

    test_x['area'] = test_x['area'] * 0.000001
    test_x['length'] = test_x['length'] * 0.001
    test_x['ori'] = ori
    test_x['dft'] = dft
    test_x['shape'] = him

    loaded_model = pickle.load(open('xgb.pickle.dat', 'rb'))

    xgb_pred = loaded_model.predict_proba(test_x)

    shutil.rmtree('result')
    os.mkdir('result')
    result = {}

    for i in range(len(xgb_pred)):
        if test_x['area'][i] < data['area'][0] * 0.25 and test_x['ori'][i] < 30 and xgb_pred[i][1] > threshold:  # 0.99,26
            information = {'编号ID': float(cases['ID'][i]), '地块名字': cases['NAME'][i], '地块所在地': cases['city'][i], '地块面积': cases['area'][i],'地块朝向': compute_ori(sf_cases.shape(i)), '容积率': cases['plot_area'][i], '价格': cases['price'][i],'绿化率': cases['greening_r'][i], '建成日期': cases['build_date'][i], 'school': float(cases['school'][i]),'mall': float(cases['mall'][i]), 'restaurant': float(cases['restaurant'][i]), 'hospital': float(cases['hospital'][i]),'subway': float(cases['subway'][i]), 'bank': float(cases['bank'][i]), 'park': float(cases['includ_g'][i]),'water': float(cases['includ_w'][i])}
            result[i] = json.dumps(information,sort_keys=True, indent=4, separators=(',', ': '),ensure_ascii=False)

            a = 1024
            b = 512
            img = np.zeros((a, a, 3))
            img.fill(255)
            landuse = sf_cases
            l_shape = landuse.shape(i)
            l_convex = Polygon(l_shape.points).convex_hull
            x_c = l_convex.centroid.xy[0][0]  # l_convex.centroid.xy:(array('d', [12945692.760656377]), array('d', [4861576.219346005]))
            y_c = l_convex.centroid.xy[1][0]
            l_dot = np.array(l_shape.points)
            l_nom_x = np.array(list(map(int, l_dot[:, 0]))) - int(x_c)
            l_nom_y = np.array(list(map(int, l_dot[:, 1]))) - int(y_c)
            l_inter = np.concatenate((l_nom_x[:, np.newaxis] + b, minus(l_nom_y)[:, np.newaxis] + b), 1)  # nom_x[:, np.newaxis]新增一个维度
            cv2.polylines(img, [np.asarray(l_inter)], True, (0, 0, 255),1)  # cv2.polylines(画布，点坐标列表，封闭，颜色，宽度)polylines点坐标不能出现浮点型，需要是整型
            cv2.imwrite('./result/' + str(cases['ID'][i]) + '.jpg', img)

    final_data = json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
    print(final_data)
    sf_cases.close()
    sf_data.close()
    del cases
    del data
    return final_data


def compute_similar(data_path,input,information_data):

    sf_data = shapefile.Reader(data_path, encoding='gbk')
    data = geopandas.read_file(data_path)
    data['area'] = momepy.Area(data).series

    data_ori = compute_ori(sf_data.shape(0))

    s_i = ['school', 'mall', 'restaurant', 'hospital', 'subway', 'bank', 'park', 'water']

    data_text = json.loads(input)
    text = json.loads(information_data)
    result = {}

    for ID in list(text.keys()): #['234', '345']
        temp = json.loads(text[ID])
        area = fuc_1(data['area'][0], temp['地块面积'])
        ori = fuc_1(data_ori, temp['地块朝向'])
        FAR = fuc_1(data_text['FAR'],temp['容积率'])
        price = fuc_1(data_text['price'],temp['价格'])
        greeningrate = fuc_1(data_text['greeningrate'],temp['绿化率'])

        tmp = 0
        for s in s_i:
            if data_text[s] != temp[s]:
                tmp += 1
        surrounding = (1-tmp/8) * 100

        tmp_result = {'面积':detected(area),'朝向':detected(ori),'容积率':detected(FAR),'价格':detected(price),'绿化率':detected(greeningrate),'周边环境':detected(surrounding)}
        result[ID] = json.dumps(tmp_result, sort_keys=True, indent=4, separators=(',', ': '),ensure_ascii=False)

    final_data = json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)

    sf_data.close()
    del data
    return final_data

def real_layout(cases_path,building_path,information_data):
    landuse = shapefile.Reader(cases_path, encoding='gbk')
    buildings = shapefile.Reader(building_path, encoding='gbk')

    information =json.loads(information_data)
    result_ID = list(information.keys())
    shutil.rmtree('layout')
    os.mkdir('layout')
    for ID in result_ID:

        a = 1024
        b = 512
        img = np.zeros((a, a, 3))
        img.fill(255)

        l_shape = landuse.shape(int(ID))  #######要改
        l_convex = Polygon(l_shape.points).convex_hull
        x_c = l_convex.centroid.xy[0][0]
        y_c = l_convex.centroid.xy[1][0]
        l_dot = np.array(l_shape.points)
        l_nom_x = np.array(list(map(int, l_dot[:, 0]))) - int(x_c)
        l_nom_y = np.array(list(map(int, l_dot[:, 1]))) - int(y_c)
        l_inter = np.concatenate((l_nom_x[:, np.newaxis] + b, minus(l_nom_y)[:, np.newaxis] + b),1)  # nom_x[:, np.newaxis]新增一个维度
        cv2.polylines(img, [np.asarray(l_inter)], True, (0, 0, 255),1)  # cv2.polylines(画布，点坐标列表，封闭，颜色，宽度)polylines点坐标不能出现浮点型，需要是整型

        for i in range(len(buildings)):
            if buildings.record(i)[3] == int(ID):

                b_shape = buildings.shape(i)
                b_dot = np.array(b_shape.points)
                b_nom_x = np.array(list(map(int, b_dot[:, 0]))) - int(x_c)
                b_nom_y = np.array(list(map(int, b_dot[:, 1]))) - int(y_c)
                b_inter = np.concatenate((b_nom_x[:, np.newaxis] + b, minus(b_nom_y)[:, np.newaxis] + b),1)  # nom_x[:, np.newaxis]新增一个维度
                floor_num = buildings.record(i)[2]
                if floor_num <= 3:  # 低层
                    grey = 210
                elif floor_num >= 4 and floor_num < 10:  # 多层
                    grey = 120
                elif floor_num >= 10:  # 高层
                    grey = 30

                cv2.fillPoly(img, [np.asarray(b_inter)], (grey, grey, grey))
        cv2.imwrite('./layout/' + str(ID) + '.jpg', img)

    landuse.close()
    buildings.close()
    return 'ok'


if __name__ == '__main__':
    os.chdir('/Users/yueyue/Desktop/演示系统设计/相似案例检索及推荐资料/demo')  #xgb.pickle.dat文件需在demo文件夾下
    data_path = './data/1.shp'
    cases_path = './cases/residential_mo.shp'
    building_path = './building/building.shp'
    # 显示图片功能
    picture_show(data_path)

    # 检索相似案例功能
    threshold = json.dumps({'threshold':0.999})
    information_data = retrieval_similar_cases(data_path, cases_path, threshold)

    # 计算雷达图的数值
    input = json.dumps({"FAR":1.7,"price":80000,"greeningrate":30,"school":1,"mall":0,"restaurant":1,"hospital":0,"subway":0,"bank":0,"park":0,"water":0})
    compute_similar(data_path, input, information_data)

    # 显示真实布局
    real_layout(cases_path, building_path, information_data)










