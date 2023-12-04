import csv

import math
from turtle import width
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np


from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)

size = (1088,1638) #选定画布尺寸
img = np.zeros((1800,1200))  #新建画布 （应调用size
#fig, ax = plt.subplots()

csv_reader_lines = csv.reader(open("Z:/CodeCluster/rawResult/Results04052022_1213complete.csv"))
date = [] #创建空集
cell_mark = 0 #计数
i = 0
with open('Z:/CodeCluster/rawResult/Results04052022_1213complete.csv','r') as read_file:
    reader = csv.reader(read_file)
    reader.__next__() #跳过首行
    for one_line in reader: #遍历行数
        date.append(one_line)
        cell_mark = cell_mark+1 #计数+1
        
        while i < cell_mark: 
            x = int(float(date[i][2])) #找坐标
            y = int(float(date[i][3]))
            width = int(float(date[i][9])) #找长短轴
            height = int(float(date[i][10]))
            FeretAngle = float(date[i][18]) #找倾角
            rr,cc = ellipse(x,y,width/2,height/2,rotation=np.deg2rad(FeretAngle)) #画椭圆
            img[rr,cc] = 1
            
            
            i = i+1
            

#use plt.matshow(img) to plot the graph in debug mod
i=0
plt.show()