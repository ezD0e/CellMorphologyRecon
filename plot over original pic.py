import csv

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import EllipseCollection
import numpy as np


from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage import color
from skimage import io

from PIL import Image

file_path = 'Z:/CodeCluster/ProcessedPics/C2-Probability maps.png'
raw_img = Image.open(file_path)
imgSize = raw_img.size
w = raw_img.width
h = raw_img.width

size = (w,h)

img_port = plt.imread('Z:/CodeCluster/ProcessedPics/C2-Probability maps.png')
img_greyscale = color.rgb2grey(color.rgba2rgb(img_port))
#img_greyscale = rotate(img_greyscale, angle =90,order = 0)

fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1)
ax1.imshow(img_greyscale,cmap=plt.cm.gray)
ax1.set_title('Original Greyscale Pic')


img = np.ones((1800,1200))  #新建内容为1的画布

#csv_reader_lines = csv.reader(open("Z:/CodeCluster/rawResult/Results04052022_1213complete.csv"))
date = [] #创建空集
cell_mark = 0 #计数
i = 0
with open('Z:/CodeCluster/rawResult/Results04052022_1213complete.csv','r') as read_file:
    reader = csv.reader(read_file)
    reader.__next__() #跳过首行
    for one_line in reader: #遍历行数
        date.append(one_line)
        cell_mark = cell_mark+1 #计数+1
        x = float(date[i][2]) #找坐标
        y = float(date[i][3])
        width = int(float(date[i][9])) #找长短轴
        height = int(float(date[i][10]))
        FeretAngle = float(date[i][18]) #找倾角
        while i < cell_mark and width < 250 and height < 250: 

            rr,cc = ellipse(x,y,width/2,height/2,rotation=np.deg2rad(FeretAngle)) #画椭圆
            img[rr,cc] = 0
            
            
            i = i+1



#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python


ax2.imshow(img,cmap=plt.cm.gray)
ax2.set_title('Rebuid Graph')
plt.show()