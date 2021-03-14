import numpy as np
from PIL import Image
import os
file_path='D:\BaiduNetdiskDownload\orl_faces\s1'#自己的pgm文件路径
output_dir='D:\BaiduNetdiskDownload\orl_faces\s1\s1_tran'#自己的jpg文件路径
n_components=10

def mean(data):#数据中心化
    mean = np.mean(data, axis=0)
    return(np.subtract(data,mean))

def cov(data):#求协方差矩阵
    x_T = np.transpose(data)
    x_cov = np.cov(x_T)
    return x_cov

def get_point(data):#求特征值特征向量
    
    point, point_Vec = np.linalg.eig(data)
    return(point, point_Vec)
 
def PCA_main(data):

    #data = np.random.normal(size=[n,k])
    mean_data = mean(data)#分布的点中心化
    #print(mean_data)

    cov_data = cov(mean_data)#求协方差矩阵
    point, point_Vec = get_point(cov_data)#求协方差矩阵的特征值和特征向量
    #print(point_Vec)
    index = np.argsort(-point)#排序后返回下标
    #print(index)
    selectVec = np.matrix(point_Vec.T[index[:n_components]])
    finalData = mean_data * selectVec.T
    reconData = (finalData * selectVec) + mean_data
    return(reconData)
    

def get_train_data(path):
    image=[]
    for id in range(1,10):
        id = str(id) + '.jpg'
        domain = os.path.abspath(path)#获取文件夹的路径

        info = os.path.join(domain,id)#将路径与文件名结合起来就是每个文件的完整路径
        img = np.array(Image.open(info))#读取图片内容
        image.append(img)
    return(image)

def PCA():
    data = get_train_data(output_dir)

    for i in data:
        pca_pic= PCA_main(i)
        img = Image.fromarray(pca_pic)
        img.show()

PCA()

