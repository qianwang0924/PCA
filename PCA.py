import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
n = 500
k =3

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


def draw_3D(data):
    data = data.transpose()
    fig = plt.figure()
    
    ax = Axes3D(fig)
    ax.scatter(data[0], data[1], data[2])
    
    plt.show()


def draw_2D(data):
    plt.scatter(data[0].tolist(),data[1].tolist(),s=10)#s为设置点的大小
    plt.show()
 
def PCA_main():
    iris = datasets.load_iris()
    X = iris.data


    #data = np.random.normal(size=[n,k])
    mean_data = mean(X)#分布的点中心化
    #print(mean_data)

    cov_data = cov(mean_data)#求协方差矩阵
    point, point_Vec = get_point(cov_data)#求协方差矩阵的特征值和特征向量
    #print(point_Vec)
    index = np.argsort(-point)#排序后返回下标
    #print(index)
    selectVec = np.matrix(point_Vec.T[index[:3]])
    finalData = mean_data * selectVec.T
    print(finalData.T)
    draw_3D(finalData)#画出降维后的3d图

PCA_main()



 


