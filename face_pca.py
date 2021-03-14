import numpy as np
from PIL import Image
import os


from sklearn import decomposition
from sklearn import datasets

file_path='D:\BaiduNetdiskDownload\orl_faces\s1'#自己的pgm文件路径
output_dir='D:\BaiduNetdiskDownload\orl_faces\s1\s1_tran'#自己的jpg文件路径
n_components=10

def tran_jpg(file_path,output_dir):
    files = os.listdir(file_path)
    print(files)
    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for file in range(1,10):
        img = Image.open("D:\BaiduNetdiskDownload\orl_faces\s1\\"+ str(file) + '.pgm')
        img.save(os.path.join(output_dir,str(file) +'.jpg'))


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
    pca_data=[]
    

    for i in data:
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(i)
        
        eigenvector = pca.components_
        X = pca.transform(i)
        econstruct = np.dot(X,eigenvector[:n_components])
        img = Image.fromarray(econstruct)
        img.show()


PCA()


