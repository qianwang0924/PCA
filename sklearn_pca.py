from sklearn import decomposition
import numpy as np


def draw_3D(data):
    data = data.transpose()
    fig = plt.figure()
    
    ax = Axes3D(fig)
    ax.scatter(data[0], data[1], data[2])
    
    plt.show()


def draw_2D(data):
    plt.scatter(data[0].tolist(),data[1].tolist(),s=10)#s为设置点的大小
    plt.show()
 
data = np.random.normal(size=[500,3])
pca = decomposition.PCA(n_components=2)
pca.fit(data)
X = pca.transform(data)