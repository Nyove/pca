import numpy as np
import os, glob, random, cv2
import matplotlib.pyplot as plt


def load_images(root=u'./FaceDB_orl', split_rate=0.8):  # 加载图像集，随机选择sampleCount张图片用于训练
    dataset = []  # 总的数据集
    x_train = [] # 训练集
    x_test = []  # 测试集
    y_train = []
    y_test = []
    dataset_y = []  # 总的标签

    # 遍历40个文件夹
    for k in range(40):
        folder = os.path.join(root, '%03d' % (k + 1))   # 当前文件夹

        # dataset是一个三维数组，(10, 112, 92)
        dataset = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder, '*.png'))]    # ①glob.glob()返回一个路径列表；②cv2.imread()读取灰度图，0表示灰度图模式

        data_train_num = int(np.array(dataset).shape[0] * split_rate)

        data_train_indexs = random.sample(range(10), data_train_num)  # random.data_train_indexs()从0-9中随机选择sampleCount个元素，return a new list

        x_train.extend([dataset[i].ravel() for i in range(10) if i in data_train_indexs])
        x_test.extend([dataset[i].ravel() for i in range(10) if i not in data_train_indexs])

        y_train.extend([k] * data_train_num)
        y_test.extend([k] * (10 - data_train_num))
        dataset_y.extend([k] * 10)    # 将文件夹名作为标签

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def pca(x_train, dim=100):
    '''
    主成分分析，将10304维度的数据降维到100维
    :param x_train: 训练集
    :param dim: 降到k维
    :return:
    '''
    x_train = np.asmatrix(x_train, np.float32)    # 转换成矩阵
    num_train = x_train.shape[0]  # 取矩阵的维度 → (320, 10304)

    # 求每一行的均值
    data_mean = np.mean(x_train, axis=0)  # axis = 0：压缩行，对各列求均值 → 1 * n 矩阵

    # 零均值化：让矩阵X_train减去每一行的均值，得到零均值化后的矩阵Z
    Z = x_train - np.tile(data_mean, (num_train, 1)) # np.tile()用于复制，这里让data_mean.shape从(1, 10304) → (320, 10304)

    D, V = np.linalg.eig(Z * Z.T)  # 求协方差矩阵的特征值与特征向量

    # V1.shape - (320,100)
    V1 = V[:, 0:dim]  # 按列取前dim个特征向量（降到多少维就取前多少个特征向量）

    V1 = Z.T * V1  # 小矩阵特征向量向大矩阵特征向量过渡

    # 降维 - Z*V1
    return np.array(Z * V1), data_mean, V1


def predict(xTrain, yTrain, num_train, data_mean, x_test, V):
    '''
    预测目标图片
    :param xTrain:
    :param yTrain:
    :param num_train:
    :param data_mean:
    :param x_test: 待测试的图片
    :param V:
    :return:
    '''
    # 降维处理
    x_test_low_dim = np.array((x_test - np.tile(data_mean, (1, 1))) * V)

    predict_result = yTrain[np.sum((xTrain - np.tile(x_test_low_dim, (num_train, 1))) ** 2, axis=1).argmin()]

    print('识别的编号为 %d' % (predict_result + 1))


def main():
    x_train, y_train, x_test, y_test = load_images()    # (320,10304); (320); (80, 10304); (80);
    num_train, num_test = x_train.shape[0], x_test.shape[0]

    # 训练pca模型
    print("Start Traning.")
    x_train_low_dim, data_mean, V = pca(x_train)    # shape(320, 100)
    print("Finish Traning.")

    # 降维处理 shape(80, 100)
    x_test_low_dim = np.array((x_test - np.tile(data_mean, (num_test, 1))) * V)

    # ①argmin()求最小值的索引 ②求欧式距离后将列压缩成一列 → (320, 1)
    predict_results = [y_train[np.sum((x_train_low_dim - np.tile(d, (num_train, 1))) ** 2, axis=1).argmin()]
                   for d in x_test_low_dim]
    print(u'欧式距离法识别率: %.2f%%' % ((predict_results == y_test).mean() * 100))

    print("\nStart Predicting.")
    testImg = "./test.png"
    predict(x_train_low_dim, y_train, num_train, data_mean, cv2.imread(testImg, 0).ravel(), V)
    print("Finish Predicting.")


if __name__ == '__main__':
    main()
