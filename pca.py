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

        dataset = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder, '*.png'))]    # ①glob.glob()返回一个路径列表；②cv2.imread()读取灰度图，0表示灰度图模式
        # dataset是一个三维数组，(10, 112, 92)

        data_train_num = int(np.array(dataset).shape[0] * split_rate)

        data_train_indexs = random.sample(range(10), data_train_num)  # random.data_train_indexs()从0-9中随机选择sampleCount个元素，return a new list

        x_train.extend([dataset[i].ravel() for i in range(10) if i in data_train_indexs])
        x_test.extend([dataset[i].ravel() for i in range(10) if i not in data_train_indexs])

        y_train.extend([k] * data_train_num)
        y_test.extend([k] * (10 - data_train_num))
        dataset_y.extend([k] * 10)    # 将文件夹名作为标签

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def pca(x_train, k):
    '''
    主成分分析，将10304维度的数据降维到100维
    :param x_train: 训练集
    :param k: 降到k维
    :return:
    '''
    x_train = np.asmatrix(x_train, np.float32)    # 转换成矩阵
    rows, cols = x_train.shape  # 取矩阵的维度 → (320, 10304)

    # 求每一行的均值
    data_mean = np.mean(x_train, axis=0)  # axis = 0：压缩行，对各列求均值 → 1 * n 矩阵
    # 让矩阵X_train减去每一行的均值，得到数据的中心点
    Z = x_train - np.tile(data_mean, (rows, 1))

    D, V = np.linalg.eig(Z * Z.T)  # 求协方差矩阵, 并求出其特征值与特征向量

    V1 = V[:, :k]  # 按列取前k个特征向量（降到多少维就取前多少个特征向量）

    # 降维
    V1 = Z.T * V1  # 小矩阵特征向量向大矩阵特征向量过渡

    for i in range(k):  # 特征向量归一化
        V1[:, i] /= np.linalg.norm(V1[:, i])  # 特征向量归一化

    # 可解释性方差
    tot = sum(D)
    var_exp = [(i / tot) * 100 for i in sorted(D, reverse=True)]
    var_exp1 = var_exp[:k]
    # print(var_exp1)
    return np.array(Z * V1), data_mean, V1, var_exp1


def plt_showVariance_ratio(x_train):
    explained_variance_ratio = []
    for i in range(1, 150):
        xTrain, data_mean, V, var_exp = pca(x_train, i)
        ele = 0
        total = 0
        while (ele < len(var_exp)):
            total = total + var_exp[ele]
            ele += 1
        explained_variance_ratio.append(total)
    plt.plot(range(1, 150), explained_variance_ratio)
    plt.savefig('task2.jpg')
    plt.show()


def plt_showVFace(V, var_exp):
    # 显示100维特征脸
    flg, axes = plt.subplots(10, 10, figsize=(15, 20), subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(V.T[i, :].reshape(112, 92), cmap="gray")
        ax.set_title(round(var_exp[i], 2))
    plt.savefig('task.jpg')
    plt.show()


def plt_showAllDataMeanFace(data):
    # 平均脸
    average_face = np.mean(data, axis=0)  # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    # print(average_face)
    plt.imshow(average_face.reshape((112, 92)), cmap="gray")
    plt.title('全部图片的平均脸')
    plt.savefig('allDataFaceMean.jpg')
    plt.show()


def plt_showTrainDataMeanFace(mean):
    plt.imshow(mean.reshape((112, 92)), cmap="gray")
    plt.title('训练数据集的平均脸')
    plt.savefig('trainDataFaceMean.jpg')
    plt.show()


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
    xTest = np.array((x_test - np.tile(data_mean, (1, 1))) * V)
    yPredict = yTrain[np.sum((xTrain - np.tile(xTest, (num_train, 1))) ** 2, 1).argmin()]
    print('识别的编号为 %d' % (yPredict + 1))


def main():
    x_train, y_train, x_test, y_test = load_images()
    num_train, num_test = x_train.shape[0], x_test.shape[0]

    # 训练pca模型
    print("Start Traning.")
    xTrain, data_mean, V, var_exp = pca(x_train, 100)
    print("Finish Traning.")

    # 得到测试脸在特征向量下的数据
    xTest = np.array((x_test - np.tile(data_mean, (num_test, 1))) * V)

    # print("特征脸：")
    # plt_showVFace(V, var_exp)   # 同时将/图片保存在本目录下
    # print("所有数据的平均脸：")
    # plt_showAllDataMeanFace(xtotal)
    # print("训练数据的平均脸：")
    # plt_showTrainDataMeanFace(data_mean)

    yPredict = [y_train[np.sum((xTrain - np.tile(d, (num_train, 1))) ** 2, 1).argmin()] for d in xTest]
    print(u'欧式距离法识别率: %.2f%%' % ((yPredict == y_test).mean() * 100))

    print("\nStart Predicting.")
    testImg = "./test.png"
    predict(xTrain, y_train, num_train, data_mean, cv2.imread(testImg, 0).ravel(), V)
    print("Finish Predicting.")


if __name__ == '__main__':
    main()
