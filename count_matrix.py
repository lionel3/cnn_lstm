import pickle
import numpy as np


def main():

    # 统计数据个数

    data_path = 'train_val_test_paths_labels.pkl'
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_labels = train_test_paths_labels[3]
    test_labels = train_test_paths_labels[5]

    print(len(train_labels))
    print(len(test_labels))

    # 输出训练集label的一个例子，应该是8维的一个向量，前七个代表tool，最后一个代表phase
    print(train_labels[0])

    # 转化为int64的numpy数组
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_labels_1 = train_labels[:, 0:7]
    train_labels_2 = train_labels[:, -1]
    test_labels_1 = test_labels[:, 0:7]
    test_labels_2 = test_labels[:, -1]


    # 计算映射矩阵，0代表存在，1代表不存在，phase对tool以及tool对phase都要求

    # 计算映射分布矩阵，求出每一个phase在tool中的比例，以及每一个tool在phase中的比例


if __name__ == "__main__":
    main()

print('Done')
print()
