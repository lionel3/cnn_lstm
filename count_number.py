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
    print(train_labels[0])
    print(len(test_labels))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_labels_1 = train_labels[:, 0:7]
    train_labels_2 = train_labels[:, -1]
    test_labels_1 = test_labels[:, 0:7]
    test_labels_2 = test_labels[:, -1]


    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0

    for i in range(train_labels_2.shape[0]):
        if train_labels_2[i] == 0:
            sum1 += 1
        elif train_labels_2[i] == 1:
            sum2 += 1
        elif train_labels_2[i] == 2:
            sum3 += 1
        elif train_labels_2[i] == 3:
            sum4 += 1
        elif train_labels_2[i] == 4:
            sum5 += 1
        elif train_labels_2[i] == 5:
            sum6 += 1
        elif train_labels_2[i] == 6:
            sum7 += 1

    print(sum1, sum2, sum3, sum4, sum5, sum6, sum7)
    print(sum1+sum2+sum3+sum4+sum5+sum6+sum7)

    # 以上统计train里phase的个数， 下面把test里phase的个数加上

    for i in range(test_labels_2.shape[0]):
        if test_labels_2[i] == 0:
            sum1 += 1
        elif test_labels_2[i] == 1:
            sum2 += 1
        elif test_labels_2[i] == 2:
            sum3 += 1
        elif test_labels_2[i] == 3:
            sum4 += 1
        elif test_labels_2[i] == 4:
            sum5 += 1
        elif test_labels_2[i] == 5:
            sum6 += 1
        elif test_labels_2[i] == 6:
            sum7 += 1
    print(sum1, sum2, sum3, sum4, sum5, sum6, sum7)
    print(sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7)

    # 统计tool个数
    x3 = np.sum(train_labels_1, axis=0)
    print(x3)
    x4 = np.sum(test_labels_1, axis=0)
    print(x4)
    x5 = x3 + x4
    print(x5)


if __name__ == "__main__":
    main()

print('Done')
print()
