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
    print(train_labels[50000])
    print(train_labels[80000])

    # 转化为int64的numpy数组
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_labels_1 = train_labels[:, 0:7]
    train_labels_2 = train_labels[:, -1]
    test_labels_1 = test_labels[:, 0:7]
    test_labels_2 = test_labels[:, -1]

    train_phase_tool = np.zeros([7, 7])
    for i in range(train_labels_2.shape[0]):
        for j in range(7):
            if train_labels_2[i] == j:
                for k in range(7):
                    if train_labels_1[i, k] == 1:
                        train_phase_tool[j, k] += 1
    print(train_phase_tool)

    test_phase_tool = np.zeros([7, 7])
    for i in range(test_labels_2.shape[0]):
        for j in range(7):
            if test_labels_2[i] == j:
                for k in range(7):
                    if test_labels_1[i, k] == 1:
                        test_phase_tool[j, k] += 1
    print(test_phase_tool)

    all_phase_tool = np.add(train_phase_tool, test_phase_tool)
    all_tool = np.sum(all_phase_tool, axis=0)
    print(all_tool)
    train_phase = [3758, 36886, 7329, 24119, 3716, 7219, 3277]
    all_phase = [8574, 74826, 14080, 58433, 7618, 14331, 6635]

    # tool到phase的映射矩阵
    all_tool_to_phase = (all_phase_tool / all_tool).transpose()
    print(all_phase_tool[0, 0])
    print(all_tool_to_phase[0, 0])

    # phase到tool的映射矩阵
    all_phase_to_tool = all_phase_tool / all_phase
    print(all_phase_to_tool[0, 0])

    print(all_tool_to_phase)
    print(all_phase_to_tool)



if __name__ == "__main__":
    main()

print('Done')
print()
