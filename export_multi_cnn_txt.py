import pickle

with open('train_val_test_paths_labels.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

test_num_each = train_test_paths_labels[8]
test_paths =train_test_paths_labels[2]
test_labels =train_test_paths_labels[5]


with open('cnn_lstm_epoch_25_length_10_opt_1_batch_400_train1_9993_train2_9971_val1_9692_val2_8647_preds_1.pkl', 'rb') as f:
    ori_preds = pickle.load(f)
print("num of ori preds", len(ori_preds))
print("preds example: ", ori_preds[0])

root_dir = './tool'

preds_all = []
count = 0
labels_count = 0
for i in range(40):
    filename = './tool/video' + str(41 + i) + '-tool.txt'
    f = open(filename, 'w+')
    f.write('Frame Tool1 Tool2 Tool3 Tool4 Tool5 Tool6 Tool7')
    f.write('\n')
    preds_each = []
    labels_count += test_num_each[i]
    for j in range(count, count + test_num_each[i]):
        f.write(str(25* (j - count)))
        f.write('\t')
        for k in range(7):
            f.write(str(ori_preds[j][k]))
            f.write('\t')
        f.write('\n')
    f.close()
    count += test_num_each[i]


print('labels:', labels_count)
print('ori_preds count:', count)
print('rsult', len(preds_each))
print('rsult of all', len(preds_all))