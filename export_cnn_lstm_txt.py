import pickle

with open('train_val_test_paths_labels.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

test_num_each = train_test_paths_labels[8]
test_paths = train_test_paths_labels[2]
test_labels = train_test_paths_labels[5]

sequence_length = 10

with open('cnn_lstm_epoch_25_length_10_opt_1_batch_400_train1_9993_train2_9971_val1_9692_val2_8647_preds_1.pkl', 'rb') as f:
    ori_preds = pickle.load(f)

num_labels = len(test_labels)
num_preds = len(ori_preds)
print('num of labels:', num_labels)
print("num of ori preds", num_preds)
print("labels example: ", test_labels[0][7])
print("preds example: ", ori_preds[0])

if num_labels == (num_preds + (sequence_length - 1) * 40):

    root_dir = './phase'
    phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
    preds_all = []
    count = 0
    for i in range(40):
        filename = './phase/video' + str(41 + i) + '-phase.txt'
        f = open(filename, 'a')
        f.write('Frame Phase')
        f.write('\n')
        preds_each = []
        for j in range(count, count + test_num_each[i] - (sequence_length - 1)):
            if j == count:
                for k in range(sequence_length - 1):
                    preds_each.append(ori_preds[j])
                    preds_all.append(ori_preds[j])
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        for k in range(len(preds_each)):
            f.write(str(25 * k))
            f.write('\t')
            f.write(phase_dict_key[preds_each[k]])
            f.write('\n')
        f.close()
        count += test_num_each[i] - (sequence_length - 1)
    test_corrects = 0

    for i in range(len(test_labels)):
        if test_labels[i][7] == preds_all[i]:
            test_corrects += 1

    print('the last video num label:', test_num_each[-1])
    print('the last video num preds:', len(preds_each))
    print('rsult of all', len(preds_all))

    print('right number:',test_corrects)
    print('test accuracy: {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')