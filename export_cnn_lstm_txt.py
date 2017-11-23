import pickle

with open('train_val_test_paths_labels.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

test_num_each = train_test_paths_labels[8]
test_paths =train_test_paths_labels[2]
test_labels =train_test_paths_labels[5]



with open('20171122_lstm_epoch_25_length_4_sgd_preds.pkl', 'rb') as f:
    ori_preds = pickle.load(f)
print("num of ori preds", len(ori_preds))
print("preds example: ", ori_preds[0])

root_dir = './phase'

phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']

preds_all = []
count = 0
labels_count = 0
for i in range(40):
    filename = './phase/video' + str(41 + i) + '-phase.txt'
    f = open(filename, 'a')
    f.write('Frame Phase')
    f.write('\n')
    preds_each = []
    labels_count += test_num_each[i]
    for j in range(count, count + test_num_each[i] - 3):
        if j == count:
            preds_each.append(ori_preds[j])
            preds_each.append(ori_preds[j])
            preds_each.append(ori_preds[j])

            preds_all.append(ori_preds[j])
            preds_all.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        preds_each.append(ori_preds[j])
        preds_all.append(ori_preds[j])
    for k in range(len(preds_each)):
        f.write(str(25* k))
        f.write('\t')
        f.write(phase_dict_key[preds_each[k]])
        f.write('\n')
    f.close()
    count += test_num_each[i] - 3



print('labels:', labels_count)
print('ori_preds count:', count)
print('rsult', len(preds_each))
print('rsult of all', len(preds_all))