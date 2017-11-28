import os
import pickle

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths = train_test_paths_labels[0]
    # val_paths = train_test_paths_labels[1]
    # test_paths = train_test_paths_labels[2]
    # train_labels = train_test_paths_labels[3]
    # val_labels = train_test_paths_labels[4]
    # test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    # val_num_each = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]
    print(train_num_each)
    print(test_num_each)
    return train_num_each, test_num_each


def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths

def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths

img_dir = '/home/lionel/cuhk/cholec/data_resize'
dst_dir = '/home/lionel/cuhk/cholec/useless_image'
img_dir_names, img_dir_paths = get_dirs(img_dir)

train_num_each, test_num_each = get_data('train_val_test_paths_labels.pkl')

for i in range(40):
    img_file_names, img_file_paths = get_files(img_dir_paths[i])
    # shutil 也可以move
    train_name = img_dir_names[i] + '-' + str(train_num_each[i] + 1) + '.jpg'
    test_name = img_dir_names[i+40] + '-' + str(test_num_each[i] + 1) + '.jpg'
    # print(train_name)
    # print(test_name)
    os.rename(os.path.join('/home/lionel/cuhk/cholec/data_resize/'+img_dir_names[i], train_name), os.path.join(dst_dir, train_name))
    os.rename(os.path.join('/home/lionel/cuhk/cholec/data_resize/'+img_dir_names[i+40], test_name), os.path.join(dst_dir, test_name))

