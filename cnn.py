import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import time
from torch.nn import DataParallel
import os
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
import time

import pickle
import numpy as np

use_gpu = torch.cuda.is_available()

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader = pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        # self.target_transform=target_transform
        self.loader = loader


    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    test_paths = train_test_paths_labels[1]
    train_labels = train_test_paths_labels[2]
    test_labels = train_test_paths_labels[3]
    # print(len(train_paths))
    # print(len(test_paths))
    # print(train_labels.shape)
    # print(test_labels.shape)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)
    return train_dataset, test_dataset


def train_model(train_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
    num_train = len(train_dataset)
    batch_size = train_loader.batch_size
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    if use_gpu:
        model = model.cuda()
    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    epoches = 100
    for epoch in range(epoches):
        model.train()
        exp_lr_scheduler.step()
        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()
        for data in train_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_2)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            _,preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            train_corrects += torch.sum(preds == labels.data)
        train_elapsed_time = time.time()- train_start_time

        print('epoch: {:4d} train completed in: {:.0f}m{:.0f}s'.format(epoch, train_elapsed_time // 60, train_elapsed_time % 60))

        # train_average_loss = train_loss / num_train
        train_accuracy = train_corrects / num_train
        print('accuracy', train_accuracy)
        # print('train loss: {:.4f} accuracy: {:.4f}'.format(train_average_loss, train_accuracy))
    torch.save(model, '20171114_epoch_25.pkl')

def test_model(test_dataset):
    test_start_time = time.time()
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    num_test = len(test_dataset)
    batch_size = test_loader.batch_size
    model = torch.load('20171113_epoch_25_.pkl')
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    if use_gpu:
        model = model.cuda()
    for data in test_loader:
        inputs, labels_1, labels_2 = data
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels_2.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels_2, volatile=True)
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs.data, 1)
        test_corrects += torch.sum(preds == labels.data)
    test_elapsed_time = time.time() - test_start_time
    print('test completed in: {:.0f}m{:.0f}s'.format(
            test_elapsed_time // 60, test_elapsed_time % 60))

    test_accuracy = test_corrects / num_test
    print('accuracy', test_accuracy)

def main():

    train_dataset, test_dataset = get_data('train_test_paths_labels.pkl')

    train_model(train_dataset)
    # test_model(test_dataset)
if __name__ == "__main__":
    main()

print('Done')
print()
