import os
import os.path as osp
import shutil
import numpy as np
import h5py
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

class BigredDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 is_train=True,
                 is_validation=False,
                 is_test=False,
                 num_channel=5,
                 transform = None,
                 pre_transform=None,
                 pre_filter=None):
        self.is_train = is_train
        self.is_validation = is_validation
        self.is_test = is_test
        self.num_channel = num_channel
        super(BigredDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if(is_train==True, is_validation==False,is_test==False):
            path = self.processed_paths[0]
        elif(is_train==False, is_validation==True, is_test==False):
            path = self.processed_paths[1]
        elif(is_train==False, is_validation==False, is_test==True):
            path = self.processed_paths[2]

        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return ['all_files.txt']

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(s) for s in ['train', 'evaluation','test']]

    # def download(self):
    #     path = download_url(self.url, self.root)
    #     extract_zip(path, self.root)
    #     os.unlink(path)
    #     shutil.rmtree(self.raw_dir)
    #     name = self.url.split(os.sep)[-1].split('.')[0]
    #     os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        # filenames = filenames[:1]
        xs_train, ys_train = [], []
        xs_test, ys_test = [], []
        xs_validation, ys_validation = [], []

        data_xyz = 0
        train_data_list, test_data_list,validation_data_list = [],[],[]
        print(filenames)
        for filename in filenames:
            print(osp.join(self.raw_dir, filename))
            # f = h5py.File(osp.join(self.raw_dir, filename))
            with h5py.File(osp.join(self.raw_dir, filename)) as f:
                print('okay')
                data_xyz = f['xyz'][:]
                train_index = int(data_xyz.shape[0]*0.7)
                validation_index = int(data_xyz.shape[0]*0.9)
                test_index =  int(data_xyz.shape[0]*1)

                a = np.zeros((data_xyz.shape[0],data_xyz.shape[1],data_xyz.shape[2]+2))
                a[:,:,:3] = f['xyz'][:]
                a[:,:,3] = f['intensity'][:]
                a[:,:,4] = f['laserID'][:]

                a = a[:,:,:self.num_channel]
                xs_train += torch.from_numpy(a[:train_index,:,:]).to(torch.float).unbind(0)
                ys_train += torch.from_numpy(f['label'][:train_index]).to(torch.long).unbind(0)

                xs_validation += torch.from_numpy(a[train_index:validation_index,:,:]).to(torch.float).unbind(0)
                ys_validation += torch.from_numpy(f['label'][train_index:validation_index]).to(torch.long).unbind(0)

                xs_test += torch.from_numpy(a[validation_index:test_index,:,:]).to(torch.float).unbind(0)
                ys_test += torch.from_numpy(f['label'][validation_index:test_index]).to(torch.long).unbind(0)



        train_data_list, test_data_list,validation_data_list = [],[],[]
        for i, (x, y) in enumerate(zip(xs_train, ys_train)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            train_data_list.append(data)

        for i, (x, y) in enumerate(zip(xs_validation, ys_validation)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            validation_data_list.append(data)

        for i, (x, y) in enumerate(zip(xs_test, ys_test)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(validation_data_list), self.processed_paths[2])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
