import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/big_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
        self.length = len(self.normalized_data)

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return self.length

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # print(self.normalized_data[idx][0:6])
        return {'input': self.normalized_data[idx][0:6], 'label': self.normalized_data[idx][6]}
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        train_length = int(len(self.nav_dataset) * 0.8)
        test_len = len(self.nav_dataset) - train_length
        train_dataset, test_dataset = data.random_split(self.nav_dataset, [train_length, test_len])
        self.train_loader = data.DataLoader(train_dataset, shuffle=False)
        self.test_loader = data.DataLoader(test_dataset, shuffle=False)
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
