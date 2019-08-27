import torch
import numpy as np
import pickle 

import torchvision


class FeatureLoader(torch.utils.data.Dataset):

    def __init__(self, path_features, path_target, get_item_id=False, split='train'):
        self.path_features = path_features
        self.path_target = path_target

        self.get_item_id = get_item_id
        self.split=split

        self.data = self.readFile()
        self.target, self.ind = self.readPickle()

    def readFileh5(self):
        with h5py.File(self.path_features) as h5f:
            if h5f['train'].shape[0] != h5f['train_target'].shape[0]:
                print('Diferente cantidad de elementos y target en Train')
            if h5f['val'].shape[0] != h5f['val_target'].shape[0]
                print('Diferente cantidad de elementos y target en Val')

            return h5f[self.split]

    def readPickle(self):
        dbfile = open(self.path_target, 'rb')
        db = pickle.load(dbfile)

        return db[self.split]['target'], db[self.split]['index'] 

    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        item = self.data[index]
        target = self.target[index]
        ind = self.ind[index]

        # format data to torch
        if self.get_item_id:
            return (item, target, ind)
        else:
            return (item, target)

    def __len__(self):
        return len(self.data['train'])


if __name__ == '__main__':

    loader = VideoFolder(root="/mnt/nas2/GrimaRepo/jahurtado/codes/smth-smth-v2-baseline-with-models/data/s2s_feats_10percent.h5")

    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=0, pin_memory=True)

    start = time.time()
    for i, a in enumerate(tqdm(batch_loader)):
        if i > 100:
            break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
