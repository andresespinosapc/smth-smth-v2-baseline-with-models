import torch
import numpy as np
import pickle 

import torchvision
import h5py

class FeatureLoader(torch.utils.data.Dataset):

    def __init__(self, path_features, path_target, get_item_id=False, split='train'):
        self.path_features = path_features
        self.path_target = path_target

        self.get_item_id = get_item_id
        self.split=split

        self.data = h5py.File(self.path_features, 'r')[self.split]

    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        item = self.data['data'][index]
        target = self.data['target'][index]
        ind = self.data['video_id'][index]

        # format data to torch
        if self.get_item_id:
            return (item, target, ind)
        else:
            return (item, target)

    def __len__(self):
        return len(self.target)


if __name__ == '__main__':

    loader = FeatureLoader(path_features="/mnt/nas2/GrimaRepo/jahurtado/codes/smth-smth-v2-baseline-with-models/data/s2s_feats_10percent.h5",
                        path_target="")

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
