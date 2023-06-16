

import torch
# import torchvision
# import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
# import string
import numpy as np



class Demostration(Dataset):
    def __init__(self, path):
        Expert_Trajs = np.load(path)
        print(Expert_Trajs.shape)
        #self.Expert_Traj = Expert_Trajs[:512]
        self.Expert_Traj = Expert_Trajs
        # for i in range(512):
        #     self.Expert_Traj[i] = Expert_Trajs[i*6]


    def __getitem__(self, index):
        Expert_Trajs = self.Expert_Traj[index]


        return torch.FloatTensor(Expert_Trajs)

    def __len__(self):
        return len(self.Expert_Traj)


if __name__ == "__main__":
    d = Demostration('/home/xuxh/Level_recognition/data/val_list.txt')

    print("000")
