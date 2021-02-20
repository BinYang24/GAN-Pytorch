import os
import glob
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim
from torch.utils.data import Dataset, DataLoader
import random


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_dataset(root):
    fname = glob.glob(os.path.join(root, '*'))
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize([64, 64]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    dataset = MyDataset(fname, transform)
    return dataset


class MyDataset(Dataset):
    def __init__(self, fname, transform):
        super(MyDataset, self).__init__()
        self.fname = fname
        self.transform = transform

    def __getitem__(self, idx):
        pic = self.fname[idx]
        img = cv.imread(pic)  # BGR
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.fname)


class Generator(nn.Module):
    # input N,in_dim
    # output N,3,64,64
    def __init__(self, indim, dim=64):
        super(Generator, self).__init__()

        def add_dimension(ind, outd):
            return nn.Sequential(nn.ConvTranspose2d(ind, outd, 5, 2, 2, 1, bias=False),
                                 nn.BatchNorm2d(outd),
                                 nn.ReLU())

        self.Linear = nn.Sequential(nn.Linear(indim, dim * 8 * 4 * 4, bias=False),
                                    nn.BatchNorm1d(dim * 8 * 4 * 4),
                                    nn.ReLU())
        self.l2 = nn.Sequential(add_dimension(dim * 8, dim * 4), add_dimension(dim * 4, dim * 2),
                                add_dimension(dim * 2, dim), nn.ConvTranspose2d(dim, 3, 5, 2, 2, 1), nn.Tanh())
        self.apply(weights_init)

    def forward(self, x):
        x = self.Linear(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.l2(x)
        return x


class Discriminator(nn.Module):
    # input N,3,64,64
    # out N,
    def __init__(self, indim, dim=64):
        super(Discriminator, self).__init__()

        def cnn(ind, outd):
            return nn.Sequential(nn.Conv2d(ind, outd, 5, 2, 2), nn.LeakyReLU(0.2))

        def cnnBatch(ind, outd):
            return nn.Sequential(nn.Conv2d(ind, outd, 5, 2, 2), nn.LeakyReLU(0.2)) #nn.BatchNorm2d(outd), nn.LeakyReLU(0.2))

        self.l = nn.Sequential(cnn(indim, dim), cnnBatch(dim, dim * 2), cnnBatch(dim * 2, dim * 4),
                               cnnBatch(dim * 4, dim * 8),
                               nn.Conv2d(dim * 8, 1, 4)), nn.Sigmoid())

        self.apply(weights_init)

    def forward(self, x):
        x = self.l(x)

        x = x.view(-1)
        # print(x.shape)
        return x


z_dim = 100
batch_size = 128
lr = 1e-4
n_epoch = 30

path = '/Data/faces'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = Generator(z_dim)
D = Discriminator(3)

same_seeds(0)

DataSet = get_dataset(path)

img = DataLoader(DataSet, num_workers=2, batch_size=batch_size, shuffle=True)
G.to(device)
D.to(device)
G.train()
D.train()
lambda_gp = 10
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()
z_sample = torch.randn(100, z_dim).to(device)
m=0
for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(img):
        imgs = data
        imgs = imgs.to(device)

        bs = imgs.size(0)

        """ Train D """
        z = torch.randn(bs, z_dim).to(device)
        r_imgs = imgs
        f_imgs = G(z)

        # label
        r_label = torch.ones((bs)).to(device)
        f_label = torch.zeros((bs)).to(device)

        # dis
        r_logit = D(r_imgs)
        f_logit = D(f_imgs.detach())

        # compute loss
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = (r_loss + f_loss) / 2

        # update model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()


        """ train G """
        # leaf
       	z = torch.randn(bs, z_dim).to(device)
		f_imgs = G(z)

		# dis
		f_logit = D(f_imgs)

		loss_G = criterion(f_logit, r_label)
		# update model
		G.zero_grad()
		loss_G.backward()
		opt_G.step()


        # log
            print(
            f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(img)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
            end='')
    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    torchvision.utils.save_image(f_imgs_sample, "SAMPLE.eps")
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    G.train()
