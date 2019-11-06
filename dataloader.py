import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Image size required by your neural network
HEIGHT = 128
WIDTH = 128

class DataLoader(object):

    def __init__(self, pathToFolders, transforms, batch_size = 4):
        self.classes = ['75','150','300','600']
        # reading data list
        self.list_img = []
        for folder in self.classes:
            self.list_img += [k for k in glob.glob(os.path.join(pathToFolders, folder,'*.tif'))]
        np.random.shuffle(self.list_img)
        # store the batch size
        self.batch_size = batch_size
        # initialize a cursor to keep track of the current image index 
        self.cursor = 0
        # store the number of batches
        self.num_batches = len(self.list_img) // batch_size
        self.index = {'75':0, '150':1, '300':2, '600':3}
        self.to_tensor = transforms

    def get_batch(self):
        # once we reach the end of the dataset, shuffle it again and reset cursor
        if self.cursor + self.batch_size > len(self.list_img):
            self.cursor = 0
            np.random.shuffle(self.list_img)
        # initialize the image tensor with arrays full of zeros
        imgs = torch.zeros(self.batch_size, 1, HEIGHT, WIDTH)
        # initialize the label tensor with zeros, 3 here is the size of one-hot encoded label for a 3-class classification problem
        labels_onehot = torch.zeros(self.batch_size, len(self.classes))
        labels = torch.LongTensor(self.batch_size)

        for idx in range(self.batch_size):
            # get the current file name pointed by the cursor
            full_img_path = self.list_img[self.cursor]
            # update cursor
            self.cursor += 1

            # read image in grayscale
            image = cv2.imread(full_img_path, 0)

            imgs[idx,0,:,:] = self.to_tensor(image)

            labels[idx] = self.index[full_img_path.split('/')[-2]]

        #labels_onehot.scatter_(1, labels, 1)

        return imgs, labels


def imshow(inp, title=None):
    # imshow for a tensor.
    #inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = inp.numpy()
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def showABatch(batch, title=None):
    imgs, labels = batch
    label_dict = {0:'600 dpi', 1:'500 dpi', 2:'400 dpi'}
    # ADD YOUR CODE HERE
    for i in range(len(batch)):
        plt.figure()
        imshow(imgs[i].squeeze(),'label: '+label_dict[torch.max(labels[i]).item()])
    plt.show()
