from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import lr_scheduler
from dataloader import DataLoader
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torchvision
import torch.optim as optim
import copy
from model import CNN
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is not available')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_variable(x,requires_grad=True):
    x.to(device)
    return Variable(x,requires_grad)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    writer = SummaryWriter('runs/optimal_resolution')
    train_writer = SummaryWriter('runs/optimal_resolution/train')
    val_writer = SummaryWriter('runs/optimal_resolution/val')
    # get some random training images
    images, _ = dataloaders['train'].get_batch()
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    matplotlib_imshow(img_grid, one_channel=True)
    # write to tensorboard
    writer.add_image('four_training_images', img_grid)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            num_batch = dataloaders[phase].num_batches
            batch_size = dataloaders[phase].batch_size
            # Iterate over data.
            for i in range(num_batch):
                inputs, labels = dataloaders[phase].get_batch()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (num_batch*batch_size)
            epoch_acc = running_corrects.double() / (num_batch*batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                # ...log the running loss
                train_writer.add_scalar('Loss', epoch_loss, epoch)
                # ...log the running accu
                train_writer.add_scalar('Accuracy', epoch_acc, epoch)
            else:
                # ...log the running loss
                val_writer.add_scalar('Loss', epoch_loss, epoch)
                # ...log the running accu
                val_writer.add_scalar('Accuracy', epoch_acc, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
 

def visualize_model(model, num_images=25):
    class_names = ['75','150','300','600']
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i in range(num_images):
            inputs, labels = dataloaders['test'].get_batch()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            ax = plt.subplot(num_images//5, 5, i+1)
            ax.axis('off')
            ax.set_title('Label: {}; Predicted: {}'.format(class_names[labels.item()], class_names[preds.item()]))
            plt.imshow(inputs.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
            plt.pause(0.001)

        model.train(mode=was_training)
    plt.show()


model_ft = CNN(in_channels=1, num_classes=4)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=70000, gamma=0.1)

# compose a series of random tranforms to do some runtime data augmentation
transformations = {x: transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]) for x in ['val', 'test']}
transformations['train'] = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
batch_size = {'train':64, 'val':64, 'test':1}
dataloaders = {x: DataLoader('./data/'+x, transformations[x], batch_size[x]) for x in ['train', 'val', 'test']}
num_batch = dataloaders['train'].num_batches# length of data / batch_size
print(num_batch)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=1)
#save the model weights
torch.save(model_ft.state_dict(), './model.pkl')

visualize_model(model_ft)