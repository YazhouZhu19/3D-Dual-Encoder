from __future__ import print_function, division
import torch
import torch.optim as optim
import numpy as np
import time
import argparse
# from network import *
from model import *
# from loss import *
from dice_loss import *
import torch.nn.functional as F
import os
from os.path import join


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_data():
    image_name = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/finalImage350.npy"
    mask_name = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/finalMask350.npy"
    # image_name = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/image/BraTS20_Training_009.npy"
    # mask_name = r"C:/Users/Windows/Desktop/ZYZ/BraTS/BraTS_training_data/preprocessed data/mask/BraTS20_Training_009.npy"
    image = np.load(image_name)
    # the size of image is (350, 160, 160, 160, 4) (batch_size, depth, height, width, channel)
    mask = np.load(mask_name)
    # the size of mask is (350, 160, 160, 160, 3) (batch_size, depth, height, width, channel)

    # image = np.reshape(image, (1, 160, 160, 160, 4))
    # mask = np.reshape(mask, (1, 160, 160, 160, 3))

    image = np.transpose(image, (0, 4, 1, 2, 3))  # (batch_size, channel, depth, height, width)
    mask = np.transpose(mask, (0, 4, 1, 2, 3))  # (batch_size, channel, depth, height, width)
    return image, mask


def minibatch(images, GT, batch_size):
    n = len(images)
    for i in range(0, n, batch_size):
        yield images[i:i+batch_size], GT[i:i+batch_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['2001'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['2'],
                        help='batch_size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.0001'], help='initial learning rate')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true', help='Save output images and masks')
    parser.add_argument('--num_epoch_decay', type=int, default=70)

    args = parser.parse_args()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # config
    model_name = "3dU-net network"
    num_epoch = int(args.num_epoch[0])
    num_epoch_decay = int(args.num_epoch_decay)
    batch_size = int(args.batch_size[0])

    Nx, Ny = 256, 256
    save_fig = args.savefig
    save_every = 5

    images, GT = get_data()
    seg_net = seg_net()
    # criterion_DICE = DiceLoss()
    # criterion_DICE = torch.nn.BCELoss()
    # criterion_DICE = SoftDiceLoss()
    # criterion_DICE = GeneralizedDiceLoss()
    criterion_DICE_1 = IoULoss()
    criterion_DICE_2 = TverskyLoss()
    criterion_DICE_3 = AsymLoss()

    optimizer = optim.Adam(seg_net.parameters(), lr=float(args.lr[0]), betas=(0.5, 0.999))

    if cuda:
        seg_net = seg_net.cuda()
        criterion_DICE_1.cuda()
        criterion_DICE_2.cuda()
        criterion_DICE_3.cuda()

    i = 0
    loss_list = []
    for epoch in range(num_epoch):
        t_start = time.time()
        train_err = 0.
        accuracy = 0.
        train_batches = 0
        mini_batch = 0
        length = 0
        for im_1, im_2 in minibatch(images, GT, batch_size):

            im_1 = torch.from_numpy(im_1).type(Tensor)
            im_2 = torch.from_numpy(im_2).type(Tensor)

            # im_1 = im_1.type(Tensor)
            SR = seg_net(im_1)

            SR = torch.sigmoid(SR)
            # SR = SR.contiguous().view(-1)
            # im_2 = im_2.contiguous().view(-1)

            optimizer.zero_grad()
            loss_DICE = criterion_DICE_1(SR, im_2) + criterion_DICE_2(SR, im_2) + criterion_DICE_3(SR, im_2)

            loss = loss_DICE
            loss.backward()
            optimizer.step()
            print("the loss of minibatch_{:d} is:\t\t {:.6f}".format(mini_batch, loss))
            train_err += loss.item()
            mini_batch += 1
        t_end = time.time()
        train_err /= mini_batch
        loss_list.append(train_err)
        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        if epoch % 1000 == 0:
            SR_final = SR.cpu()
            SR_final = SR_final.detach().numpy()
            SR_name = "./result_3/final_output" + "_" + str(epoch) + ".npy"
            print(SR_name)
            np.save(SR_name, SR_final)
            print("the result is saved")

        if epoch % 1000 == 0:
            name = model_name + str(epoch) + ".npz"
            save_dir = r"./model_3/"
            torch.save(seg_net.state_dict(), join(save_dir, name))
            print("the model has been saved")
    np.save("loss.npy", loss_list)






