import numpy as np
import torch
import torch.utils.data as Data


def whole2_train_and_test_data(img_size, img):
    H0, W0, C = img.shape
    if H0 < img_size:
        gap = img_size - H0
        mirror_img = img[(H0 - gap):H0, :, :]
        img = np.concatenate([img, mirror_img], axis=0)
    if W0 < img_size:
        gap = img_size - W0
        mirror_img = img[:, (W0 - gap):W0, :]
        img = np.concatenate([img, mirror_img], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H + 1) * img_size - H
        mirror_img = img[(H - gap):H, :, :]
        img = np.concatenate([img, mirror_img], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        img = np.concatenate([img, mirror_img], axis=1)

    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)
    return sub_imgs, num_H, num_W


def whole_get_train_and_test_data(img_size, img_3d, batch_size):
    x_train, num_H, num_W = whole2_train_and_test_data(img_size, img_3d)
    x_test, num_H, num_W = whole2_train_and_test_data(img_size, img_3d)
    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

    Label_train = Data.TensorDataset(x_train)
    Label_test = Data.TensorDataset(x_test)

    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=num_H, shuffle=False)
    return label_train_loader, label_test_loader, num_H, num_W, img_size
