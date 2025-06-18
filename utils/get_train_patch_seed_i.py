import torch
import numpy as np
import torch.utils.data as Data


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)

    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize

    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]

    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]

    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]

    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    return mirror_hsi


def get_train_data(mirror_image, band, train_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)

    return x_train.squeeze()


def get_train_patch_seed_i(img_size, img, train_number, batch_size):
    height, width, band = img.shape
    data_label = np.ones([height, width])
    position = np.array(np.where(data_label == 1)).transpose(1, 0)
    selected_i = np.random.choice(position.shape[0], int(train_number), replace=False)
    selected_i = position[selected_i]

    TR = np.zeros(data_label.shape)
    for i in range(int(train_number)):
        TR[selected_i[i][0], selected_i[i][1]] = 1
    total_pos_train = np.argwhere(TR == 1)
    mirror_img = mirror_hsi(height, width, band, img, patch=img_size)
    img_train = get_train_data(mirror_img, band, total_pos_train,
                               patch=img_size, band_patch=1)

    img_train = torch.from_numpy(img_train.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    Label_train = Data.TensorDataset(img_train)
    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)

    return label_train_loader
