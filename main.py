import os
import numpy as np
import torch
import scipy.io as scio
import torch.utils
import torch.utils.data
from torch import nn
from time import time
from tqdm import tqdm
from model.model import SSST_GAN
from model.Disc import Discriminator
from utils.load_data import load_data
from utils.get_train_patch_seed_i import get_train_patch_seed_i
from utils.get_whole import whole_get_train_and_test_data
from utils.loss import SAD_loss
from utils.result_em import result_em

output_path = './result_out/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training on', device)

cases = ['Urban']
case = cases[0]
epochs = 100


# ---------------------weights_init---------------------
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


def train(case, lr=1e-4, beta=0.5, gamma=7e-4, lamda=0.05):
    # -----------load dataset and set parameter-----------
    data_name = case
    print('data_name:', data_name)
    img_3d, endmember_GT, abundance_GT, init_em = load_data(data_name)
    print('img_3d.shape:', img_3d.shape)
    H, W, Channels = img_3d.shape

    # -----------set model-----------
    model = SSST_GAN(32, 162, 2, 64, 768, 4).to(device)
    model.apply(weights_init)
    model.decoder.weight.data = torch.tensor(init_em).to(device)

    discriminator = Discriminator(162, 4).cuda()
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    label_train_loader = get_train_patch_seed_i(32, img_3d, 64, 32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------train-----------
    tic = time()
    criterionMSE = torch.nn.MSELoss()
    losses = []
    l = 0.0
    for epoch in tqdm(range(epochs)):
        model.train()
        for i, data in enumerate(label_train_loader):
            data = data[0].cuda()
            abunds, output, endmembers, out_nl = model(data)

            # D
            D_fake_score1, D_fake_score3, D_fake_score5, D_fake_score = discriminator(output.detach())
            D_fake_loss = criterionMSE(D_fake_score, torch.ones_like(D_fake_score))
            D_real_score1, D_real_score3, D_real_score5, D_real_score = discriminator(data.detach())
            D_real_loss = criterionMSE(D_real_score, torch.zeros_like(D_real_score))
            D_adv = (D_fake_loss + D_real_loss) / 2
            D_loss = beta * D_adv
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # G
            loss_sad = SAD_loss(data, output)

            endmembers_mean = endmembers.mean(dim=1, keepdim=True)
            loss_tv = gamma * ((endmembers - endmembers_mean) ** 2).sum() / endmember_GT.shape[1] / \
                      endmember_GT.shape[0]

            G_real_score1, G_real_score3, G_real_score5, G_real_score = discriminator(data)
            G_real_loss = criterionMSE(G_real_score, torch.zeros_like(G_real_score))
            G_fake_score1, G_fake_score3, G_fake_score5, G_fake_score = discriminator(output)
            G_fake_loss = criterionMSE(G_fake_score, torch.ones_like(G_fake_score))
            G_adv = (G_fake_loss + G_real_loss) / 2

            loss = loss_sad + loss_tv + lamda * G_adv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l += loss.item()

        losses.append(loss.detach().cpu().numpy())
    toc = time()
    scio.savemat(output_path + 'loss.mat', {'loss': losses})

    # -----------result-----------
    model.eval()
    with torch.no_grad():
        _, label_test_loader, num_H, num_W, img_size_test = whole_get_train_and_test_data(32, img_3d, 32)

        endmembers = model.decoder.weight.data.cpu().numpy()
        num_em = 4
        pred = torch.zeros([num_W, num_H, num_em, img_size_test, img_size_test])

        for batch_idx, (batch_data) in enumerate(label_test_loader):
            batch_data = batch_data[0].cuda()
            batch_pred = model.getAbundances(batch_data).detach().cpu()
            pred[batch_idx] = batch_pred
        pred = torch.permute(pred, [2, 0, 3, 1, 4])
        abundances = np.reshape(pred, [num_em, num_H * img_size_test, num_W * img_size_test])

        abundances = abundances[:, :H, :W]
        abundances = np.array(abundances)
        abundances = abundances.reshape(num_em, -1)
        abundance_GT = abundance_GT.reshape(num_em, -1)
        EM_hat = np.reshape(endmembers, (Channels, num_em))

        dev = np.zeros([num_em, num_em])
        for i in range(num_em):
            for j in range(num_em):
                dev[i, j] = np.mean((abundances[i, :] - abundance_GT[j, :]) ** 2)
        pos = np.argmin(dev, axis=0)

        A_hat = abundances[pos, :]
        EM_hat = EM_hat[:, pos]

        em_sad, asad_em, armse_em, class_rmse, armse, armse_a = result_em(EM_hat, endmember_GT.T, A_hat, abundance_GT)

        scio.savemat(output_path + 'results.mat', {'EM': EM_hat.T,
                                                   'A': A_hat})

        return em_sad, asad_em, armse_em, class_rmse, armse, armse_a, toc - tic


if __name__ == '__main__':
    em_sad, asad_em, armse_em, class_rmse, armse, armse_a, tim = train(case)

