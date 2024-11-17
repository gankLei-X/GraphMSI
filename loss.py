from torch import nn
import torch
import torch.nn.functional as F

class MaxEntropy(nn.Module):
    def __init__(self):
        super(MaxEntropy, self).__init__()

    def forward(self, c_i):
        p_i = F.softmax(c_i, dim=1)
        p_i = p_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i)).sum()
        return ne_i * 0.01

class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()
        self.loss_hpy = torch.nn.MSELoss()
        self.loss_hpz = torch.nn.MSELoss()

    def forward(self, embedding_image):

        hidden_layer_dim, m, n = embedding_image.shape

        embedding_image = embedding_image.reshape(m, n, -1)
        HPy_target = torch.zeros(m - 1, n, hidden_layer_dim)
        HPz_target = torch.zeros(m, n - 1, hidden_layer_dim)
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
        HPy = embedding_image[1:, :, :] - embedding_image[0:-1, :, :]
        HPz = embedding_image[:, 1:, :] - embedding_image[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, HPy_target) * 10
        lhpz = self.loss_hpz(HPz, HPz_target) * 10

        return lhpy + lhpz