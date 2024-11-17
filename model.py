from torch import nn
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from collections import defaultdict
import cv2
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

def setup_seed(seed=0):

    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    np.random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

def nor_std(data):
    data_sum = np.sum(data, axis=1).reshape(-1, 1)
    data_sum = np.where(data_sum == 0, 1, data_sum)
    data_TIC = data / data_sum

    b = np.percentile(data_TIC, 99.9, axis=0)
    return_data = np.zeros_like(data_TIC)

    for i in range(len(data_TIC[0])):
        da = data_TIC[:, i]
        return_data[:, i] = np.where(da > b[i], b[i], da)

    return_data = MinMaxScaler().fit_transform(return_data)

    return return_data

class ParametricUMAP(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ParametricUMAP, self).__init__()
        self.linear = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, latent_dim)

        self.bn = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        return x

def glorot_init(input_dim, output_dim):

    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    initial = initial.type(torch.float64)
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        x = inputs
        x = torch.mm(self.adj, x)
        x = torch.mm(x, self.weight)

        x = x.type(torch.float64)
        outputs = self.activation(x)

        return outputs

class GCN(nn.Module):
    def __init__(self, adj, input_dim, hidden_layer_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvSparse(input_dim=input_dim, output_dim=hidden_layer_dim, adj=adj)
        self.gcn2 = GraphConvSparse(input_dim=hidden_layer_dim, output_dim=hidden_layer_dim, adj=adj)
        self.gcn3 = GraphConvSparse(input_dim=hidden_layer_dim, output_dim=hidden_layer_dim, adj=adj, activation=lambda x:x)
        self.bn = nn.BatchNorm2d(hidden_layer_dim)

    def forward(self, x, m, n):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = x.type(torch.float32)
        x = self.bn(x.view(1, m, n, -1).permute(0, 3, 1, 2)) #[batch, feature, x, y]
        return x


# model = GCN(adj, 20, 100)
#
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")
#
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters: {trainable_params}")


def filtering(feature, m, n):
    feature = MinMaxScaler().fit_transform(feature)
    feature = feature.reshape(m, n, 20) * 255
    feature = cv2.medianBlur(np.uint8(feature), 3)
    feature = feature.reshape(m * n, 20) / 255
    return feature

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def GraphSlider(feature,input_shape):

    height, width = input_shape
    threhold = 0

    HE_max = []
    MSI_network = defaultdict(list)
    feature = filtering(feature, height, width)
    HE_feature = np.array(feature)

    dist = np.zeros([height, width, 8])

    for axix_x in range(height):
        for axix_y in range(width):
            location = (axix_x, axix_y)
            left = (axix_x, axix_y - 1)
            right = (axix_x, axix_y + 1)
            up = (axix_x - 1, axix_y)
            down = (axix_x + 1, axix_y)
            left_up = (axix_x - 1, axix_y - 1)
            left_down = (axix_x + 1, axix_y - 1)
            right_up = (axix_x - 1, axix_y + 1)
            right_down = (axix_x + 1, axix_y + 1)

            location_num = axix_y + axix_x * width
            left_num = axix_y - 1 + axix_x * width
            right_num = axix_y + 1 + axix_x * width
            up_num = axix_y + (axix_x - 1) * width
            down_num = axix_y + (axix_x + 1) * width
            left_up_num = axix_y - 1 + (axix_x - 1) * width
            left_down_num = axix_y - 1 + (axix_x + 1) * width
            right_up_num = axix_y + 1 + (axix_x - 1) * width
            right_down_num = axix_y + 1 + (axix_x + 1) * width

            HE_location_feature = HE_feature[location_num, :]

            if left[1] >= 0:
                HE_left_feature = HE_feature[left_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_left_feature)
                dist[axix_x, axix_y, 0] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(left_num)

            if right[1] <= width - 1:
                HE_right_feature = HE_feature[right_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_right_feature)
                dist[axix_x, axix_y, 1] = HE_distance
                HE_max.append(HE_distance)

                if HE_distance <= threhold:
                    MSI_network[location_num].append(right_num)

            if up[0] >= 0:
                HE_up_feature = HE_feature[up_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_up_feature)
                dist[axix_x, axix_y, 2] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(up_num)

            if down[0] <= height - 1:
                HE_down_feature = HE_feature[down_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_down_feature)
                dist[axix_x, axix_y, 3] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(down_num)

            if left_up[0] >= 0 and left_up[1] >= 0:
                HE_left_up_feature = HE_feature[left_up_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_left_up_feature)
                dist[axix_x, axix_y, 4] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(left_up_num)

            if left_down[1] >= 0 and left_down[0] <= height - 1:
                HE_left_down_feature = HE_feature[left_down_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_left_down_feature)
                dist[axix_x, axix_y, 5] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(left_down_num)

            if right_up[1] <= width - 1 and right_up[0] >= 0:
                HE_right_up_feature = HE_feature[right_up_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_right_up_feature)
                dist[axix_x, axix_y, 6] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(right_up_num)

            if right_down[1] <= width - 1 and right_down[0] <= height - 1:
                HE_right_down_feature = HE_feature[right_down_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_right_down_feature)
                dist[axix_x, axix_y, 7] = HE_distance
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(right_down_num)

    def judge_edge(data, threhold):
        mask = np.zeros([height * width])
        for i in range(data.shape[0]):
            flag = 1
            for number in data[i]:
                if number > threhold:
                    flag = 0
                    break
            if flag == 1:
                mask[i] = 1
        return mask

    threhold = np.mean(dist)
    data = dist.reshape(height * width, -1)

    mask = judge_edge(data, threhold)

    mask = mask.reshape(height, width)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.25)
    img = plt.imshow(mask)

    axcolor = 'lightgoldenrodyellow'
    axblur = plt.axes([0.15, 0.1, 0.75, 0.03], facecolor=axcolor)  # start_x, start_y, length, heightm
    max_dist_blur = np.max(dist)
    min_dist_blur = np.min(dist)
    sblur = Slider(axblur, 'Threhold', min_dist_blur, max_dist_blur, valinit=threhold)

    def update(val):
        threhold = sblur.val
        img.set_data(judge_edge(data, threhold).reshape(height, width))
        fig.canvas.draw_idle()

    sblur.on_changed(update)
    plt.show()
