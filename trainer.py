from scipy import sparse
from torch import optim
import umap
from model import *
from collections import defaultdict
import numpy as np
from scipy.spatial import distance
import networkx as nx
import scipy.sparse as sp
from torch.autograd import Variable
from loss import *

def DimensionalityReduction(ori_data, args):

    ori_data = nor_std(ori_data)
    dim = args.n_components
    print("Start Dimensionality Reduction using Parametric-UMAP Model ...")

    m, n = ori_data.shape
    ytarget = umap.UMAP(metric='cosine', n_components=dim, random_state=0).fit_transform(ori_data)
    ori_data = torch.tensor(ori_data, dtype=torch.float)
    ytarget = torch.tensor(ytarget, dtype=torch.float)

    model = ParametricUMAP(input_dim = n, latent_dim = dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.2)
    ori_data = ori_data.cuda()
    ytarget = ytarget.cuda()

    LUMAP = nn.MSELoss()
    epochs = 1000
    for epoch in range(epochs):
        yPred = model(ori_data)
        loss = LUMAP(yPred, ytarget)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, '/', epochs, '|', '| loss:', loss.item())

    z = yPred.cpu().detach().numpy()
    torch.save(model.state_dict(), "ParametricUMAP.pt")
    return z

def GraphConstruction(feature, shape, threhold = 0.216):

    height, width = shape

    print("Start Graph Construction...")

    HE_max = []
    MSI_network = defaultdict(list)
    feature = filtering(feature, height, width)
    HE_feature = np.array(feature)

    for axix_x in range(height):
        for axix_y in range(width):

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
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(left_num)

            if right[1] <= width - 1:
                HE_right_feature = HE_feature[right_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_right_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(right_num)

            if up[0] >= 0:
                HE_up_feature = HE_feature[up_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_up_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(up_num)

            if down[0] <= height - 1:
                HE_down_feature = HE_feature[down_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_down_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(down_num)

            if left_up[0] >= 0 and left_up[1] >= 0:
                HE_left_up_feature = HE_feature[left_up_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_left_up_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(left_up_num)

            if left_down[1] >= 0 and left_down[0] <= height - 1:
                HE_left_down_feature = HE_feature[left_down_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_left_down_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(left_down_num)

            if right_up[1] <= width - 1 and right_up[0] >= 0:
                HE_right_up_feature = HE_feature[right_up_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_right_up_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(right_up_num)

            if right_down[1] <= width - 1 and right_down[0] <= height - 1:
                HE_right_down_feature = HE_feature[right_down_num, :]
                HE_distance = distance.euclidean(HE_location_feature, HE_right_down_feature)
                HE_max.append(HE_distance)
                if HE_distance <= threhold:
                    MSI_network[location_num].append(right_down_num)

    return MSI_network

def FeatureClustering(feature, graph, args):

    print("Start Feature Clustering...")
    m,n = args.input_shape
    hidden_layer_dim = 200
    feature_new = []
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feature_new.append(feature[i, j])
    feature_new = np.array(feature_new).reshape(feature.shape[0], feature.shape[1])
    features = sparse.lil_matrix(feature_new)

    objects = []
    objects.append(graph)
    graph = tuple(objects)

    num_nodes = feature.shape[0]
    network = nx.Graph()

    for num_node in range(num_nodes):
        network.add_node(num_node)
    for num_node in range(num_nodes):
        for num_edge in range(len(graph[0][num_node])):
            if graph[0][num_node][num_edge] <= num_nodes:
                network.add_edge(num_node, graph[0][num_node][num_edge])

    adj = nx.adjacency_matrix(network).todense()

    adj = sparse.csr_matrix(adj)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)

    adj_orig.eliminate_zeros()

    adj_train = adj

    adj_norm = preprocess_graph(adj)

    features = sparse_to_tuple(features.tocoo())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    adj_norm = adj_norm.cuda()

    if args.use_scribble:

        scribble = np.loadtxt(args.input_scribble)
        mask = scribble.reshape(-1)

        mask_inds = np.unique(mask)

        inds_sim = torch.from_numpy(np.where(mask == 0)[0])
        inds_scr = torch.from_numpy(np.where(mask != 0)[0])
        target_scr = torch.from_numpy(mask.astype('int64'))

        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()

        target_scr = Variable(target_scr)


    adj_norm = adj_norm.type(torch.float64)
    model = GCN(adj_norm, args.n_components, hidden_layer_dim)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")
    #
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {trainable_params}")

    # if args.use_scribble:
    #     print("load pre-train model")
    #     model.load_state_dict(torch.load("/DATA_2/zjy/segment_mouse_update/src/modelpara/sim/checkpoint_sim_500.pt"))
    model = model.cuda()

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss_sim = nn.CrossEntropyLoss()

    loss_tv = TotalVariation()
    loss_ent = MaxEntropy()
    loss_scr = nn.CrossEntropyLoss()
    features = features.cuda()

    epochs = 1000
    label_colors = np.random.randint(255, size=(hidden_layer_dim, 3))
    features = features.type(torch.float64)
    for epoch in range(epochs):

        embedding_img = model(features, m, n)[0]

        embedding = embedding_img.permute(1, 2, 0).contiguous()

        embedding = embedding.view(-1, hidden_layer_dim)

        optimizer.zero_grad()

        ignore, target = torch.max(embedding, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        if args.use_scribble:
            loss = loss_sim(embedding[inds_sim], target[inds_sim]) + \
                   loss_tv(embedding_img) + \
                   loss_ent(embedding) + \
                   loss_scr(embedding[inds_scr], target_scr[inds_scr])
        else:

            loss = loss_sim(embedding, target) + loss_ent(embedding) + loss_tv(embedding_img)

        loss.backward()
        optimizer.step()

        print(epoch, '/', epochs, '|', 'label num:', nLabels, '| loss:', loss.item())

        im_target_rgb = np.array([label_colors[c % hidden_layer_dim] for c in im_target])
        result_label = np.array([c for c in im_target])
        im_target_rgb = im_target_rgb.reshape(m, n, 3).astype(np.uint8)

        # if epoch % 20 == 0:
        #     cv2.imwrite("seg.png", im_target_rgb)
        #     np.savetxt("seg.csv", result_label,
        #                    delimiter=',')

    torch.save(model.state_dict(), "GCN.pt")

    return im_target


def Predicting(ori_data,args):
    ori_data = nor_std(ori_data)

    dim = args.n_components
    print("Predicting without re-training ...")

    m, n = ori_data.shape

    ori_data = torch.tensor(ori_data, dtype=torch.float)

    model = ParametricUMAP(input_dim=n, latent_dim=dim).cuda()
    model.load_state_dict(torch.load("ParametricUMAP.pt"))

    ori_data = ori_data.cuda()

    yPred = model(ori_data)

    feature = yPred.cpu().detach().numpy()

    graph = GraphConstruction(feature, args.input_Pshape,threhold = 0.6)

    m,n = args.input_Pshape

    hidden_layer_dim = 200
    feature_new = []
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feature_new.append(feature[i, j])
    feature_new = np.array(feature_new).reshape(feature.shape[0], feature.shape[1])
    features = sparse.lil_matrix(feature_new)

    objects = []
    objects.append(graph)
    graph = tuple(objects)

    num_nodes = feature.shape[0]
    network = nx.Graph()

    for num_node in range(num_nodes):
        network.add_node(num_node)
    for num_node in range(num_nodes):
        for num_edge in range(len(graph[0][num_node])):
            if graph[0][num_node][num_edge] <= num_nodes:
                network.add_edge(num_node, graph[0][num_node][num_edge])

    adj = nx.adjacency_matrix(network).todense()

    adj = sparse.csr_matrix(adj)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # Remove diagonal elements

    adj_orig.eliminate_zeros()
    # print(adj)
    # adj_train, train_edges, _, _, _, _ = mask_test_edges(adj)
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # build test set with 10% positive links
    # adj = adj_train
    adj_train = adj

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    features = sparse_to_tuple(features.tocoo())
    # print('features', features)

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    adj_norm = adj_norm.cuda()
    adj_norm = adj_norm.type(torch.float64)
    model = GCN(adj_norm, args.n_components, hidden_layer_dim)
    model = model.cuda()
    model.load_state_dict(torch.load("GCN.pt"))

    features = features.cuda()
    features = features.type(torch.float64)
    label_colors = np.random.randint(255, size=(hidden_layer_dim, 3))

    embedding = model(features, m, n)[0]

    embedding = embedding.permute(1, 2, 0).contiguous()

    embedding = embedding.view(-1, hidden_layer_dim)

    ignore, target = torch.max(embedding, 1)
    im_target = target.data.cpu().numpy()

    im_target_rgb = np.array([label_colors[c % hidden_layer_dim] for c in im_target])
    result_label = np.array([c for c in im_target])
    im_target_rgb = im_target_rgb.reshape(m, n, 3).astype(np.uint8)

    cv2.imwrite("segRef.png", im_target_rgb)
    np.savetxt("segRef.csv", result_label,
                       delimiter=',')

    return im_target

