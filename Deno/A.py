from torch import nn
import torch.nn.functional as F
import torch
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time

init = nn.init.xavier_uniform_


class Model(nn.Module):
    def __init__(self, opt, user, item):
        super(Model, self).__init__()
        self.opt = opt
        self.user = user
        self.item = item
        self.device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.uEmbeds = nn.Parameter(init(torch.empty(self.user, self.opt["hidden_dim_"])))
        self.iEmbeds = nn.Parameter(init(torch.empty(self.item, self.opt["hidden_dim_"])))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.opt["GNN"])])

    def forward_gcn(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        # embedsLst.to(self.user.device)A
        return mainEmbeds[:self.user], mainEmbeds[self.user:]

    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers :
            embeds = gcn(adj, embedsLst[-1]).to(self.device)
            if embeds is not None:
                embedsLst.append(embeds.to(self.device))
        mainEmbeds = sum(embedsLst).to(self.device)

        return mainEmbeds

    def forward_graphcl_(self, generator):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        count = 0
        for gcn in self.gcnLayers:
            with torch.no_grad():
                adj = generator.generate(x=embedsLst[-1], layer=count)
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    # def loss_graphcl(self, x1, x2, users, items):
    #     T = self.opt["temp"]
    #     user_embeddings1, item_embeddings1 = torch.split(x1, [self.user, self.item], dim=0)
    #     user_embeddings2, item_embeddings2 = torch.split(x2, [self.user, self.item], dim=0)
    #     user_embeddings1 = F.normalize(user_embeddings1, dim=1)
    #     item_embeddings1 = F.normalize(item_embeddings1, dim=1)
    #     user_embeddings2 = F.normalize(user_embeddings2, dim=1)
    #     item_embeddings2 = F.normalize(item_embeddings2, dim=1)
    #
    #     user_embs1 = F.embedding(users, user_embeddings1)
    #     item_embs1 = F.embedding(items, item_embeddings1)
    #     user_embs2 = F.embedding(users, user_embeddings2)
    #     item_embs2 = F.embedding(items, item_embeddings2)
    #     all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
    #     all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)
    #     all_embs1_abs = all_embs1.norm(dim=1)
    #     all_embs2_abs = all_embs2.norm(dim=1)
    #     # print(all_embs1.shape, all_embs2.shape)
    #
    #     sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
    #                                                                                 all_embs2_abs)
    #     sim_matrix = torch.exp(sim_matrix / T)
    #     pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #     loss = - torch.log(loss)
    #
    #     return loss
    def loss_graphcl(self, x1, x2, users, items):
        T = self.opt["temp"]

        # Step 1: 分割用户和物品嵌入
        user_embeddings1, item_embeddings1 = torch.split(x1, [self.user, self.item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [self.user, self.item], dim=0)

        # # 打印分割后嵌入的形状
        # print("user_embeddings1 shape:", user_embeddings1.shape)
        # print("item_embeddings1 shape:", item_embeddings1.shape)
        # print("user_embeddings2 shape:", user_embeddings2.shape)
        # print("item_embeddings2 shape:", item_embeddings2.shape)

        # Step 2: 归一化处理
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        # # 打印归一化后的范数
        # print("user_embeddings1 norm mean:", user_embeddings1.norm(dim=1).mean().item())
        # print("item_embeddings1 norm mean:", item_embeddings1.norm(dim=1).mean().item())
        # print("user_embeddings2 norm mean:", user_embeddings2.norm(dim=1).mean().item())
        # print("item_embeddings2 norm mean:", item_embeddings2.norm(dim=1).mean().item())

        # # Step 3: 检查索引范围并过滤非法值
        # print("users max:", users.max().item(), "users min:", users.min().item())
        # print("items max:", items.max().item(), "items min:", items.min().item())
        # print("item_embeddings1 size(0):", item_embeddings1.size(0))
        # print("user_embeddings1 size(0):", user_embeddings1.size(0))

        # 用户索引过滤
        user_mask = (users >= 0) & (users < user_embeddings1.size(0))  # 合法索引范围
        valid_users = users[user_mask]

        # 物品索引过滤
        item_mask = (items >= 0) & (items < item_embeddings1.size(0))  # 合法索引范围
        valid_items = items[item_mask]

        # 检查过滤后的索引是否为空
        if valid_users.numel() == 0 or valid_items.numel() == 0:
            raise ValueError("All users or items indices are invalid after filtering!")
        #
        # print("Filtered users shape:", valid_users.shape)
        # print("Filtered items shape:", valid_items.shape)

        # Step 4: 嵌入操作
        user_embs1 = F.embedding(valid_users, user_embeddings1)
        item_embs1 = F.embedding(valid_items, item_embeddings1)
        user_embs2 = F.embedding(valid_users, user_embeddings2)
        item_embs2 = F.embedding(valid_items, item_embeddings2)
        #
        # # 打印嵌入结果的形状
        # print("user_embs1 shape:", user_embs1.shape)
        # print("item_embs1 shape:", item_embs1.shape)
        # print("user_embs2 shape:", user_embs2.shape)
        # print("item_embs2 shape:", item_embs2.shape)

        # Step 5: 拼接用户和物品嵌入
        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        # # 打印拼接结果的形状
        # print("all_embs1 shape:", all_embs1.shape)
        # print("all_embs2 shape:", all_embs2.shape)

        # Step 6: 计算嵌入的范数
        all_embs1_abs = all_embs1.norm(dim=1) + 1e-8  # 避免除零
        all_embs2_abs = all_embs2.norm(dim=1) + 1e-8  # 避免除零
        # print("all_embs1_abs mean:", all_embs1_abs.mean().item())
        # print("all_embs2_abs mean:", all_embs2_abs.mean().item())

        # Step 7: 计算相似度矩阵
        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        # print("sim_matrix shape:", sim_matrix.shape)
        # print("sim_matrix sample:", sim_matrix[:5, :5])  # 打印前 5x5 的相似度矩阵

        # Step 8: 计算正样本相似度
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
        # print("pos_sim shape:", pos_sim.shape)
        # print("pos_sim mean:", pos_sim.mean().item())

        # Step 9: 计算损失
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss + 1e-8)  # 避免 log(0)

        # 检查损失是否包含非法值
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Warning: loss contains NaN or Inf values. Fixing...")
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)

        # print("loss shape:", loss.shape)
        # print("loss mean:", loss.mean().item())

        return loss


    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)
        return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        if flag:
            embeds = embeds.to(adj.device)
            return torch.spmm(adj, embeds)
        else:
            return torch.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class vgae_encoder(Model):
    def __init__(self, opt, user, item, device):
        # 正确的super调用，确保父类的初始化方法被调用
        super(vgae_encoder, self).__init__(opt,user,item)

        # 初始化类属性
        self.opt = opt
        self.user = user
        self.item = item
        self.sigmoid = nn.Sigmoid()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        hidden = self.opt['hidden_dim_']

        # 定义编码器
        self.encoder_mean = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )
        self.encoder_std = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.Softplus()
        )

    def forward(self, adj):
        # 进行图卷积或其他操作（假设是从adj中获取特征）
        x = self.forward_graphcl(adj)

        # 计算均值和标准差
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)

        # 确保噪声与x_mean在同一设备上
        gaussian_noise = torch.randn_like(x_mean).to(x_mean.device)
        x = gaussian_noise * x_std + x_mean

        return x, x_mean, x_std



class vgae_decoder(Model):
    def __init__(self, opt, user, item):
        super(vgae_decoder, self).__init__(opt,user,item)
        self.opt = opt
        self.user = user
        self.item = item
        self.devide = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.hidden = self.opt['hidden_dim_']
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(self.hidden, self.hidden), nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [self.user, self.item], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / self.opt["batch_size"]
        regLoss = calcRegLoss(encoder) * self.opt["reg"]

        beta = 0.1
        loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()

        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self, data, edge_index, adj):
        x, _, _ = self.encoder(data)

        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class DenoisingNet(nn.Module):
    def __init__(self, gcnLayers, features, opt,user,item):
        super(DenoisingNet, self).__init__()
        self.opt = opt
        self.features = features

        self.gcnLayers = gcnLayers

        self.user = user
        self.item = item

        self.edge_weights = []
        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])

        hidden = self.opt["hidden_dim_"]

        self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden, 1))
        self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden, 1))

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def get_attention(self, input1, input2, layer=0):
        nb_layer = None
        selflayer = None

        nb_layer = self.nblayers_0
        selflayer = self.selflayers_0


        input1 = nb_layer(input1)
        input2 = selflayer(input2)

        input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)

        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = self.opt["gamma"]
        zeta = self.opt["zeta"]

        if training:
            debug_var = 1e-7
            bias = 0.0
            np_random = np.random.uniform(low=debug_var, high=1.0 - debug_var,
                                          size=np.shape(log_alpha.cpu().detach().numpy()))
            random_noise = bias + torch.tensor(np_random)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def generate(self, x, layer=0):
        f1_features = x[self.row, :]
        f2_features = x[self.col, :]

        weight = self.get_attention(f1_features, f2_features, layer=0)

        mask = self.hard_concrete_sample(weight, training=False)

        mask = torch.squeeze(mask)
        adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape)

        ind = deepcopy(adj._indices())
        row = ind[0, :]
        col = ind[1, :]

        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = torch.mul(adj._values(), row_inv_sqrt)
        values = torch.mul(values, col_inv_sqrt)

        support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape)

        return support

    def l0_norm(self, log_alpha, beta):
        gamma = self.opt["gamma"]
        zeta = self.opt["zeta"]
        gamma = torch.tensor(gamma)
        zeta = torch.tensor(zeta)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma / zeta))

        return torch.mean(reg_per_weight)

    def set_fea_adj(self, nodes, adj):
        self.node_size = nodes
        self.adj_mat = adj

        ind = deepcopy(adj._indices())

        self.row = ind[0, :]
        self.col = ind[1, :]

    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.maskes = []

        x = self.features.detach()
        layer_index = 0
        embedsLst = [self.features.detach()]

        for layer in self.gcnLayers:
            xs = []
            f1_features = x[self.row, :]
            f2_features = x[self.col, :]
            #f1_features = torch.index_select(x, 0, self.row)
            #f2_features = torch.index_select(x, 0, self.col)

            weight = self.get_attention(f1_features, f2_features, layer=layer_index)
            mask = self.hard_concrete_sample(weight, temperature, training)

            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = torch.squeeze(mask)

            adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
            ind = deepcopy(adj._indices())
            row = ind[0, :]
            col = ind[1, :]

            rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = torch.mul(adj.values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)
            support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce().detach()

            nextx = layer(support, x, True)
            xs.append(nextx)
            x = xs[0]
            embedsLst.append(x)
            layer_index += 1
        return sum(embedsLst)

    def lossl0(self, temperature):
        l0_loss = torch.zeros([]).cuda()
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def forward(self, users, items, neg_items, temperature):
        self.freeze(self.gcnLayers)
        x = self.call(temperature, True)
        x_user, x_item = torch.split(x, [self.user, self.item], dim=0)
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / self.opt["batch_size"]
        regLoss = calcRegLoss(self) * self.opt["reg"]

        lossl0 = self.lossl0(temperature) * self.opt["lambda0"]
        return bprLoss + regLoss + lossl0









