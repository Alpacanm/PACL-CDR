import argparse
import os.path
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import scipy as scipy

import os
from model import metrics, dataloader, utils, mini_batch_test
import ast


class PAAC(torch.nn.Module):
    def __init__(self, opt, data):
        super(PAAC, self).__init__()

        # model
        self.emb_size = opt["emb_size"]
        self.decay = opt["decay"]
        self.layers = opt["layers"]
        self.device = torch.device(opt["device"])
        self.eps = opt["eps_PAAC"]
        self.cl_rate = opt["cl_rate"]
        self.temperature = opt["temperature"]
        self.pop_train = data.pop_train_count
        self.lambda2 = opt["lambda"]
        self.gamma = opt["gama"]


        # data
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.adj = data.norm_adj.to(self.device)

        # embedding
        user_emb_weight = torch.nn.init.normal_(torch.empty(
            self.num_users, self.emb_size), mean=0, std=0.1)
        item_emb_weight = torch.nn.init.normal_(torch.empty(
            self.num_items, self.emb_size), mean=0, std=0.1)
        self.user_embeddings = torch.nn.Embedding(
            self.num_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = torch.nn.Embedding(
            self.num_items, self.emb_size, _weight=item_emb_weight)

    #
    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

        # all_emb = [ego_embeddings]

        all_emb = []

        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise,
                                                                                           dim=1) * self.eps
            all_emb = all_emb + [ego_embeddings]
        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(
            all_emb, [self.num_users, self.num_items])
        return user_emb, item_emb

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        pos_score = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_emb).sum(dim=1)
        bpr_loss = - \
            torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).mean()

        l2_loss = torch.norm(user_emb, p=2) + torch.norm(pos_emb, p=2)
        l2_loss = self.decay * l2_loss

        return bpr_loss, l2_loss

    def cl_loss(self, u_idx, i_idx, j_idx):
        # batch里采样
        u_idx = torch.tensor(u_idx)
        bacth_pop, batch_unpop = utils.split_bacth_items(i_idx, self.pop_train)
        batch_users = torch.unique(u_idx).type(torch.long).to(self.device)
        bacth_pop = torch.tensor(bacth_pop)
        bacth_pop = torch.unique(bacth_pop).type(torch.long).to(self.device)
        batch_unpop = torch.tensor(batch_unpop)
        batch_unpop = torch.unique(batch_unpop).type(torch.long).to(self.device)
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_cl_loss = metrics.InfoNCE(
            user_view_1[batch_users], user_view_2[batch_users], self.temperature) * self.cl_rate
        item_cl_pop = self.gamma * metrics.InfoNCE_i(item_view_1[bacth_pop], item_view_2[bacth_pop],
                                                     item_view_2[batch_unpop], self.temperature, self.lambda2)
        # save_path = "./embeddings"
        # os.makedirs(save_path, exist_ok=True)
        # # 保存 view1 的嵌入（文件名由外部调用控制）
        # # 临时文件名，实际文件名由调用时指定
        # torch.save(item_view_1[bacth_pop], f"{save_path}/temp_embeddings_view1_pop.pt")
        #
        # # 输出 view1 的嵌入规模
        # print(f"物品嵌入规模 (viewpop1): {item_view_1[bacth_pop].shape}")

        item_cl_unpop = (1 - self.gamma) * metrics.InfoNCE_i(item_view_1[batch_unpop], item_view_2[batch_unpop],
                                                             item_view_2[bacth_pop], self.temperature, self.lambda2)
        # # save_path = "./embeddings"
        # # os.makedirs(save_path, exist_ok=True)
        # # 保存 view1 的嵌入（文件名由外部调用控制）
        # # 临时文件名，实际文件名由调用时指定
        # torch.save(item_view_1[batch_unpop], f"{save_path}/temp_embeddings_view1_unpop.pt")
        #
        # # 输出 view1 的嵌入规模
        # print(f"物品嵌入规模 (viewunpop1): {item_view_1[batch_unpop].shape}")
        item_cl_loss = (item_cl_pop + item_cl_unpop) * self.cl_rate
        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, user_cl_loss, item_cl_loss

    def batch_loss(self, u_idx, i_idx, j_idx):

        user_embedding, item_embedding = self.forward(perturbed=False)
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        neg_emb = item_embedding[j_idx]
        bpr_loss, l2_loss = self.bpr_loss(user_emb, pos_emb, neg_emb)
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx, j_idx)
        batch_loss = bpr_loss + l2_loss + cl_loss
        return batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss

