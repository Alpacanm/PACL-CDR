import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker_(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.user = set()
        self.item = set()
        data=[]
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip("\t") 
                if not line:  
                    continue
                line = line.split("\t")  
                if len(line) == 2: 
                    user_id = int(line[0]) 
                    item_id = int(line[1])
                if len(line) == 3:
                    user_id = int(line[0])
                    item_id = int(line[1])
                data.append([user_id, item_id]) 
                self.user.add(user_id)
                self.item.add(item_id)

        opt["number_user"] = max(self.user) + 1
        opt["number_item"] = max(self.item) + 1



        
        self.raw_data = data

        self.UV,self.VU, self.adj = self.preprocess(data, opt)


    def preprocess(self,data,opt):
        # 用户到物品的边
        UV_edges = []
        # 物品到用户的边
        VU_edges = []
        # 包含所有边的列表
        all_edges = []
        # 存储实际的用户-物品图
        real_adj = {}
        # 初始化存储用户和物品对应关系的字典
        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            # UV_edges 存储用户到物品的边（edge[0]为用户，edge[1]为物品）
            UV_edges.append([edge[0], edge[1]])

            # 判断当前用户是否已在 user_real_dict 字典中
            if edge[0] not in user_real_dict.keys():
                # 如果当前用户不在字典中，则为该用户创建一个空集合
                user_real_dict[edge[0]] = set()
            # 向该用户的集合中添加物品（表示用户与物品之间的交互）
            user_real_dict[edge[0]].add(edge[1])

            # VU_edges 存储物品到用户的边（反向）
            VU_edges.append([edge[1], edge[0]])

            # 判断当前物品是否已在 item_real_dict 字典中
            if edge[1] not in item_real_dict.keys():
                # 如果当前物品不在字典中，则为该物品创建一个空集合
                item_real_dict[edge[1]] = set()
            # 向该物品的集合中添加用户（表示物品与用户之间的交互）
            item_real_dict[edge[1]].add(edge[0])

            # 添加所有边到 all_edges，用于构建全图邻接矩阵
            all_edges.append([edge[0], edge[1] + opt["number_user"]])  # 偏移量为用户数量
            all_edges.append([edge[1] + opt["number_user"], edge[0]])  # 反向边也添加

            # 判断当前用户是否已在 real_adj 字典中
            if edge[0] not in real_adj:
                # 如果当前用户不在字典中，则为该用户创建一个空字典
                real_adj[edge[0]] = {}
            # 将物品与该用户的关系添加到字典中（边的权重为 1）
            real_adj[edge[0]][edge[1]] = 1
        # 将 UV_edges, VU_edges 和 all_edges 转换为 NumPy 数组，方便后续操作
        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        # 创建用户-物品的邻接矩阵
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item"]),
                               dtype=np.float32)
        # 创建物品-用户的邻接矩阵
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item"], opt["number_user"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),
                                shape=(opt["number_item"]+opt["number_user"], opt["number_item"]+opt["number_user"]),
                                dtype=np.float32)
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        # print("real graph loaded!")
        return UV_adj, VU_adj, all_adj

    def preprocess_batch(self, source_user_item_pairs, opt):
        # 包含所有边的列表
        all_edges = []

        # 遍历 source_user_item_pairs，构建联合边
        for user, item in source_user_item_pairs:
            # 用户到物品的边（用户索引不变，物品索引偏移 opt["number_user"]）
            all_edges.append([user, item + opt["number_user"]])
            # 物品到用户的边（物品索引偏移 opt["number_user"]，用户索引不变）
            all_edges.append([item + opt["number_user"], user])

        # 将 all_edges 转换为 NumPy 数组
        all_edges = np.array(all_edges)

        # 创建联合邻接矩阵
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),
                                shape=(
                                opt["number_user"] + opt["number_item"], opt["number_user"] + opt["number_item"]),
                                dtype=np.float32)

        # 归一化
        all_adj = normalize(all_adj)

        # 转换为稀疏张量
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)


        return all_adj

