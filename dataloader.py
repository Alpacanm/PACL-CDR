import pandas as pd
from collections import defaultdict
import numpy as np
import torch
import random
import torch.nn.functional as F
from random import shuffle, choice


class Data(object):
    def __init__(self, opt, logger, domains):
        self.dataset_name = opt["domains"]
        self.dataset_path = '/'.join((opt["dataset_path"], domains))
        print(self.dataset_path)
        self.num_neg = opt["bpr_num_neg"]
        self.num_users, self.num_items, self.train_U2I, self.training_data, self.test_U2I, self.pop_train_count, _, self.test_I2U, self.train_I2U, self.val_U2I, self.test_iid_U2I, self.val_sum, self.test_iid_sum = self.load_data()
        logger.info('num_users:{:d}   num_items:{:d}   density:{:.6f}%'.format(
            self.num_users, self.num_items, _ / self.num_items / self.num_users * 100))

    def load_data(self):
        training_data = []
        num_items, num_users = 0, 0
        train_num, test_num = 0, 0
        train_interation = []
        train_U2I, test_U2I, test_I2U, train_I2U = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(
            list)
        train_file = pd.read_table(
            self.dataset_path + '/train_p.txt', header=None)
        test_file = pd.read_table(self.dataset_path + '/test_p.txt', header=None)

        # Read train data
        for l in range(len(train_file)):
            line = train_file.iloc[l, 0]
            text = line.split(" ")
            text = list(map(lambda x: int(x), text))
            uid = int(text[0])
            num_users = max(num_users, uid)
            items = text[1:]
            if len(items) == 0:
                continue
            train_interation.extend(items)
            num_items = max(num_items, max(items))
            train_num += len(items)
            train_U2I[uid].extend(items)
            for item in items:
                training_data.append([uid, item])
                train_I2U[item].append(uid)

        # Read test data
        for l in range(len(test_file)):
            line = test_file.iloc[l, 0]
            text = line.split(" ")
            text = list(map(lambda x: int(x), text))
            uid = int(text[0])
            num_users = max(num_users, uid)
            items = text[1:]
            test_num += len(items)
            num_items = max(num_items, max(items))
            test_U2I[uid].extend(items)
            for item in items:
                test_I2U[item].append(uid)

        # Handle validation and test data for specific datasets
        if self.dataset_name in ['amazon-book.new', 'tencent.new']:
            val_U2I, test_iid_U2I = defaultdict(list), defaultdict(list)
            val_sum, test_iid_sum = 0, 0
            val_file = pd.read_table(self.dataset_path + '/valid.txt', header=None)
            test_iid_file = pd.read_table(self.dataset_path + '/test_id.txt', header=None)
            for l in range(len(val_file)):
                line = val_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))
                uid = int(text[0])
                num_users = max(num_users, uid)
                items = text[1:]
                val_sum += len(items)
                num_items = max(num_items, max(items))
                val_U2I[uid].extend(items)
            for l in range(len(test_iid_file)):
                line = test_iid_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))
                uid = int(text[0])
                num_users = max(num_users, uid)
                items = text[1:]
                test_iid_sum += len(items)
                num_items = max(num_items, max(items))
                test_iid_U2I[uid].extend(items)
        elif self.dataset_name in ['meituan', 'douban.new', 'coat', 'yahoo.new']:
            val_U2I = defaultdict(list)
            val_sum = 0
            val_file = pd.read_table(self.dataset_path + '/valid.txt', header=None)
            for l in range(len(val_file)):
                line = val_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))
                uid = int(text[0])
                num_users = max(num_users, uid)
                items = text[1:]
                val_sum += len(items)
                num_items = max(num_items, max(items))
                val_U2I[uid].extend(items)
            test_iid_U2I = test_U2I
            test_iid_sum = 0
        else:
            val_U2I = test_U2I
            test_iid_U2I = test_U2I
            val_sum, test_iid_sum = 0, 0

        # Update counts
        num_users += 1
        num_items += 1

        # Negative sampling for training data
        training_data = [
            val for val in training_data for i in range(self.num_neg)
        ]

        # Compute item popularity
        ps = pd.Series(train_interation)
        vc = ps.value_counts(sort=False)
        vc.sort_index(inplace=True)
        pop_train = []
        if num_items == len(np.unique(np.array(train_interation))):
            for item in range(num_items):
                pop_train.append(vc[item])
        else:
            for item in range(num_items):
                if item not in list(vc.index):
                    pop_train.append(0)
                else:
                    pop_train.append(vc[item])

        return num_users, num_items, train_U2I, training_data, test_U2I, pop_train, train_num + test_num + val_sum + test_iid_sum, test_I2U, train_I2U, val_U2I, test_iid_U2I, val_sum, test_iid_sum


class Graph(object):
    def __init__(self, num_users, num_items, train_U2I, gama):
        self.num_users = num_users
        self.num_items = num_items
        self.train_U2I = train_U2I
        self.gama = gama

    def to_edge(self):
        train_U, train_I = [], []
        for u, items in self.train_U2I.items():
            train_U.extend([u] * len(items))
            train_I.extend(items)

        train_U = np.array(train_U)
        train_I = np.array(train_I)

        row = np.concatenate([train_U, train_I + self.num_users])
        col = np.concatenate([train_I + self.num_users, train_U])

        edge_weight = np.ones_like(row).tolist()
        edge_index = np.stack([row, col]).tolist()

        return train_U, train_I, edge_index, edge_weight


class LaplaceGraph(Graph):
    def __init__(self, num_users, num_items, train_U2I, gama=0.5):
        super().__init__(num_users, num_items, train_U2I, gama)

    def generate(self):
        graph_u, graph_i, edge_index, edge_weight = self.to_edge()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        edge_index, edge_weight = self.norm(edge_index, edge_weight)
        return self.mat(edge_index, edge_weight)

    def add_self_loop(self, edge_index, edge_weight):
        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)
        return edge_index, edge_weight

    def norm(self, edge_index, edge_weight):
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32).scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-1 * self.gama)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    @property
    def num_nodes(self):
        return self.num_users + self.num_items

    def mat(self, edge_index, edge_weight):
        return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([self.num_nodes, self.num_nodes]))


def next_batch_pairwise(data, batch_size):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(range(data.num_items))
        for i, user in enumerate(users):
            i_idx.append(items[i])
            u_idx.append(user)
            neg_item = choice(item_list)
            while neg_item in data.train_U2I[user]:
                neg_item = choice(item_list)
            j_idx.append(neg_item)
        yield u_idx, i_idx, j_idx


def user_items_2_group_pop(data):
    G1, G2 = [], []
    for u in data.train_U2I.keys():
        items = data.train_U2I[u]
        items_sorted = list(np.array(items)[np.argsort(np.array(data.pop_train_count)[items])])
        if len(items) % 2 != 0:
            items_sorted = np.delete(items_sorted, random.sample(range(len(items_sorted)), 1)[0])
        num = int(len(items_sorted) / 2)
        G1.extend(items_sorted[0:num])
        G2.extend(items_sorted[num:])
    return np.array(G1), np.array(G2)