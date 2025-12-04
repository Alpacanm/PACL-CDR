import json
import random
import torch
import numpy as np
import codecs

class dataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    Modified to sample 4096 source and 4096 target samples per batch.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        if "dual" in opt["task"]:
            self.batch_size = batch_size / 2  # 每个域的批次大小，例如 4096
            self.opt = opt
            self.eval = evaluation

            # ************* Source Data *****************
            if opt["inject"] == 0:
                source_train_data = f"./datasets/{opt['task']}/dataset/{filename}/train.txt"
            elif opt["inject"] == 0.05:
                source_train_data = f"./datasets/{opt['task']}/dataset/{filename}/train_noisy_0.05.txt"
            else:
                source_train_data = f"./datasets/{opt['task']}/dataset/{filename}/train_noisy_0.1.txt"

            source_valid_data = f"./datasets/{opt['task']}/dataset/{filename}/valid.txt"
            source_test_data = f"./datasets/{opt['task']}/dataset/{filename}/test.txt"

            self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(source_train_data)
            if evaluation == -1:
                opt["source_user_num"] = max(self.source_user_set) + 1
                opt["source_item_num"] = max(self.source_item_set) + 1

            # ************* Target Data *****************
            filename = filename.split("_")
            filename = filename[1] + "_" + filename[0]
            if opt["inject"] == 0:
                target_train_data = f"./datasets/{opt['task']}/dataset/{filename}/train.txt"
            elif opt["inject"] == 0.05:
                target_train_data = f"./datasets/{opt['task']}/dataset/{filename}/train_noisy_0.05.txt"
            else:
                target_train_data = f"./datasets/{opt['task']}/dataset/{filename}/train_noisy_0.1.txt"
            target_valid_data = f"./datasets/{opt['task']}/dataset/{filename}/valid.txt"
            target_test_data = f"./datasets/{opt['task']}/dataset/{filename}/test.txt"
            self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_user_set, self.target_item_set = self.read_train_data(target_train_data)
            if evaluation == -1:
                opt["target_user_num"] = max(self.target_user_set) + 1
                opt["target_item_num"] = max(self.target_item_set) + 1

            if evaluation == 1:
                self.test_data = self.read_test_data(source_test_data, self.source_item_set)
            elif evaluation == 2:
                self.test_data = self.read_test_data(target_test_data, self.target_item_set)
            elif evaluation == 3:
                self.test_data = self.read_test_data(source_valid_data, self.source_item_set)
            elif evaluation == 4:
                self.test_data = self.read_test_data(target_valid_data, self.target_item_set)

            if evaluation < 0:
                self.source_data, self.target_data = self.preprocess()  # 分开存储源域和目标域数据
            else:
                self.data = self.preprocess_for_predict()  # 测试数据保持不变

            # Shuffle and prepare batches for training
            if evaluation == -1:
                # 源域数据打乱并分批
                source_indices = list(range(len(self.source_data)))
                random.shuffle(source_indices)
                self.source_data = [self.source_data[i] for i in source_indices]
                if len(self.source_data) < batch_size:
                    raise ValueError("Source data size is less than batch_size!")
                self.source_batches = [self.source_data[i:i + batch_size] for i in range(0, len(self.source_data) - batch_size + 1, batch_size)]

                # 目标域数据打乱并分批
                target_indices = list(range(len(self.target_data)))
                random.shuffle(target_indices)
                self.target_data = [self.target_data[i] for i in target_indices]
                if len(self.target_data) < batch_size:
                    raise ValueError("Target data size is less than batch_size!")
                self.target_batches = [self.target_data[i:i + batch_size] for i in range(0, len(self.target_data) - batch_size + 1, batch_size)]

                # 确保批次数一致，取最小值
                self.num_batches = min(len(self.source_batches), len(self.target_batches))
                self.source_batches = self.source_batches[:self.num_batches]
                self.target_batches = self.target_batches[:self.num_batches]
            else:
                self.num_batches = len(self.data)

            self.index = 0

    def read_train_data(self, train_file):
        """读取训练数据"""
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user_set = set()
            item_set = set()
            ma = {}
            ma_list = {}
            for line in infile:
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                train_data.append([user, item])
                if user not in ma:
                    ma[user] = set()
                    ma_list[user] = []
                ma[user].add(item)
                ma_list[user].append(item)
                user_set.add(user)
                item_set.add(item)
        return ma, ma_list, train_data, user_set, item_set

    def read_test_data(self, test_file, item_set):
        """读取测试数据"""
        user_item_set = {}
        ma_list_ = {}
        self.MIN_USER = 10000000
        self.MAX_USER = 0
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                if user not in user_item_set:
                    user_item_set[user] = set()
                    ma_list_[user] = []
                ma_list_[user].append(item)
                user_item_set[user].add(item)
                self.MIN_USER = min(self.MIN_USER, user)
                self.MAX_USER = max(self.MAX_USER, user)

        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            item_list = sorted(list(item_set))
            cnt = 0
            for line in infile:
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                if item in item_set:
                    ret = [item]
                    for i in range(self.opt["test_sample_number"]):
                        while True:
                            rand = item_list[random.randint(0, len(item_set) - 1)]
                            if self.eval == 1 and rand in ma_list_[user]:
                                continue
                            ret.append(rand)
                            break
                    test_data.append([user, ret])
                else:
                    cnt += 1
            print("un test:", cnt)
            print("test length:", len(test_data))
        return test_data

    def preprocess(self):
        """预处理数据，分别生成源域和目标域数据"""
        source_processed = []
        target_processed = []
        if "dual" in self.opt["task"]:
            for d in self.source_train_data:
                d = [d[1], d[0]]
                source_processed.append(d + [-1])  # i u -1
            for d in self.target_train_data:
                target_processed.append([-1] + d)  # -1 u i
        return source_processed, target_processed

    def preprocess_for_predict(self):
        """为预测预处理数据"""
        return self.test_data

    def find_pos(self, ma_list, user):
        """查找正样本"""
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        """查找负样本"""
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self):
        """返回批次数量"""
        return self.num_batches if self.eval == -1 else len(self.data)

    def __getitem__(self, key):
        """根据索引获取一个批次，每个域固定 batch_size 个样本"""
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= self.num_batches:
            raise IndexError

        if self.eval != -1:
            batch = self.data[key]
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))
        else:
            source_batch = self.source_batches[key]
            target_batch = self.target_batches[key]

            source_user = []
            target_user = []
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []

            # 处理源域数据
            for b in source_batch:
                user = b[1]
                item = b[0]
                source_user.append(user)
                source_pos_tmp.append(item)
                source_neg_tmp.append(self.find_neg(self.source_ma_set, user, "source_item_num"))

            # 处理目标域数据
            for b in target_batch:
                user = b[1]
                item = b[2]
                target_user.append(user)
                target_pos_tmp.append(item)
                target_neg_tmp.append(self.find_neg(self.target_ma_set, user, "target_item_num"))

            # # 每个列表长度应等于 batch_size
            # assert len(source_user) == self.batch_size, f"Source batch size mismatch: {len(source_user)} vs {self.batch_size}"
            # assert len(target_user) == self.batch_size, f"Target batch size mismatch: {len(target_user)} vs {self.batch_size}"

            return (
                torch.LongTensor(source_user),
                torch.LongTensor(source_pos_tmp),
                torch.LongTensor(source_neg_tmp),
                torch.LongTensor(target_user),
                torch.LongTensor(target_pos_tmp),
                torch.LongTensor(target_neg_tmp),
            )

    def __iter__(self):
        """初始化迭代器"""
        self.index = 0
        return self

    def __next__(self):
        """获取下一个批次，每个域固定 batch_size 个样本"""
        if self.index >= self.num_batches:
            raise StopIteration

        if self.eval != -1:
            batch = self.data[self.index]
            self.index += 1
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))
        else:
            source_batch = self.source_batches[self.index]
            target_batch = self.target_batches[self.index]
            self.index += 1

            source_user = []
            target_user = []
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []

            # 处理源域数据
            for b in source_batch:
                user = b[1]
                item = b[0]
                source_user.append(user)
                source_pos_tmp.append(item)
                source_neg_tmp.append(self.find_neg(self.source_ma_set, user, "source_item_num"))

            # 处理目标域数据
            for b in target_batch:
                user = b[1]
                item = b[2]
                target_user.append(user)
                target_pos_tmp.append(item)
                target_neg_tmp.append(self.find_neg(self.target_ma_set, user, "target_item_num"))

            # 每个列表长度应等于 batch_size
            # assert len(source_user) == self.batch_size, f"Source batch size mismatch: {len(source_user)} vs {self.batch_size}"
            # assert len(target_user) == self.batch_size, f"Target batch size mismatch: {len(target_user)} vs {self.batch_size}"

            return (
                torch.LongTensor(source_user),
                torch.LongTensor(source_pos_tmp),
                torch.LongTensor(source_neg_tmp),
                torch.LongTensor(target_user),
                torch.LongTensor(target_pos_tmp),
                torch.LongTensor(target_neg_tmp),
            )