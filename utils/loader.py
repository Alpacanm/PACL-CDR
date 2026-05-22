"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs

class dataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        if "dual" in opt["task"]:
            self.batch_size = batch_size
            self.opt = opt
            self.eval = evaluation

            # ************* source data *****************
            source_train_data = "./datasets/" +str(opt["task"]) + "/dataset/"+ filename + "/train.txt"
            source_valid_data = "./datasets/" +str(opt["task"]) + "/dataset/" +filename + "/valid.txt"
            source_test_data = "./datasets/" +str(opt["task"]) + "/dataset/"+ filename + "/test.txt"

            self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(source_train_data)
            if evaluation == -1:
                opt["source_user_num"] = max(self.source_user_set) + 1
                opt["source_item_num"] = max(self.source_item_set) + 1
            # ************* target data *****************
            filename = filename.split("_")
            filename = filename[1] + "_" + filename[0]
            target_train_data = "./datasets/" +str(opt["task"]) + "/dataset/"+ filename + "/train.txt"
            target_valid_data = "./datasets/" +str(opt["task"]) + "/dataset/"+ filename + "/valid.txt"
            target_test_data = "./datasets/" +str(opt["task"]) + "/dataset/"+ filename + "/test.txt"
            self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_user_set, self.target_item_set = self.read_train_data(
                target_train_data)
            if evaluation == -1:
                opt["target_user_num"] = max(self.target_user_set) + 1
                opt["target_item_num"] = max(self.target_item_set) + 1

            if evaluation == 1:
                self.test_data = self.read_test_data(source_test_data, self.source_item_set)
            elif evaluation == 2:
                self.test_data = self.read_test_data(target_test_data, self.target_item_set)


            if evaluation == 3:
                self.test_data = self.read_test_data(source_valid_data, self.source_item_set)
            elif evaluation == 4:
                self.test_data = self.read_test_data(target_valid_data, self.target_item_set)


            # assert opt["source_user_num"] == opt["target_user_num"]
            if evaluation < 0:
                data = self.preprocess()
            else :
                data = self.preprocess_for_predict()
            # shuffle for training
            if evaluation == -1:
                indices = list(range(len(data)))
                random.shuffle(indices)
                data = [data[i] for i in indices]
                if batch_size > len(data):
                    batch_size = len(data)
                    self.batch_size = batch_size
                if len(data)%batch_size != 0:
                    data += data[:batch_size]
                data = data[: (len(data)//batch_size) * batch_size]
            self.num_examples = len(data)

            # chunk into batches
            data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
            self.data = data
        # if "multi" in opt["task"]:
        #     self.batch_size = batch_size
        #     self.opt = opt
        #     self.eval = evaluation
        #
        #     # ************* data *****************
        #     # print(filename)
        #     # print("=========================")
        #     train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename + "/train.txt"
        #     valid_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename + "/valid.txt"
        #     test_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename + "/test.txt"
        #     self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
        #         train_data)
        #     if evaluation == -1:
        #         opt["m1_user_num"] = max(self.source_user_set) + 1
        #         opt["m1_item_num"] = max(self.source_item_set) + 1
        #         # print(train_data)
        #         # print("=====================")
        #     filename = filename.split("_")
        #     filename_2 = filename[0] + "_2"
        #     print(filename_2)
        #     print("=====================")
        #     train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_2 + "/train.txt"
        #     self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
        #         train_data)
        #     if evaluation == -1:
        #         opt["m2_user_num"] = max(self.source_user_set) + 1
        #         opt["m2_item_num"] = max(self.source_item_set) + 1
        #         print(train_data)
        #         print("=====================")
        #     filename = filename_2.split("_")
        #     filename_3 = filename[0] + "_3"
        #     train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_3 + "/train.txt"
        #     self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
        #         train_data)
        #     if evaluation == -1:
        #         opt["m3_user_num"] = max(self.source_user_set) + 1
        #         opt["m3_item_num"] = max(self.source_item_set) + 1
        #         print(train_data)
        #         print("=====================")
        #     filename = filename_3.split("_")
        #     filename_4 = filename[0] + "_4"
        #     train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_4 + "/train.txt"
        #     self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
        #         train_data)
        #     if evaluation == -1:
        #         opt["m4_user_num"] = max(self.source_user_set) + 1
        #         opt["m4_item_num"] = max(self.source_item_set) + 1
        #         print(train_data)
        #         print("=====================")
        #     filename = filename_4.split("_")
        #     filename_5 = filename[0] + "_5"
        #     train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_5 + "/train.txt"
        #     self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
        #         train_data)
        #     if evaluation == -1:
        #         opt["m5_user_num"] = max(self.source_user_set) + 1
        #         opt["m5_item_num"] = max(self.source_item_set) + 1
        #         print(train_data)
        #         print("=====================")
        #
        #     # assert opt["source_user_num"] == opt["target_user_num"]
        #     if evaluation < 0:
        #         data = self.preprocess()
        #     else:
        #         data = self.preprocess_for_predict()
        #     # shuffle for training
        #     if evaluation == -1:
        #         indices = list(range(len(data)))
        #         random.shuffle(indices)
        #         data = [data[i] for i in indices]
        #         if batch_size > len(data):
        #             batch_size = len(data)
        #             self.batch_size = batch_size
        #         if len(data) % batch_size != 0:
        #             data += data[:batch_size]
        #         data = data[: (len(data) // batch_size) * batch_size]
        #     self.num_examples = len(data)
        #
        #     # chunk into batches
        #     data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        #     self.data = data

    def read_train_data(self, train_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user_set = set()
            item_set = set()
            ma = {}
            ma_list = {}
            for line in infile:
                line=line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                train_data.append([user, item])
                # ma[user]：一个 集合，存储该用户交互过的唯一物品。
                # ma_list[user]：一个 列表，存储该用户交互过的所有物品，包括重复的物品。
                # user_set：一个 集合，存储所有出现过的用户。
                # item_set：一个 集合，存储所有出现过的物品。
                # train_data : 用户与项目的一个列表
                if user not in ma.keys():
                    ma[user] = set()
                    ma_list[user] = []
                ma[user].add(item)
                ma_list[user].append(item)
                user_set.add(user)
                item_set.add(item)
        return ma, ma_list, train_data, user_set, item_set

    def read_test_data(self, test_file, item_set):
        user_item_set = {}  # 初始化一个字典，用来存储每个用户的物品集合
        ma_list_ = {}
        self.MIN_USER = 10000000  # 初始化最小用户 ID 为一个非常大的数
        self.MAX_USER = 0  # 初始化最大用户 ID 为 0
        with codecs.open(test_file, "r", encoding="utf-8") as infile:  #  打开测试数据文件
            for line in infile:  # 遍历文件中的每一行
                line = line.strip().split("\t")  #  处理每一行数据，按制表符分隔
                user = int(line[0])  #  获取用户 ID
                item = int(line[1])  #  获取物品 ID
                if user not in user_item_set:  #  如果该用户还没有记录物品集合
                    user_item_set[user] = set()  #  初始化该用户的物品集合
                    ma_list_[user] = []
                ma_list_[user].append(item)
                user_item_set[user].add(item)  #  将物品添加到该用户的物品集合中
                self.MIN_USER = min(self.MIN_USER, user)  # 更新最小用户 ID
                self.MAX_USER = max(self.MAX_USER, user)  # 更新最大用户 ID

        # 重新读取测试数据文件以生成测试样本
        with codecs.open(test_file, "r", encoding="utf-8") as infile:  #  再次打开测试数据文件
            test_data = []  #  用来存储最终生成的测试数据
            item_list = sorted(list(item_set))  # 将所有物品从 item_set 转换为排序后的列表
            cnt = 0  #  统计没有找到测试物品的行数
            for line in infile:  #  遍历文件中的每一行
                line = line.strip().split("\t")  # 处理每一行数据，按制表符分隔
                user = int(line[0])  #  获取用户 ID
                item = int(line[1])  #获取物品 ID
                if item in item_set:  #  如果该物品在给定的物品集合中
                    ret = [item]  #  初始化测试样本，先将当前物品加入
                    for i in range(self.opt["test_sample_number"]):  # 为每个用户生成一定数量的负样本
                        while True:  # 不断循环直到生成一个合法的负样本
                            rand = item_list[random.randint(0, len(item_set) - 1)]  # 随机选择一个物品
                            if self.eval == 1:  # 如果是评估模式（eval == 1）
                                # if rand in user_item_set[user]:  # 如果随机物品是用户已经交互过的物品，跳过
                                # 只要不是该用户交互过的商品都算负样本
                                if rand in ma_list_[user]:
                                    continue
                            ret.append(rand)  #  将随机选择的物品添加到测试样本中
                            break  #  退出当前循环，继续选择下一个负样本
                    test_data.append([user, ret])  #将用户和生成的测试样本添加到测试数据列表中
                else:
                    cnt += 1  # 如果物品不在 item_set 中，增加计数器
            print("un test:", cnt)  # 打印没有在测试集中找到的物品数量
            print("test length:", len(test_data))  # 打印生成的测试数据集的长度
        return test_data  #  返回最终的测试数据集
    # 提取每对的第一个和第二个元素，将他们作为子列表添加到processed中

    # 创造特定的格式
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        if "dual" in self.opt["task"]:
            for d in self.source_train_data:
                d = [d[1], d[0]]
                processed.append(d + [-1]) # i u -1
            for d in self.target_train_data:
                processed.append([-1] + d) # -1 u i
            # for d in  self.over_train_data:
            #     processed.append([-2] + d)# -2 u i
        return processed
    def find_pos(self,ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1:
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else :
            source_user = []
            target_user = []
            # over_user = []
            # source_neg_tmp = []
            # target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []
            # over_pos_tmp = []
            for b in batch:
                if b[0] == -1: # -1 u i
                    target_user.append(b[1])
                    target_pos_tmp.append(b[2])
                    # target_neg_tmp.append(self.find_neg(self.target_ma_set, b[1], "target_item_num"))
                if b[2] == -1:
                    source_user.append(b[1])
                    source_pos_tmp.append(b[0])
                    # source_neg_tmp.append(self.find_neg(self.source_ma_set, b[1], "source_item_num"))
                # if b[0] == -2:
                #     over_user.append(b[1])
                #     over_pos_tmp.append(b[2])
            return (torch.LongTensor(source_user), torch.LongTensor(source_pos_tmp), torch.LongTensor(target_user), torch.LongTensor(target_pos_tmp))

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)