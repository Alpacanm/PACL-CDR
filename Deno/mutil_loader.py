"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs

class mutil_dataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
    
        if "multi" in opt["task"]:
            self.batch_size = batch_size
            self.opt = opt
            self.eval = evaluation

            train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename + "/train.txt"
            valid_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename + "/valid.txt"
            test_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename + "/test.txt"
            self.source_ma_set, self.source_ma_list, self.m1_source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
                train_data)
            if evaluation == -1:
                opt["m1_user_num"] = max(self.source_user_set) + 1
                opt["m1_item_num"] = max(self.source_item_set) + 1

            filename = filename.split("_")
            filename_2 = filename[0] + "_2"
            train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_2 + "/train.txt"
            self.source_ma_set, self.source_ma_list, self.m2_source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
                train_data)
            if evaluation == -1:
                opt["m2_user_num"] = max(self.source_user_set) + 1
                opt["m2_item_num"] = max(self.source_item_set) + 1
            filename = filename_2.split("_")
            filename_3 = filename[0] + "_3"
            train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_3 + "/train.txt"
            self.source_ma_set, self.source_ma_list, self.m3_source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
                train_data)
            if evaluation == -1:
                opt["m3_user_num"] = max(self.source_user_set) + 1
                opt["m3_item_num"] = max(self.source_item_set) + 1
            filename = filename_3.split("_")
            filename_4 = filename[0] + "_4"
            train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_4 + "/train.txt"
            self.source_ma_set, self.source_ma_list, self.m4_source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
                train_data)
            if evaluation == -1:
                opt["m4_user_num"] = max(self.source_user_set) + 1
                opt["m4_item_num"] = max(self.source_item_set) + 1
            filename = filename_4.split("_")
            filename_5 = filename[0] + "_5"
            train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + filename_5 + "/train.txt"
            self.source_ma_set, self.source_ma_list, self.m5_source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(
                train_data)
            if evaluation == -1:
                opt["m5_user_num"] = max(self.source_user_set) + 1
                opt["m5_item_num"] = max(self.source_item_set) + 1
            if evaluation < 0:
                data = self.preprocess()
            else:
                data = self.preprocess_for_predict()
            # shuffle for training
            if evaluation == -1:
                indices = list(range(len(data)))
                random.shuffle(indices)
                data = [data[i] for i in indices]
                if batch_size > len(data):
                    batch_size = len(data)
                    self.batch_size = batch_size
                if len(data) % batch_size != 0:
                    data += data[:batch_size]
                data = data[: (len(data) // batch_size) * batch_size]
            self.num_examples = len(data)

            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            self.data = data

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
                if user not in ma.keys():
                    ma[user] = set()
                    ma_list[user] = []
                ma[user].add(item)
                ma_list[user].append(item)
                user_set.add(user)
                item_set.add(item)
        return ma, ma_list, train_data, user_set, item_set

    def read_test_data(self, test_file, item_set):
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
                            if self.eval == 1: 
                                if rand in ma_list_[user]:
                                    continue
                            ret.append(rand)  
                            break  
                    test_data.append([user, ret])  
                else:
                    cnt += 1  
        return test_data 
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        if "multi" in self.opt["task"]:
            for d in self.m1_source_train_data:
                processed.append(d + [-1]) # u i -1
            for d in self.m2_source_train_data:
                processed.append(d + [-2]) # u i -2
            for d in self.m3_source_train_data:
                processed.append(d + [-3])
            for d in self.m4_source_train_data:
                processed.append(d + [-4])
            for d in self.m5_source_train_data:
                processed.append(d + [-5])
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
            m1_user = []
            m1_item = []
            m2_user = []
            m2_item = []
            m3_user = []
            m3_item = []
            m4_user = []
            m4_item = []
            m5_user = []
            m5_item = []
            for b in batch:
                if b[2] == -1: # -1 u i
                    m1_user.append(b[0])
                    m1_item.append(b[1])
                if b[2] == -2:
                    m2_user.append(b[0])
                    m2_item.append(b[1])
                if b[2] == -3:
                    m3_user.append(b[0])
                    m3_item.append(b[1])
                if b[2] == -4:
                    m4_user.append(b[0])
                    m4_item.append(b[1])
                if b[2] == -5:
                    m5_user.append(b[0])
                    m5_item.append(b[1])
            return (torch.LongTensor(m1_user), torch.LongTensor(m1_item),
                    torch.LongTensor(m2_user), torch.LongTensor(m2_item),
                    torch.LongTensor(m3_user), torch.LongTensor(m3_item),
                    torch.LongTensor(m4_user), torch.LongTensor(m4_item),
                    torch.LongTensor(m5_user), torch.LongTensor(m5_item))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)