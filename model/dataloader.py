import pandas as pd
from collections import defaultdict
import numpy as np
import random
from random import shuffle, choice


class Data(object):
    def __init__(self, config, logger):
        self.dataset_name = config.dataset_name
        self.dataset_path = '/'.join((config.dataset_path, config.dataset_name))
        self.num_neg = config.bpr_num_neg
        self.num_users, self.num_items, self.train_U2I, self.training_data, self.test_U2I, self.pop_train_count, _, self.test_I2U, self.train_I2U, self.val_U2I, self.test_iid_U2I, self.val_sum, self.test_iid_sum = self.load_data()

        # 打印基本数据统计信息
        logger.info('num_users:{:d}   num_items:{:d}   density:{:.6f}%'.format(
            self.num_users, self.num_items, _ / self.num_items / self.num_users * 100))

        # 初始化 batch 生成器状态
        self.batch_generator = None

    def load_data(self):
        """
        加载数据集，解析训练集、测试集和验证集。
        """
        training_data = []
        num_items, num_users = 0, 0
        train_num, test_num = 0, 0
        train_interation = []
        train_U2I, test_U2I, test_I2U, train_I2U = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(
            list)

        # 加载训练集
        train_file = pd.read_table(self.dataset_path + '/train.txt', header=None)
        for l in range(len(train_file)):
            line = train_file.iloc[l, 0]
            text = line.split(" ")
            text = list(map(int, text))
            uid = text[0]
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

        # 加载测试集
        test_file = pd.read_table(self.dataset_path + '/test.txt', header=None)
        for l in range(len(test_file)):
            line = test_file.iloc[l, 0]
            text = line.split(" ")
            text = list(map(int, text))
            uid = text[0]
            num_users = max(num_users, uid)
            items = text[1:]
            test_num += len(items)
            num_items = max(num_items, max(items))
            test_U2I[uid].extend(items)
            for item in items:
                test_I2U[item].append(uid)

        # 加载验证集或其他额外数据（按数据集类型判断）
        if self.dataset_name in ['amazon-book.new', 'tencent.new']:
            val_U2I, test_iid_U2I = defaultdict(list), defaultdict(list)
            val_sum, test_iid_sum = 0, 0
            val_file = pd.read_table(self.dataset_path + '/valid.txt', header=None)
            test_iid_file = pd.read_table(self.dataset_path + '/test_id.txt', header=None)
            for l in range(len(val_file)):
                line = val_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(int, text))
                uid = text[0]
                num_users = max(num_users, uid)
                items = text[1:]
                val_sum += len(items)
                num_items = max(num_items, max(items))
                val_U2I[uid].extend(items)
            for l in range(len(test_iid_file)):
                line = test_iid_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(int, text))
                uid = text[0]
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
                text = list(map(int, text))
                uid = text[0]
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

        # 统计用户和物品数
        num_users += 1
        num_items += 1

        return num_users, num_items, train_U2I, training_data, test_U2I, [], train_num + test_num + val_sum + test_iid_sum, test_I2U, train_I2U, val_U2I, test_iid_U2I, val_sum, test_iid_sum

    def reset_batch_generator(self, batch_size):
        """
        重置 batch 生成器，每次从头开始。
        """
        self.batch_generator = self._batch_pairwise_generator(batch_size)

    def get_next_batch(self):
        """
        获取下一个 batch，如果生成器为空或者结束，则引发异常。
        """
        if self.batch_generator is None:
            raise RuntimeError("Batch generator is not initialized. Call reset_batch_generator() first.")
        try:
            return next(self.batch_generator)
        except StopIteration:
            self.batch_generator = None  # 重置生成器状态
            raise StopIteration("All batches have been processed.")

    def _batch_pairwise_generator(self, batch_size):
        """
        内部生成器函数，用于生成 pairwise 的训练数据。
        """
        training_data = self.training_data
        shuffle(training_data)
        batch_id = 0
        data_size = len(training_data)
        item_list = list(range(self.num_items))

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
            for i, user in enumerate(users):
                i_idx.append(items[i])
                u_idx.append(user)
                neg_item = choice(item_list)
                while neg_item in self.train_U2I[user]:
                    neg_item = choice(item_list)
                j_idx.append(neg_item)
            yield u_idx, i_idx, j_idx