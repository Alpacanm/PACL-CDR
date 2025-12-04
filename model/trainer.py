from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.UniCDR import UniCDR
from utils import torch_utils
import numpy as np
import pdb
from tqdm import tqdm
import math
from model.model import DCCF
from utils.load_data import Data
from model.VBGE import VBGE
import  dataloader, utils
import datetime
from itertools import cycle
import os
from model.PAAC import PAAC
from random import shuffle, choice
from utils.Utils import innerProduct, pairPredict, calcRegLoss
from Deno.A import Model, vgae_encoder, vgae_decoder, DenoisingNet, vgae
class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch=None):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list

class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "UniCDR":
            self.model = UniCDR(opt).to(opt["device"])
            self.model_x = Model(opt, self.opt["source_user_num"], self.opt["source_item_num"])
            self.model_y = Model(opt, self.opt["target_user_num"], self.opt["target_item_num"])

            # ************* source data *****************
            source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0]
            source_data_generator = Data(opt, source_train_data)
            # source_n_samples = source_data_generator.uniform_sample()
            # source_n_batch = int(np.ceil(source_n_samples / opt["batch_size"]))
            config_x = dict()
            config_x['n_users'] = source_data_generator.n_users
            config_x['n_items'] = source_data_generator.n_items

            """
            *********************************************************
            Generate the adj matrix
            """
            x_plain_adj = source_data_generator.get_adj_mat()
            all_h_list, all_t_list, all_v_list = load_adjacency_list_data(x_plain_adj)
            config_x['plain_adj'] = x_plain_adj
            config_x['all_h_list'] = all_h_list
            config_x['all_t_list'] = all_t_list
            print("源域的数据开始读取")

            self.model_DCCF_s = DCCF(config_x, opt).to(opt["device"])
            # ************* target data *****************
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][1]
            target_data_generator = Data(opt, target_train_data)
            # target_n_samples = target_data_generator.uniform_sample()
            # target_n_batch = int(np.ceil(target_n_samples / opt["batch_size"]))
            config_y = dict()
            config_y['n_users'] = target_data_generator.n_users
            config_y['n_items'] = target_data_generator.n_items

            """
            *********************************************************
            Generate the adj matrix
            """
            plain_adj = target_data_generator.get_adj_mat()
            all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
            config_y['plain_adj'] = plain_adj
            config_y['all_h_list'] = all_h_list
            config_y['all_t_list'] = all_t_list
            self.model_DCCF_t = DCCF(config_y, opt).to(opt["device"])
            print("目标域的数据开始读取")

            self.source_GNN = VBGE(opt).to(opt["device"])
            self.target_GNN = VBGE(opt).to(opt["device"])
        else :
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'], opt["weight_decay"])
        self.epoch_rec_loss = []
    def generator_generate(self, generator, adj):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj_1 = deepcopy(adj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(adj, idxs, adj_1)

        return view
    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            item = inputs[1]
            context_item = inputs[2]
            context_score = inputs[3]
            global_item = inputs[4]
            global_score = inputs[5]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            item = inputs[1]
            context_item = inputs[2]
            context_score = inputs[3]
            global_item = inputs[4]
            global_score = inputs[5]
        return user, item, context_item, context_score, global_item, global_score

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            pos_item = inputs[1]
            neg_item = inputs[2]
            context_item = inputs[3]
            context_score = inputs[4]
            global_item = inputs[5]
            global_score = inputs[6]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            pos_item = inputs[1]
            neg_item = inputs[2]
            context_item = inputs[3]
            context_score = inputs[4]
            global_item = inputs[5]
            global_score = inputs[6]
        return user, pos_item, neg_item, context_item, context_score, global_item, global_score

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def test(self, model):
        user_embedding, item_embedding = model.forward()
        return user_embedding.detach().cpu().numpy(), item_embedding.detach().cpu().numpy()

    def next_batch_pairwise(self, data, batch_size):
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


    def reconstruction_loss_x(self, opt,domain_id,users, item, n_item,adj, temperature):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # s_user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        self.model_x = self.model_x.to(self.device)
        # user = s_user.to(self.device)
        users = users.to(self.device)
        item = item.to(self.device)
        n_item = n_item.to(self.device)
        encoder_x = vgae_encoder(self.opt, self.opt["source_user_num"], self.opt["source_item_num"], self.device)
        decoder_x = vgae_decoder(self.opt, self.opt["source_user_num"], self.opt["source_item_num"])
        self.encoder_x = encoder_x.to(self.device)
        self.decoder_x = decoder_x.to(self.device)
        # 初始化生成器并移动到设
        self.generator_1 = vgae(self.encoder_x, self.decoder_x).to(self.device)
        self.x_generator_2 = DenoisingNet(self.model_x.getGCN(), self.model_x.getEmbeds(), self.opt,
                                          self.opt["source_user_num"], self.opt["source_item_num"]).to(self.device)
        self.x_generator_2.set_fea_adj(self.opt["source_user_num"] + self.opt["source_item_num"],
                                       deepcopy(adj).to(self.device))
        data1 = self.generator_generate(self.generator_1, adj.to(self.device))
        out_x = self.model_x.forward_graphcl(data1)
        x_out = self.model_x.forward_graphcl_(self.x_generator_2)
        assert out_x.requires_grad, "out_x does not require gradients"
        assert x_out.requires_grad, "x_out does not require gradients"
        # 损失计算 ib

        loss_x = (self.model_x.loss_graphcl(out_x, x_out, users, item))
        loss_x = (loss_x ).mean() * self.opt["ssl_reg"]
        x_out1 = self.model_x.forward_graphcl(data1)
        x_out2 = self.model_x.forward_graphcl_(self.x_generator_2)
        x_loss_ib = (self.model_x.loss_graphcl(out_x, out_x.detach(), users, item)
                     + self.model_x.loss_graphcl(x_out, x_out.detach(), users, item))
        # loss_x += x_loss_ib.mean()
        loss_x += x_loss_ib.mean() * self.opt["ib_reg"]
        #
        # x_usrEmbeds, x_itmEmbeds = self.model_x.forward_gcn(deepcopy(adj).to(self.device))
        # x_ancEmbeds = x_usrEmbeds[users]
        # x_posEmbeds = x_itmEmbeds[item]
        # x_negEmbeds = x_itmEmbeds[n_item]
        # x_scoreDiff = pairPredict(x_ancEmbeds, x_posEmbeds, x_negEmbeds)
        # x_bprLoss = -(x_scoreDiff).sigmoid().log().sum() / self.opt["data_batch_size"]
        # x_regLoss = calcRegLoss(self.model_x) * self.opt["reg"]
        #
        #
        # loss_x += x_bprLoss + x_regLoss
        # x_loss_1 = self.generator_1(deepcopy(adj).to(self.device), users, item, n_item)
        # x_loss_2 = self.x_generator_2(users, item, n_item, temperature)
        # loss_x += x_loss_1 + x_loss_2

        return loss_x

    def reconstruction_loss_y(self, opt, domain_id,user,item, n_item, adj, temperature ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # s_user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        self.model_y = self.model_y.to(self.device)
        # S_user, S_pos_item, S_n_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        # user = s_user.to(self.device)
        user = user.to(self.device)
        item = item.to(self.device)
        n_item = n_item.to(self.device)
        encoder_y = vgae_encoder(self.opt, self.opt["target_user_num"], self.opt["target_item_num"], self.device)
        decoder_y = vgae_decoder(self.opt, self.opt["target_user_num"], self.opt["target_item_num"])
        self.encoder_y = encoder_y.to(self.device)
        self.decoder_y = decoder_y.to(self.device)
        self.generator_2 = vgae(self.encoder_y, self.decoder_y).to(self.device)
        self.y_generator_2 = DenoisingNet(self.model_y.getGCN(), self.model_y.getEmbeds(), self.opt,
                                          self.opt["target_user_num"], self.opt["target_item_num"]).to(self.device)
        self.y_generator_2.set_fea_adj(self.opt["target_user_num"] + self.opt["target_item_num"],
                                       deepcopy(adj).to(self.device))
        data2 = self.generator_generate(self.generator_2, adj.to(self.device))
        out_y = self.model_y.forward_graphcl(data2)
        y_out = self.model_y.forward_graphcl_(self.y_generator_2)
        assert out_y.requires_grad, "out_y does not require gradients"
        assert y_out.requires_grad, "y_out does not require gradients"
        loss_y = (self.model_y.loss_graphcl(out_y, y_out, user, item) )
        loss_y = (loss_y ).mean() * self.opt["ssl_reg"]
        y_out1 = self.model_y.forward_graphcl(data2)

        y_out2 = self.model_y.forward_graphcl_(self.y_generator_2)
        y_loss_ib = self.model_y.loss_graphcl(y_out1, out_y.detach(), user,
                                              item) + self.model_y.loss_graphcl(y_out2, y_out.detach(),
                                                                                      user, item)
        # loss_y += y_loss_ib.mean()
        loss_y += y_loss_ib.mean()*self.opt["ib_reg"]
        #
        # y_usrEmbeds, y_itmEmbeds = self.model_y.forward_gcn(deepcopy(adj).to(self.device))
        # y_ancEmbeds = y_usrEmbeds[user]
        # y_posEmbeds = y_itmEmbeds[item]
        # y_negEmbeds = y_itmEmbeds[n_item]
        # y_scoreDiff = pairPredict(y_ancEmbeds, y_posEmbeds, y_negEmbeds)
        # y_bprLoss = -(y_scoreDiff).sigmoid().log().sum() / self.opt["data_batch_size"]
        # y_regLoss = calcRegLoss(self.model_y) * self.opt["reg"]
        # loss_y += y_bprLoss + y_regLoss
        # y_loss_1 = self.generator_2(deepcopy(adj).to(self.device), user, item, n_item)
        # y_loss_2 = self.y_generator_2(user,item,n_item, temperature)
        # loss_y += y_loss_1 + y_loss_2
        return loss_y

    def train(self, config, data, model, early_stopping, logger, train_step=1):
        model.train()  # Set the model to training mode
        start = datetime.datetime.now()

        # Initialize training results
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
            'cl_loss': 0.0,
            'batch_loss': 0.0,
            'align_loss': 0.0,
        }

        # Initialize batch generator
        batch_loader = cycle(self.next_batch_pairwise(data, self.opt["batch_size"]))

        # Train loop
        for step in range(train_step):
            # Fetch the next batch
            user_idx, pos_idx, neg_idx = next(batch_loader)
            # print(f"Step {step}: user_idx={user_idx}")

            # Perform batch training
            batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss = model.batch_loss(
                user_idx, pos_idx, neg_idx
            )

            # Accumulate losses
            train_res['batch_loss'] += batch_loss.item()
            train_res['bpr_loss'] += bpr_loss.item()
            train_res['emb_loss'] += l2_loss.item()
            train_res['cl_loss'] += cl_loss.item()
        end = datetime.datetime.now()
        return cl_loss * self.opt["PAAC_rate"]

    def cal_loss(self, opt, emb1, emb2):
        # 找到需要补齐的维度
        batch_size = max(emb1.size(0), emb2.size(0))  # 取最大的 batch_size
        feature_dim = emb1.size(1)

        # 如果 emb1 的样本数不足，进行零填充
        if emb1.size(0) < batch_size:
            padding = torch.zeros((batch_size - emb1.size(0), feature_dim), device=emb1.device)
            emb1 = torch.cat([emb1, padding], dim=0)

        # 如果 emb2 的样本数不足，进行零填充
        if emb2.size(0) < batch_size:
            padding = torch.zeros((batch_size - emb2.size(0), feature_dim), device=emb2.device)
            emb2 = torch.cat([emb2, padding], dim=0)

        # 计算正得分和负得分
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.opt["temp"])
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.opt["temp"]), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss
    def reconstruct_graph(self,opt, domain_id, batch,x_graphs,y_graphs,temperature):
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        user_numpy = user.detach().cpu().numpy()
        pos_item_numpy = pos_item.detach().cpu().numpy()
        neg_item_numpy = neg_item.detach().cpu().numpy()
        neg_item_flattened = neg_item_numpy.flatten()
        if domain_id == 0:
            batch_mf_loss, batch_emb_loss, batch_cen_loss, batch_cl_loss,int_emb = self.model_DCCF_s(user_numpy, pos_item_numpy,
                                                                                    neg_item_numpy[:, 0],domain_id)
            de_loss = self.reconstruction_loss_x(opt, domain_id, user, pos_item, neg_item[:,0], x_graphs,
                                                    temperature)
        else :
            batch_mf_loss, batch_emb_loss, batch_cen_loss, batch_cl_loss, int_emb = self.model_DCCF_t(user_numpy, pos_item_numpy,
                                                                                    neg_item_numpy[:, 0],domain_id)
            de_loss = self.reconstruction_loss_y(opt, domain_id, user, pos_item, neg_item[:,0], y_graphs,
                                                    temperature)
        DCCF_loss =  ( batch_emb_loss + batch_cen_loss + batch_cl_loss) * self.opt["DCCF_rate"]
        DE_loss = de_loss * self.opt["De_rate"]
        # 获取批量嵌入特征
        user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
        pos_item_feature = self.model.forward_item(domain_id, pos_item)
        neg_item_feature = self.model.forward_item(domain_id, neg_item)

        # 预测分数
        pos_score = self.model.predict_dot(user_feature, pos_item_feature)
        neg_score = self.model.predict_dot(user_feature, neg_item_feature)

        # 构造标签
        pos_labels, neg_labels = torch.ones(pos_score.size()), torch.zeros(neg_score.size())
        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        # 主损失
        loss = self.opt["lambda_loss"] * (
                self.criterion(pos_score, pos_labels) + self.criterion(neg_score, neg_labels)
        ) + (1 - self.opt["lambda_loss"]) * self.model.critic_loss
        # V = pos_item_feature
        # U = user_feature
        # # # 计算正则化项
        # # # 正则化项计算
        # VU_t = torch.matmul(V, U.T)  # (batch_size × batch_size)
        # e = torch.ones(U.size(0), device=U.device)  # 全 1 向量
        # VU_t_e = torch.matmul(VU_t, e)  # (batch_size)
        # numerator = torch.norm(VU_t_e) ** 2  # ||VU⊤e||²
        # #
        # UV_t = torch.matmul(U, V.T)  # (batch_size × batch_size)
        # UV_t_VU_t_e = torch.matmul(UV_t, VU_t_e)  # (batch_size)
        # epsilon = 1e-5  # 数值稳定性偏移
        # denominator = torch.norm(UV_t_VU_t_e) ** 2 + epsilon  # ||UV⊤VU⊤e||²
        # #
        # # # 调整正则化项的计算
        # scale_factor = 1e6  # 增大放大系数
        # beta = self.opt.get("beta", 0.8)  # 保留原有 β 值
        # #
        # # # 对分子和分母取平方根，调整量级
        # regularization = scale_factor * beta * (numerator / denominator)
        # # # print(regularization)
        # # # 更新损失
        # loss += regularization
        # 如果使用 Transformer 聚合器

        return loss, None, DCCF_loss, DE_loss, int_emb
    def predict(self, domain_id, eval_dataloader):
        MRR = 0.0
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        valid_entity = 0

        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)

            scores = scores.data.detach().cpu().numpy()

            for pred in scores:

                rank = (-pred).argsort().argsort()[0].item()

                valid_entity += 1
                MRR += 1 / (rank + 1)
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if valid_entity % 100 == 0:
                    print('+', end='')

        print("")
        metrics = {}
        # metrics["MRR"] = MRR / valid_entity
        # metrics["NDCG_5"] = NDCG_5 / valid_entity
        metrics["NDCG_10"] = NDCG_10 / valid_entity
        # metrics["HT_1"] = HT_1 / valid_entity
        # metrics["HT_5"] = HT_5 / valid_entity
        metrics["HT_10"] = HT_10 / valid_entity

        return metrics


    def predict_full_rank(self, domain_id, eval_dataloader, train_map, eval_map):

        def nDCG(ranked_list, ground_truth_length):
            dcg = 0
            idcg = IDCG(ground_truth_length)
            for i in range(len(ranked_list)):
                if ranked_list[i]:
                    rank = i + 1
                    dcg += 1 / math.log(rank + 1, 2)
            return dcg / idcg

        def IDCG(n):
            idcg = 0
            for i in range(n):
                idcg += 1 / math.log(i + 2, 2)
            return idcg

        def precision_and_recall(ranked_list, ground_number):
            hits = sum(ranked_list)
            pre = hits / (1.0 * len(ranked_list))
            rec = hits / (1.0 * ground_number)
            return pre, rec

        ndcg_list = []
        pre_list = []
        rec_list = []

        NDCG_10 = 0.0
        HT_10 = 0

        # pdb.set_trace()
        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)

            scores = scores.data.detach().cpu().numpy()
            user = user.data.detach().cpu().numpy()
            # pdb.set_trace()
            for idx, pred in enumerate(scores):
                rank = (-pred).argsort()
                score_list = []

                hr=0
                for i in rank:
                    i = i + 1
                    if (i in train_map[user[idx]]) and (i not in eval_map[user[idx]]):
                        continue
                    else:
                        if i in eval_map[user[idx]]:
                            hr = 1
                            # nd += 1 / np.log2(len(score_list) + 2)
                            score_list.append(1)
                        else:
                            score_list.append(0)
                        if len(score_list) == 10:
                            break

                HT_10 += hr

                pre, rec = precision_and_recall(score_list, len(eval_map[user[idx]]))
                pre_list.append(pre)
                rec_list.append(rec)
                ndcg_list.append(nDCG(score_list, len(eval_map[user[idx]])))

                if len(ndcg_list) % 100 == 0:
                    print('+', end='')
        print("")

        metrics = {}
        metrics["HT_10"] = HT_10 / len(ndcg_list)
        metrics["NDCG_10"] = sum(ndcg_list) / len(ndcg_list)

        # metrics["MRR"] = 0
        # precision = sum(pre_list) / len(pre_list)
        # recall = sum(rec_list) / len(rec_list)
        # metrics["F_10"] = 2 * precision * recall / (precision + recall + 0.00000001)

        return metrics
