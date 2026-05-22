import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
#
class IEM(nn.Module):
    def __init__(self, data_config, opt):
        super(IEM, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = opt["embed_size"]
        self.n_layers = opt["n_layers"]
        self.n_intents = opt["n_intents"]
        self.temp = opt["temp"]

        self.batch_size = opt["batch_size"]
        self.emb_reg = opt["emb_reg"]
        self.cen_reg = opt["cen_reg"]
        self.ssl_reg = opt["ssl_reg"]

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)

        _user_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)
        _item_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def _cal_sparse_adj(self):

        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])

        return G_indices, G_values

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = head_embeddings.float().to('cuda')
        tail_embeddings = tail_embeddings.float().to('cuda')
        self.all_h_list = self.all_h_list.long().to('cuda')
        self.all_t_list = self.all_t_list.long().to('cuda')
        if torch.any(torch.sum(head_embeddings ** 2, dim=-1) == 0):
            print("Warning: Zero norm detected in head_embeddings!")
        if torch.any(torch.sum(tail_embeddings ** 2, dim=-1) == 0):
            print("Warning: Zero norm detected in tail_embeddings!")
        head_embeddings = torch.nn.functional.normalize(head_embeddings, p=2, dim=-1, eps=1e-8)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings, p=2, dim=-1, eps=1e-8)

        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha,
                                             sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        del edge_alpha, A_tensor, D_scores_inv, head_embeddings, tail_embeddings
        torch.cuda.empty_cache()  
        return G_indices, G_values
    def inference(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        gnn_embeddings = []
        int_embeddings = []
        gaa_embeddings = []
        iaa_embeddings = []

        for i in range(0, self.n_layers):
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)

            gaa_layer_embeddings = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])
            iaa_layer_embeddings = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])

            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gaa_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)

            all_embeddings.append(
                gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings +
                all_embeddings[i])

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        return gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb ,id):
        users = users.to(dtype=torch.long)
        items = items.to(dtype=torch.long)
        users = torch.clamp(users, min=0, max=self.n_users - 1)
        items = torch.clamp(items, min=0, max=self.n_items - 1)
        users[torch.isnan(users)] = 0
        users[torch.isinf(users)] = 0
        items[torch.isnan(items)] = 0
        items[torch.isinf(items)] = 0
        device = gnn_emb[0].device
        users = users.to(device)
        items = items.to(device)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.n_users, self.n_items], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.n_users, self.n_items], 0)
            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_int_embs)

            return cl_loss , u_int_embs
            return loss


    def forward(self, users, pos_items, neg_items,id):
        int_emb_s = []
        int_emb_t = []
        users = torch.LongTensor(users).cuda()
        pos_items = torch.LongTensor(pos_items).cuda()
        neg_items = torch.LongTensor(neg_items).cuda()
        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self.inference()
        if self.ua_embedding is None or self.ia_embedding is None:
            raise ValueError("ua_embedding or ia_embedding is not initialized!")
        pos_items = torch.clamp(pos_items, min=0, max=self.ia_embedding.size(0) - 1)
        neg_items = torch.clamp(neg_items, min=0, max=self.ia_embedding.size(0) - 1)
        if torch.min(users) < 0 or torch.max(users) >= self.ua_embedding.size(0):
            raise ValueError(
                f"Users index out of range: min={torch.min(users)}, max={torch.max(users)}, embedding size={self.ua_embedding.size(0)}")
        if torch.min(pos_items) < 0 or torch.max(pos_items) >= self.ia_embedding.size(0):
            raise ValueError(
                f"Pos_items index out of range: min={torch.min(pos_items)}, max={torch.max(pos_items)}, embedding size={self.ia_embedding.size(0)}")
        if torch.min(neg_items) < 0 or torch.max(neg_items) >= self.ia_embedding.size(0):
            raise ValueError(
                f"Neg_items index out of range: min={torch.min(neg_items)}, max={torch.max(neg_items)}, embedding size={self.ia_embedding.size(0)}")
        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)  
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)  
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores)) 
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) +
                    pos_embeddings_pre.norm(2).pow(2) +
                    neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss
        cen_loss = (self.user_intent.norm(2).pow(2) +
                    self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_reg * cen_loss
        if id == 0:
            cl_loss, int_emb= self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, gaa_embeddings,
                                                     iaa_embeddings, id)
        else:
            cl_loss, int_emb = self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings,
                                                     gaa_embeddings,
                                                     iaa_embeddings, id)

        return mf_loss, emb_loss, cen_loss, self.ssl_reg *cl_loss , int_emb

    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(users).cuda()]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings