import torch
import torch.nn.functional as F
import os

def InfoNCE(view1, view2, temperature):
    view1, view2 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3,temperature,gama):
    view1, view2,view3 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
    ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (gama*ttl_score_2+ttl_score_1+pos_score))
    return torch.mean(cl_loss)

# def InfoNCE_i(view1, view2, view3, temperature, gama):
#     # 确保保存目录存在
#     save_path = "./embeddings"
#     os.makedirs(save_path, exist_ok=True)
#
#
#
#     # 原始 InfoNCE 损失计算
#     view1, view2, view3 = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(view3, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#     ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
#     ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
#     ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
#     ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)
#
#     cl_loss = -torch.log(pos_score / (gama * ttl_score_2 + ttl_score_1 + pos_score))
#     return torch.mean(cl_loss)
