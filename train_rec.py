import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils.GraphMaker import GraphMaker
from model.trainer import CrossTrainer
from utils.data import *
import os
import json
import sys
import pickle
import pdb
import time
import copy
import os
from utils.load_data import Data
from tqdm import tqdm
import torch.optim as optim
from model.model import DCCF
from utils.loader import dataLoader
from utils.GraphMaker_ import GraphMaker_
from model.dataloader import Data
import datetime
import dataloader,utils
from model.PAAC import PAAC
from Deno.GraphMaker_ import GraphMaker_
from Deno.loader import dataLoader



sys.path.insert(1, 'src')

def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('WSDM')

    # DATA  Arguments
    parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us", help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training')
    parser.add_argument('--task', type=str, default='dual-user-intra', help='dual-user-intra, dual-user-inter, multi-item-intra, multi-user-intra')

    # MODEL Arguments
    parser.add_argument('--model', type=str, default='UniCDR', help='right model name')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate of interactions')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epoches')
    parser.add_argument('--aggregator', type=str, default='mean', help='switching the user-item aggregation')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-7, help='the L2 weight')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='decay learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimensions')
    parser.add_argument('--num_negative', type=int, default=10, help='num of negative samples during training')
    parser.add_argument('--maxlen', type=int, default=10, help='num of item sequence')
    parser.add_argument('--dropout', type=float, default=0.3, help='random drop out rate')
    parser.add_argument('--save', action='store_true', help='save model?')
    parser.add_argument('--lambda', type=float, default=50, help='the parameter of EASE')
    parser.add_argument('--lambda_a', type=float, default=0.5, help='for our aggregators')
    parser.add_argument('--lambda_loss', type=float, default=1.0, help='the parameter of loss function')
    parser.add_argument('--lambda_pp', type=float, default=1.0, help='for our aggregators')
    parser.add_argument('--static_sample', action='store_true', help='accelerate the dataloader')

    # others
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=45, help='manual seed init')
    parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')

    parser.add_argument('--hidden_dim_', type=int, default=32, help='GNN network hidden embedding dimension.')
    parser.add_argument('--gamma', type=float, default=-0.45)
    parser.add_argument('--zeta', type=float, default=1.05)
    parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss on laplacian matrix.')
    parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
    parser.add_argument('--ssl_reg_cl', default=100, type=float, help='weight for contrative learning')
    parser.add_argument('--ssl_reg_game', default=1, type=float, help='weight for contrative learning')
    parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')

    parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--user_batch_size', type=int, default=64, help='Training batch size.')
    # parser.add_argument('--GNN_', type=int, default=2, help='GNN layer.')
    parser.add_argument('--bce', dest='bce', action='store_true', default=False)
    parser.add_argument('--inject', type=float, default=0, help='Inject 0 , 0.05, 0.1')
    parser.add_argument('--lambda_test', type=float, default=0.1, help='')
    parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
    parser.add_argument('--data_batch_size', type=int, default=1024, help='Training batch size.')
    parser.add_argument("--beta1", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.7)



    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
    parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
    parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
    parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
    parser.add_argument('--emb_reg', type=float, default=2.5e-5, help='Regularizations.')
    parser.add_argument('--cen_reg', type=float, default=5e-3, help='Regularizations.')
    parser.add_argument('--mf_reg', type=float, default=0.1, help='Regularizations.')
    parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
    parser.add_argument('--ssl_reg', type=float, default=1e-1, help='Reg weight for ssl loss')
    parser.add_argument('--n_layers', type=int, default=4, help='Layer numbers.')
    parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
    # parser.add_argument('--temp', type=float, default=1, help='temperature in ssl loss')
    parser.add_argument('--show_step', type=int, default=1, help='Test every show_step epochs.')
    parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
    # parser.add_argument('--GNN', type=int, default=3, help='GNN layer.')

    parser.add_argument('--leakey', type=float, default=0.1)
    # parser.add_argument('--GNN_', type=int, default=2, help='GNN layer.')

    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--ssl_ib', type=float, default=0.1, help='Reg weight for ib loss')

    # model description
    # args = argparse.ArgumentParser(description="PAAC")

    # dataset

    ##OOD
    parser.add_argument('--dataset_path', default='./datasets/dual-user-intra/dataset', type=str)
    parser.add_argument('--result_path', default='OOD_result', type=str)

    parser.add_argument('--bpr_num_neg', default=1, type=int)

    # LightGCN model
    parser.add_argument('--model_p', default='PAAC', type=str)
    parser.add_argument('--decay', default=0.0001, type=float)
    # parser.add_argument('--lr', default=0.001, type=float)
    # parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--eps_PAAC', default=0.2, type=float)
    # 控制pop和unpop的参数
    parser.add_argument('--cl_rate', default=0.5, type=float)
    parser.add_argument('--temperature', default=0.2, type=float)
    # parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--align_reg', default=1, type=int)
    parser.add_argument('--lambada', default=0.5, type=float)
    parser.add_argument('--lambda_3', default=0.5, type=float)
    parser.add_argument('--gama', default=0.5, type=float)
    # args.add_argument('--align_reg_list', default='[100]', type=str)

    # train
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--EarlyStop', default=10, type=int)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--DCCF_rate',default=1, type=float)
    parser.add_argument('--PAAC_rate',default=1, type=float)
    parser.add_argument('--De_rate', default=1, type=float)
    # parser.add_argument('--PAAC_rate', default=1, type=float)
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='number of GNN layers for item enhancement')
    # parser.add_argument('--num_epoch', default=1, type=int)

    parser.add_argument(
        '--topks', default='[20]', type=str)

    return parser

def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    opt = vars(args)


    opt["device"] = torch.device('cuda' if torch.cuda.is_available() and opt["cuda"] else 'cpu')





    def print_config(config):
        info = "Running with the following configs:\n"
        for k, v in config.items():
            info += "\t{} : {}\n".format(k, str(v))
        print("\n" + info + "\n")

    if opt["task"] == "multi-user-intra":
        opt["maxlen"] = 50

    #print_config(opt)

    print(f'Running experiment on device: {opt["device"]}')

    # def unpack_batch_( batch):
    #     inputs = [Variable(b) for b in batch]
    #     source_user = inputs[0]
    #     # print("===========================")
    #     source_pos_item = inputs[1]
    #     source_neg_item = inputs[2]
    #
    #     target_user = inputs[3]
    #     target_pos_item = inputs[4]
    #     target_neg_item = inputs[5]
    #
    #     return source_user, source_pos_item,source_neg_item, target_user, target_pos_item, target_neg_item

    def seed_everything(seed=1111):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(opt["seed"])

    ############
    ## All Domains Data
    ############

    if "dual" in opt["task"]:
        filename = opt["domains"].split("_")
        opt["domains"] = []
        opt["domains"].append(filename[0] + "_" + filename[1])
        opt["domains"].append(filename[1] + "_" + filename[0])

    else:
        opt["domains"] = opt["domains"].split('_')

    print("Loading domains:", opt["domains"])

    domain_list = opt["domains"]
    opt["user_max"] = []
    opt["item_max"] = []
    task_gen_all = {}
    domain_id = {}


    all_domain_list = []
    all_domain_set = []
    all_inter = 0
    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join("./datasets/"+str(opt["task"]) + "/dataset/", cur_domain + "/train.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        all_domain_list.append({})
        all_domain_set.append({})
        max_user = 0
        max_item = 0
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter+=1
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                max_user = max(max_user, user)
                max_item = max(max_item, item)
                if user not in all_domain_list[idx].keys():
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    all_domain_list[idx][user].append(item)
                    all_domain_set[idx][user].add(item)

        opt["user_max"].append(max_user + 1)
        opt["item_max"].append(max_item + 1)
        # ************* source data *****************
    if opt["inject"] == 0:
        source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train.txt"
    elif opt["inject"] == 0.05:
        source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train_noisy_0.05.txt"
    else:
        source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train_noisy_0.1.txt"
    # total_graphs = GraphMaker(opt, all_domain_list)
    if "dual" in opt["task"]:
        x_graphs = GraphMaker_(opt, source_train_data).adj
        # x_UV = GraphMaker_(opt, source_train_data).UV
        # x_VU = GraphMaker_(opt, source_train_data).VU
        # # x_batch = dataLoader(opt["domains"][0],opt["batch_size"],opt,evaluation=-1)
        print(opt["domains"][0] + "数据已经准备好了")
        print(source_train_data)

    # source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train.txt"
    # x_UV = GraphMaker_(opt, source_train_data).UV
    # x_VU = GraphMaker_(opt, source_train_data).VU
    total_graphs = GraphMaker(opt, all_domain_list)


    # repeat the above operation, add the item similarity (ease) value for each interaction.
    all_domain_list = []
    all_domain_set = []
    all_inter = 0

    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/train.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        if opt["aggregator"] == "item_similarity":
            ease_dense = total_graphs.ease[idx].to_dense()
        #if opt["aggregator"] == "Transformer" and opt["task"] == "multi-item-intra":
        if opt["aggregator"] == "Transformer" and "multi" in opt["task"]:
            ease_dense = total_graphs.ease[idx].to_dense()

        all_domain_list.append({})
        all_domain_set.append({})
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                if user not in all_domain_list[idx].keys():
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    if opt["aggregator"] == "item_similarity":
                        all_domain_list[idx][user].append([item, ease_dense[user][item]])
                    #elif opt["task"] == "multi-item-intra" and opt["aggregator"] == "Transformer":
                    elif "multi" in opt["task"] and opt["aggregator"] == "Transformer":
                        all_domain_list[idx][user].append([item, ease_dense[user][item]])
                    else:
                        all_domain_list[idx][user].append([item, 1])
                    all_domain_set[idx][user].add(item)

        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        cur_src_task_generator = TaskGenerator(cur_src_data_dir, opt, all_domain_list, all_domain_set, idx,
                                               total_graphs)
        task_gen_all[idx] = cur_src_task_generator
        domain_id[cur_domain] = idx
    if "dual" in opt["task"]:
        # ************* target data *****************
        if opt["inject"] == 0:
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][1] + "/train.txt"
        elif opt["inject"] == 0.05:
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][
                1] + "/train_noisy_0.05.txt"
        else:
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][
                1] + "/train_noisy_0.1.txt"
        y_graphs = GraphMaker_(opt, target_train_data).adj
        # y_UV = GraphMaker_(opt, target_train_data).UV
        # y_VU = GraphMaker_(opt, target_train_data).VU
        # print(target_train_data)
        data = dataLoader(opt["domains"][0], opt["batch_size"], opt, evaluation=-1)
        # source_user,source_item,target_user,target_item = dataLoader.get_source_target_users(data)
        # print(source_user, source_item, target_user, target_item)
        print(opt["domains"][1] + "数据已经准备好了")
        print("source_user_num", opt["source_user_num"])
        print("target_user_num", opt["target_user_num"])
        print("source_item_num", opt["source_item_num"])
        print("target_item_num", opt["target_item_num"])
        print(cur_src_data_dir)

    # data = dataLoader(opt["domains"][0], opt["batch_size"], opt, evaluation=-1)

    target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][1] + "/train.txt"
    # y_UV = GraphMaker_(opt, target_train_data).UV
    # y_VU = GraphMaker_(opt, target_train_data).VU


    train_domains = MetaDomain_Dataset(task_gen_all, num_negatives=opt["num_negative"], meta_split='train')
    train_dataloader = MetaDomain_DataLoader(train_domains, sample_batch_size=opt["batch_size"] // len(domain_list), shuffle=True)
    opt["num_domains"] = train_dataloader.num_domains
    opt["domain_id"] = domain_id

    ############
    ## Validation and Test
    ############
    if "inter" in opt["task"]:
        opt["shared_user"] = 1e9
    valid_dataloader = {}
    test_dataloader = {}
    for cur_domain in domain_list:
        if opt["task"] == "dual-user-intra":
            domain_valid = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/test.txt")
        else:
            domain_valid = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/valid.txt")
        domain_test = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/test.txt")
        valid_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_valid, 100)
        test_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_test, 100)
    

    print("the user number of different domains", opt["user_max"])
    print("the item number of different domains", opt["item_max"])

    ############
    ## Model
    ############
    mymodel = CrossTrainer(opt)
    mymodel.model.graph_maker = total_graphs


    ############
    ## Train
    ############
    dev_score_history = []
    for i in range(opt["num_domains"]):
        dev_score_history.append([0])


    current_lr = opt['lr']
    iteration_num = 500

    print("per batch of an epoch:", iteration_num)
    global_step = 0

    # 在循环外部定义最佳结果记录变量（每个域独立跟踪）

    best_ndcg_per_domain = {domain: 0 for domain in valid_dataloader.keys()}  # 新增存储最大 NDCG_10
    best_hr_per_domain = {domain: 0 for domain in valid_dataloader.keys()}  # 新增存储最大 HR
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    file_name = '_'.join(
        (str(opt["layers"]), str(opt["cl_rate"]), str(opt["align_reg"]), str(opt["gama"]),
         str(opt["lambda"]), timestamp))
    print(file_name)
    print(opt["result_path"])
    print(opt["model_p"])
    print(opt["domains"][1])
    result_path = '/'.join((opt["result_path"],opt["model_p"], opt["domains"][1], file_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logger_file_name = os.path.join(result_path, 'train_logger')
    logger = utils.get_logger(logger_file_name)
        # Load Data
    logger.info('------Load Source_Data-----')
    source_data = dataloader.Data(opt, logger, opt["domains"][0])
    source_data.norm_adj = dataloader.LaplaceGraph(
        source_data.num_users, source_data.num_items, source_data.train_U2I).generate()

    # Load Model
    logger.info('------Load Model-----')
    source_model = PAAC(opt, source_data)
    source_model.to(source_model.device)
    # Load Data
    logger.info('------Load Target_Data-----')
    target_data = dataloader.Data(opt, logger, opt["domains"][1])
    target_data.norm_adj = dataloader.LaplaceGraph(
        target_data.num_users, target_data.num_items, target_data.train_U2I).generate()

    # Load Model
    logger.info('------Load Model-----')
    target_model = PAAC(opt, target_data)
    target_model.to(target_model.device)
    # EarlyStopping
    early_stopping = utils.EarlyStopping(logger,opt["EarlyStop"], verbose=True, path=result_path)

    for epoch in range(0, opt["num_epoch"] + 1):
        start_time = time.time()
        print('Epoch {} starts !'.format(epoch))
        total_loss = [0]
        total_ploss = [0]
        total_DCCF = [0]
        total_PAAC = [0]
        total_CL = [0]
        int_emb_s = []
        int_emb_t = []

        train_loss = 0
        mymodel_loss = 0
        loss_list = []
        ep =0
        # de_loss = mymodel.train_IB(opt, x_UV, x_VU, y_UV, y_VU).detach()
        for i in range(opt["num_domains"]):
            loss_list.append([0])
        for iteration in range(iteration_num):
            if epoch == 0:
               continue
            if iteration % 10 == 0:

                print(".", end="")
            mymodel.model.train()
            mymodel.optimizer.zero_grad()
            mymodel.model.item_embedding_select()
            mymodel.optimizer.zero_grad()
            mymodel_loss = 0
            myprop_loss = 0
            for idx in range(opt["num_domains"]):  # get one batch from each dataloade
                temperature = max(0.05, 2 * pow(0.98, ep))
                ep += 1
                global_step += 1
                cur_train_dataloader = train_dataloader.get_iterator(idx)
                try:
                    batch_data = next(cur_train_dataloader)
                except:
                    new_train_iterator = iter(train_dataloader[idx])
                    batch_data = next(new_train_iterator)
                cur_loss,prop_loss,DCCF_loss,De_loss,int_emb = mymodel.reconstruct_graph(opt,idx, batch_data, x_graphs, y_graphs, temperature)
                if idx == 0:
                    int_emb_s = int_emb
                    PAAC_loss = mymodel.train(opt, source_data, source_model, early_stopping, logger)
                else :
                    int_emb_t = int_emb
                    PAAC_loss = mymodel.train(opt, target_data, target_model, early_stopping, logger)
                mymodel_loss += De_loss
                mymodel_loss += cur_loss
                mymodel_loss += DCCF_loss
                mymodel_loss += PAAC_loss
                loss_list[idx].append(cur_loss.item())
                total_PAAC.append(PAAC_loss.item())
                total_CL.append(De_loss.item())
                total_DCCF.append(DCCF_loss.item())
                total_loss.append(cur_loss.item())
                total_loss.append(DCCF_loss.item())
                total_loss.append(PAAC_loss.item())
                total_loss.append(De_loss.item())

                # if opt["aggregator"] == "Transformer1":
                #     mymodel_loss += prop_loss
                #     if "multi" in opt["task"] and mymodel.model.warmup == 1:
                #         total_ploss.append(0)
                #     else:
                #         total_ploss.append(prop_loss.item())

            s_t_loss = mymodel.cal_loss(opt, int_emb_s, int_emb_t)
            mymodel_loss += s_t_loss * opt["lambda_3"]
            total_DCCF.append(s_t_loss.item())
            mymodel_loss.backward()
            mymodel.optimizer.step()
        if opt["aggregator"] == "Transformer":
            print("Average loss:", sum(total_loss) / len(total_loss), "prop_loss", sum(total_ploss) / len(total_ploss),
                  "测试：DCCF_loss * DCCF_rate:", sum(total_DCCF) / len(total_DCCF), "PAAC_loss:", sum(total_PAAC) / len(total_PAAC),"DE_loss :", sum(total_CL) / len(total_CL),
                   "time: ", (time.time() - start_time) / 60, "(min) current lr: ", current_lr)
        print("s_T_LOSS：Average loss:", sum(total_loss) / len(total_loss), "time: ", (time.time() - start_time) / 60,
              "(min) current lr: ",
              current_lr)

        print('-' * 80)

        if epoch % 5:
            continue

        for idx in range(opt["num_domains"]):
            print(idx, "loss is: ", sum(loss_list[idx]) / len(loss_list[idx]))

        print('Make prediction:')
        # validation data prediction
        valid_start = time.time()

        mymodel.model.eval()
        mymodel.model.item_embedding_select()

        decay_switch = 0


        for idx, cur_domain in enumerate(valid_dataloader):
            if opt["task"] == "multi-user-intra":
                metrics = mymodel.predict_full_rank(idx, valid_dataloader[cur_domain], all_domain_set[idx],
                                                    task_gen_all[idx].eval_set)
            else:
                metrics = mymodel.predict(idx, valid_dataloader[cur_domain])

            print("\n-------------------" + cur_domain + "--------------------")
            print(metrics)

            if metrics["NDCG_10"] > max(dev_score_history[idx]):
                # test data prediction
                print(cur_domain, " better results!")

                # mymodel.save("./checkpoints/best_inter_sp.pt")
                # print("best model saved!")

                if opt["task"] == "multi-user-intra":
                    test_metrics = mymodel.predict_full_rank(idx, test_dataloader[cur_domain], all_domain_set[idx],
                                                             task_gen_all[idx].eval_set)
                else:
                    test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])

                print(test_metrics)

                # 更新当前域的最大 NDCG_10
                if metrics["NDCG_10"] > best_ndcg_per_domain[cur_domain]:
                    if metrics["NDCG_10"] > test_metrics["NDCG_10"]:
                        best_ndcg_per_domain[cur_domain] = metrics["NDCG_10"]
                    else:
                        best_ndcg_per_domain[cur_domain] = test_metrics["NDCG_10"]
                # 更新当前域的最大 HR
                if "HT_10" in metrics and metrics["HT_10"] > best_hr_per_domain[cur_domain]:
                    if metrics["HT_10"] > test_metrics["HT_10"]:
                        best_hr_per_domain[cur_domain] = metrics["HT_10"]
                    else:
                        best_hr_per_domain[cur_domain] = test_metrics["HT_10"]
                    # best_hr_per_domain[cur_domain] = test_metrics["HT_10"]
            else:
                decay_switch += 1
            dev_score_history[idx].append(metrics["NDCG_10"])
        print("valid time:  ", (time.time() - valid_start) / 60, "(min)")

        if epoch > opt['decay_epoch']:
            print("Before warmup:", mymodel.model.warmup)
            mymodel.model.warmup = 0
            print("Now warmup:", mymodel.model.warmup)

        # lr schedule
        print("decay_switch: ", decay_switch)
        if (epoch > opt['decay_epoch']) and (decay_switch > opt["num_domains"] // 2) and (opt['optim'] in ['sgd', 'adagrad','adadelta','adam']):
            current_lr *= opt['lr_decay']
            mymodel.update_lr(current_lr)

            # 在训练结束时输出两个域的最佳 NDCG_10 和 HR_10
    print('\nFinal Best Results per Domain:')
    for domain in valid_dataloader.keys():
        print(f"Domain {domain}: Best NDCG_10 = {best_ndcg_per_domain[domain]}, Best HT_10 = {best_hr_per_domain[domain]}")
    print('Experiment finished successfully!')

if __name__ == "__main__":
    main()
