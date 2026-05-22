import argparse
import random
import numpy as np
import torch
import os
import sys
import time
import datetime
import codecs
from itertools import cycle

from utils.GraphMaker import GraphMaker
from model.trainer import CrossTrainer
from utils.data import *
from Deno.GraphMaker_ import GraphMaker_
from Deno.loader import dataLoader
import dataloader, utils
from model.DPM import DPM

sys.path.insert(1, 'src')


def create_arg_parser():
    parser = argparse.ArgumentParser('WSDM')

    # Data
    parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us",
                        help='Source domains separated by "||"')
    parser.add_argument('--task', type=str, default='dual-user-intra', help='dual-user-intra')

    # Model
    parser.add_argument('--model', type=str, default='PACL_CDR')
    parser.add_argument('--mask_rate', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--aggregator', type=str, default='Transformer')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--l2_reg', type=float, default=1e-7)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_negative', type=int, default=10)
    parser.add_argument('--maxlen', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--lambda', type=float, default=50)
    parser.add_argument('--lambda_a', type=float, default=0.5)
    parser.add_argument('--lambda_loss', type=float, default=1.0)
    parser.add_argument('--lambda_pp', type=float, default=1.0)
    parser.add_argument('--static_sample', action='store_true')

    # Training hyperparameters
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--decay_epoch', type=int, default=10)
    parser.add_argument('--hidden_dim_', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=-0.45)
    parser.add_argument('--zeta', type=float, default=1.05)
    parser.add_argument('--lambda0', type=float, default=1e-4)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--ib_reg', type=float, default=0.1)
    parser.add_argument('--ssl_reg_cl', type=float, default=100)
    parser.add_argument('--ssl_reg_game', type=float, default=1)
    parser.add_argument('--GNN', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--user_batch_size', type=int, default=64)
    parser.add_argument('--bce', dest='bce', action='store_true', default=False)
    parser.add_argument('--inject', type=float, default=0, help='Noise injection rate: 0, 0.05, or 0.1')
    parser.add_argument('--lambda_test', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--data_batch_size', type=int, default=1024)
    parser.add_argument('--beta1', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--train_num', type=int, default=10000)
    parser.add_argument('--sample_num', type=int, default=40)
    parser.add_argument('--emb_reg', type=float, default=2.5e-5)
    parser.add_argument('--cen_reg', type=float, default=5e-3)
    parser.add_argument('--mf_reg', type=float, default=0.1)
    parser.add_argument('--n_batch', type=int, default=40)
    parser.add_argument('--ssl_reg', type=float, default=1e-1)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_intents', type=int, default=15)
    parser.add_argument('--show_step', type=int, default=1)
    parser.add_argument('--Ks', nargs='?', default='[20, 40]')
    parser.add_argument('--leakey', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--ssl_ib', type=float, default=0.1)

    # Dataset / OOD
    parser.add_argument('--dataset_path', default='./datasets', type=str)
    parser.add_argument('--result_path', default='OOD_result', type=str)
    parser.add_argument('--bpr_num_neg', default=1, type=int)
    parser.add_argument('--topks', default='[20]', type=str)

    # DPM / loss weights
    parser.add_argument('--model_p', default='DPM', type=str)
    parser.add_argument('--decay', default=0.0001, type=float)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--eps_DPM', default=0.2, type=float)
    parser.add_argument('--cl_rate', default=0.5, type=float)
    parser.add_argument('--temperature', default=0.4, type=float)
    parser.add_argument('--align_reg', default=1, type=int)
    parser.add_argument('--lambada', default=0.5, type=float)
    parser.add_argument('--lambda_3', default=0.5, type=float)
    parser.add_argument('--gama', default=0.6, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--EarlyStop', default=10, type=int)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--IEM_rate', default=1, type=float)
    parser.add_argument('--DPM_rate', default=1, type=float)
    parser.add_argument('--De_rate', default=1, type=float)
    parser.add_argument('--num_gnn_layers', type=int, default=2)

    return parser


def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)
    return all_h_list, all_t_list, all_v_list


def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    opt = vars(args)

    opt["device"] = torch.device('cuda' if torch.cuda.is_available() and opt["cuda"] else 'cpu')
    print(f'Running experiment on device: {opt["device"]}')

    if opt["task"] == "multi-user-intra":
        opt["maxlen"] = 50

    seed_everything(opt["seed"])
    if "dual" in opt["task"]:
        filename = opt["domains"].split("_")
        opt["domains"] = [
            filename[0] + "_" + filename[1],
            filename[1] + "_" + filename[0],
        ]
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

    # First pass: collect user/item counts
    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join("./datasets/" , cur_domain + "/train.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')
        all_domain_list.append({})
        all_domain_set.append({})
        max_user = 0
        max_item = 0
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                max_user = max(max_user, user)
                max_item = max(max_item, item)
                if user not in all_domain_list[idx]:
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    all_domain_list[idx][user].append(item)
                    all_domain_set[idx][user].add(item)
        opt["user_max"].append(max_user + 1)
        opt["item_max"].append(max_item + 1)

    # Build source graph
    def _train_path(domain_idx):
        base = "./datasets/" + opt["domains"][domain_idx]
        if opt["inject"] == 0:
            return base + "/train.txt"
        elif opt["inject"] == 0.05:
            return base + "/train_noisy_0.05.txt"
        else:
            return base + "/train_noisy_0.1.txt"

    source_train_data = _train_path(0)
    target_train_data = _train_path(1)

    if "dual" in opt["task"]:
        x_graphs = GraphMaker_(opt, source_train_data).adj
        print(f"Source domain graph ready: {opt['domains'][0]}")

    total_graphs = GraphMaker(opt, all_domain_list)

    # Second pass: build interaction lists with optional item similarity weights
    all_domain_list = []
    all_domain_set = []
    all_inter = 0

    for idx, cur_domain in enumerate(domain_list):
        cur_src_data_dir = os.path.join("./datasets/" , cur_domain + "/train.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        use_ease = (
            opt["aggregator"] == "item_similarity"
            or (opt["aggregator"] == "Transformer" and "multi" in opt["task"])
        )
        ease_dense = total_graphs.ease[idx].to_dense() if use_ease else None

        all_domain_list.append({})
        all_domain_set.append({})
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                if user not in all_domain_list[idx]:
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    weight = ease_dense[user][item] if use_ease else 1
                    all_domain_list[idx][user].append([item, weight])
                    all_domain_set[idx][user].add(item)

        cur_src_task_generator = TaskGenerator(cur_src_data_dir, opt, all_domain_list, all_domain_set, idx, total_graphs)
        task_gen_all[idx] = cur_src_task_generator
        domain_id[cur_domain] = idx

    if "dual" in opt["task"]:
        y_graphs = GraphMaker_(opt, target_train_data).adj
        data = dataLoader(opt["domains"][0], opt["batch_size"], opt, evaluation=-1)
        print(f"Target domain graph ready: {opt['domains'][1]}")
        print(f"  source_user_num={opt['source_user_num']}, source_item_num={opt['source_item_num']}")
        print(f"  target_user_num={opt['target_user_num']}, target_item_num={opt['target_item_num']}")

    train_domains = MetaDomain_Dataset(task_gen_all, num_negatives=opt["num_negative"], meta_split='train')
    train_dataloader = MetaDomain_DataLoader(
        train_domains, sample_batch_size=opt["batch_size"] // len(domain_list), shuffle=True
    )
    opt["num_domains"] = train_dataloader.num_domains
    opt["domain_id"] = domain_id

    if "inter" in opt["task"]:
        opt["shared_user"] = 1e9

    valid_dataloader = {}
    test_dataloader = {}
    for cur_domain in domain_list:
        if opt["task"] == "dual-user-intra":
            domain_valid = os.path.join("./datasets/" , cur_domain + "/test.txt")
        else:
            domain_valid = os.path.join("./datasets/" , cur_domain + "/valid.txt")
        domain_test = os.path.join("./datasets/", cur_domain + "/test.txt")
        valid_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(domain_valid, 100)
        test_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(domain_test, 100)

    print("User counts per domain:", opt["user_max"])
    print("Item counts per domain:", opt["item_max"])

    mymodel = CrossTrainer(opt)
    mymodel.model.graph_maker = total_graphs

    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    file_name = '_'.join((
        str(opt["layers"]), str(opt["cl_rate"]), str(opt["align_reg"]),
        str(opt["gama"]), str(opt["lambda"]), timestamp
    ))
    result_path = '/'.join((opt["result_path"], opt["model_p"], opt["domains"][1], file_name))
    os.makedirs(result_path, exist_ok=True)

    logger_file_name = os.path.join(result_path, 'train_logger')
    logger = utils.get_logger(logger_file_name)

    logger.info('Loading source data...')
    source_data = dataloader.Data(opt, logger, opt["domains"][0])
    source_data.norm_adj = dataloader.LaplaceGraph(
        source_data.num_users, source_data.num_items, source_data.train_U2I
    ).generate()
    source_model = DPM(opt, source_data)
    source_model.to(source_model.device)

    logger.info('Loading target data...')
    target_data = dataloader.Data(opt, logger, opt["domains"][1])
    target_data.norm_adj = dataloader.LaplaceGraph(
        target_data.num_users, target_data.num_items, target_data.train_U2I
    ).generate()
    target_model = DPM(opt, target_data)
    target_model.to(target_model.device)

    early_stopping = utils.EarlyStopping(logger, opt["EarlyStop"], verbose=True, path=result_path)

    dev_score_history = [[0] for _ in range(opt["num_domains"])]
    top_ndcg_per_domain = {domain: 0.0 for domain in valid_dataloader}
    top_hr_per_domain = {domain: 0.0 for domain in valid_dataloader}

    current_lr = opt['lr']
    iteration_num = 500
    global_step = 0
    ep = 0

    print(f"Iterations per epoch: {iteration_num}")

    for epoch in range(0, opt["num_epoch"] + 1):
        start_time = time.time()
        print(f'Epoch {epoch}')
        epoch_losses = {
            'total':  [],
            'IEM':   [],
            'DPM':   [],
            'DE':     [],
        }
        domain_losses = [[0] for _ in range(opt["num_domains"])]
        int_emb_s, int_emb_t = [], []

        for iteration in range(iteration_num):
            if epoch == 0:
                continue
            if iteration % 10 == 0:
                print('.', end='', flush=True)

            mymodel.model.train()
            mymodel.optimizer.zero_grad()
            mymodel.model.item_embedding_select()
            mymodel.optimizer.zero_grad()

            mymodel_loss = 0
            for idx in range(opt["num_domains"]):
                temperature = max(0.05, 2 * pow(0.98, ep))
                ep += 1
                global_step += 1

                cur_train_dataloader = train_dataloader.get_iterator(idx)
                try:
                    batch_data = next(cur_train_dataloader)
                except StopIteration:
                    new_train_iterator = iter(train_dataloader[idx])
                    batch_data = next(new_train_iterator)

                cur_loss, prop_loss, IEM_loss, De_loss, int_emb = mymodel.reconstruct_graph(
                    opt, idx, batch_data, x_graphs, y_graphs, temperature
                )

                if idx == 0:
                    int_emb_s = int_emb
                    DPM_loss = mymodel.train(opt, source_data, source_model, early_stopping, logger)
                else:
                    int_emb_t = int_emb
                    DPM_loss = mymodel.train(opt, target_data, target_model, early_stopping, logger)

                mymodel_loss += cur_loss + IEM_loss + De_loss + DPM_loss

                domain_losses[idx].append(cur_loss.item())
                epoch_losses['total'].append(cur_loss.item())
                epoch_losses['IEM'].append(IEM_loss.item())
                epoch_losses['DPM'].append(DPM_loss.item())
                epoch_losses['DE'].append(De_loss.item())

            s_t_loss = mymodel.cal_loss(opt, int_emb_s, int_emb_t)
            mymodel_loss += s_t_loss * opt["lambda_3"]
            epoch_losses['IEM'].append(s_t_loss.item())

            mymodel_loss.backward()
            mymodel.optimizer.step()

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        elapsed = (time.time() - start_time) / 60
        print(f'\n'
              f'  avg loss : {avg(epoch_losses["total"]):.4f}  |  '
              f'IEM: {avg(epoch_losses["IEM"]):.4f}  |  '
              f'DPM: {avg(epoch_losses["DPM"]):.4f}  |  '
              f'DE: {avg(epoch_losses["DE"]):.4f}  |  '
              f'lr: {current_lr:.6f}  |  '
              f'time: {elapsed:.2f} min')

        if epoch % 5:
            continue

        for idx in range(opt["num_domains"]):
            print(f'  domain {idx} avg loss: {avg(domain_losses[idx]):.4f}')

        print('\nValidation:')
        valid_start = time.time()
        mymodel.model.eval()
        mymodel.model.item_embedding_select()
        decay_switch = 0
 
        for idx, cur_domain in enumerate(valid_dataloader):
            if opt["task"] == "multi-user-intra":
                metrics = mymodel.predict_full_rank(
                    idx, valid_dataloader[cur_domain], all_domain_set[idx], task_gen_all[idx].eval_set
                )
            else:
                metrics = mymodel.predict(idx, valid_dataloader[cur_domain])
 
            print(f'  [{cur_domain}]  NDCG@10={metrics["NDCG_10"]:.4f}  HT@10={metrics["HT_10"]:.4f}')
 
            if metrics["NDCG_10"] > max(dev_score_history[idx]):
                if opt["task"] == "multi-user-intra":
                    test_metrics = mymodel.predict_full_rank(
                        idx, test_dataloader[cur_domain], all_domain_set[idx], task_gen_all[idx].eval_set
                    )
                else:
                    test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])
 
                print(f'  [test] NDCG@10={test_metrics["NDCG_10"]:.4f}  HT@10={test_metrics["HT_10"]:.4f}')
 
                top_ndcg_per_domain[cur_domain] = max(metrics["NDCG_10"], test_metrics["NDCG_10"])
                if "HT_10" in metrics:
                    top_hr_per_domain[cur_domain] = max(metrics["HT_10"], test_metrics["HT_10"])
            else:
                decay_switch += 1
 
            dev_score_history[idx].append(metrics["NDCG_10"])
 
        print(f'  Validation time: {(time.time() - valid_start) / 60:.2f} min')
 
        if epoch > opt['decay_epoch']:
            mymodel.model.warmup = 0
 
        if (
            epoch > opt['decay_epoch']
            and decay_switch > opt["num_domains"] // 2
            and opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']
        ):
            current_lr *= opt['lr_decay']
            mymodel.update_lr(current_lr)
            print(f'  LR decayed to {current_lr:.6f}')
 
    print('\n' + '=' * 80)
    print('Results per Domain:')
    for domain in valid_dataloader:
        print(f'  {domain}: NDCG@10={top_ndcg_per_domain[domain]:.4f}  HT@10={top_hr_per_domain[domain]:.4f}')
    print('Experiment finished.')


if __name__ == "__main__":
    main()