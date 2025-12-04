

Official codebase for the paper PACL-CDR:Popularity-Adaptive Contrastive Learning for Cross-Domain Recommendation



## Overview

![overview](https://github.com/Canyizl/PPA-for-CDR/blob/main/fig/introfig.png)

**Abstract:** Cross-domain recommendation (CDR) has garnered significant attention for its potential to mitigate data sparsity and cold-start issues by facilitating knowledge transfer from source to target domains. While recent contrastive CDR methods have attempted to address information heterogeneity and behavioral discrepancies via positive and negative pair construction, two persistent challenges remain: (1) popularity bias, where unpopular items are frequently misclassified as negative samples, resulting in an overemphasis on popular items and the neglect of unpopular items; and (2) over-reliance on overlapping users, which restricts the construction of positive and negative pairs, hinders the modeling of non-overlapping user preferences, and results in incomplete or biased knowledge transfer. 

To address these challenges, we propose a novel cross-domain recommendation framework, PACL-CDR, which incorporates a Debiasing Popularity Module (DPM) and an Intent Enhancement Module (IEM). (1) DPM leverages alignment and contrastive learning strategies to balance the feature representations of head and tail items, and further enhances semantic consistency between them by constructing cross-domain item tags. (2) IEM shifts the focus of knowledge transfer from overlapping users to underlying user intents, thereby strengthen user connections and facilitating preference modeling for non-overlapping users. Extensive experiments on four benchmark cross-domain recommendation datasets demonstrate that PACL-CDR consistently outperforms state-of-the-art baselines, validating its effectiveness and novelty.

## Datasets

We use the datasets provided by [DisenCDR](https://github.com/cjx96/DisenCDR)

All used datasets can be downloaded at [WSDM2023-UniCDR-datasets](https://drive.google.com/drive/folders/1DCYiFU6GCVj681GKYUY2d_BJFln1-8gL?usp=share_link)
Note that all datasets are required to unzip in the root directory.



## Usage

Running example:

```shell
# sport_phone
CUDA_VISIBLE_DEVICES=1 python -u train_rec.py  --static_sample --cuda --domains sport_cloth --aggregator Transformer > Tppa_dual_user_intra_sport_cloth.log 2>&1&

# electronic_phone
CUDA_VISIBLE_DEVICES=3 python -u train_rec.py  --static_sample --cuda --domains sport_cloth --task dual-user-inter --aggregator Transformer --num_epoch 100 --batch_size 1024 --lr 0.001  --n_intents 15 --dataset_path ./datasets/dual-user-inter/dataset --emb_size 129 --seed 30 --n_layers 3 --layers 3 --embed_size 32 --lambada 0.5 --gama 0.5

```
