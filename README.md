

## PACL-CDR: Popularity-Adaptive Contrastive Learning for Cross-Domain Recommendation

## Overview

<table>
<tr>
<td align="center"><img src="https://github.com/Alpacanm/PACL-CDR/blob/main/fig/sample.png" width="420"></td>
<td align="center"><img src="https://github.com/Alpacanm/PACL-CDR/blob/main/fig/model.png" width="420"></td>
</tr>
</table>

**Abstract:** Cross-domain recommendation (CDR) has garnered significant attention for its potential to mitigate data sparsity and cold-start issues by facilitating knowledge transfer from source to target domains. While recent contrastive CDR methods have attempted to address information heterogeneity and behavioral discrepancies via positive and negative pair construction, two persistent challenges remain: (1) popularity bias, where unpopular items are frequently misclassified as negative samples, resulting in an overemphasis on popular items and the neglect of unpopular items; and (2) over-reliance on overlapping users, which restricts the construction of positive and negative pairs, hinders the modeling of non-overlapping user preferences, and results in incomplete or biased knowledge transfer.

To address these challenges, we propose a novel cross-domain recommendation framework, PACL-CDR, which incorporates a Debiasing Popularity Module (DPM) and an Intent Enhancement Module (IEM). (1) DPM leverages alignment and contrastive learning strategies to balance the feature representations of head and tail items, and further enhances semantic consistency between them by constructing cross-domain item tags. (2) IEM shifts the focus of knowledge transfer from overlapping users to underlying user intents, thereby strengthen user connections and facilitating preference modeling for non-overlapping users. Extensive experiments on four benchmark cross-domain recommendation datasets demonstrate that PACL-CDR consistently outperforms state-of-the-art baselines, validating its effectiveness and novelty.

## Datasets

We use the datasets provided by [DisenCDR](https://github.com/cjx96/DisenCDR).

## Baselines

The baseline implementations are based on the following repositories:

- [DisenCDR](https://github.com/cjx96/DisenCDR)
- [UniCDR](https://github.com/cjx96/UniCDR)
- [PPA-for-CDR](https://github.com/zyx-nuaa/PPA-for-CDR)
- [CVGAE](https://github.com/YudiXiong/CVGAE)

## Environment

- Python: 3.8
- Key dependencies: `scipy==1.10.1`, `tensorboard==2.13.0`, `torch==2.4.1`, `numpy==1.22.4`
- Full dependencies: [requirements.txt](requirements.txt)
- Hardware: NVIDIA RTX 4090

## Usage

```shell
# sport_phone
python -u train_rec.py --domains sport_phone --De_rate 1 --DPM_rate 1 --IEM_rate 1

# electronic_phone
python -u train_rec.py --domains electronic_phone --De_rate 1.6 --DPM_rate 0.8 --IEM_rate 1

# sport_cloth
python -u train_rec.py --domains electronic_phone --De_rate 1.6 --DPM_rate 0.8 --IEM_rate 1

# electronic_cloth
python -u train_rec.py --domains electronic_phone --De_rate 1.6 --DPM_rate 0.8 --IEM_rate 1
```
