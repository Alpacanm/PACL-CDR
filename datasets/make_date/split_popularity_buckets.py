"""
按 item popularity 将数据集切分为 head / middle / tail 三组，
每组独立生成 train.txt、test.txt、valid.txt。

文件格式（只处理 tab 分隔格式）：
  train.txt / test.txt：每行  user_id<TAB>item_id<TAB>1

分桶规则（popularity 只从 train.txt 统计，不用 test，避免泄露）：
  head   = 训练交互数排名前 20% 的 item
  middle = 排名 20%~50%
  tail   = 排名后 50%

输出结构：
  <output_dir>/<domain>/
    head/
      train.txt   ← 只含 head item 的训练交互
      test.txt    ← 只含 head item 的测试交互
      valid.txt   ← 与 test.txt 相同（原数据集无单独 valid）
    middle/
      train.txt / test.txt / valid.txt
    tail/
      train.txt / test.txt / valid.txt
    bucket_summary.txt
"""

import argparse
from collections import Counter
from pathlib import Path


# ──────────────────────────────────────────────────────
# 1. 统计 popularity
# ──────────────────────────────────────────────────────

def count_item_popularity(train_txt: Path) -> Counter:
    """从 train.txt 统计每个 item 的出现次数。"""
    counter = Counter()
    with train_txt.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                counter[int(parts[1])] += 1
    return counter


# ──────────────────────────────────────────────────────
# 2. 建桶
# ──────────────────────────────────────────────────────

def build_buckets(pop: Counter):
    """
    按 item 数量比例切分，返回：
      head_set   前 20% item（高频）
      middle_set 20%~50%
      tail_set   后 50%（低频）
    """
    ranked = sorted(pop.items(), key=lambda x: (-x[1], x[0]))
    n = len(ranked)
    head_end   = max(1, int(n * 0.20))
    middle_end = max(head_end, int(n * 0.50))

    head_set   = {item for item, _ in ranked[:head_end]}
    middle_set = {item for item, _ in ranked[head_end:middle_end]}
    tail_set   = {item for item, _ in ranked[middle_end:]}
    return head_set, middle_set, tail_set


# ──────────────────────────────────────────────────────
# 3. 过滤单个文件
# ──────────────────────────────────────────────────────

def filter_tab_file(src: Path, dst: Path, keep_items: set) -> int:
    """
    只保留 item 在 keep_items 中的行写入 dst，格式不变。
    返回保留的行数。
    """
    kept = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and int(parts[1]) in keep_items:
                kept.append(line.rstrip("\n"))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(kept))
        if kept:
            f.write("\n")
    return len(kept)


# ──────────────────────────────────────────────────────
# 4. 写 bucket_summary.txt
# ──────────────────────────────────────────────────────

def write_summary(path: Path, pop: Counter,
                  head_set: set, middle_set: set, tail_set: set,
                  train_counts: dict, test_counts: dict):
    total_items = len(pop)
    total_inter = sum(pop.values())
    with path.open("w", encoding="utf-8") as f:
        f.write("bucket\titem_count\titem_ratio\t"
                "train_inter\ttrain_inter_ratio\t"
                "test_inter\t"
                "pop_min\tpop_max\tpop_mean\n")
        for label, item_set in [("head", head_set),
                                 ("middle", middle_set),
                                 ("tail", tail_set)]:
            pops = [pop[i] for i in item_set] if item_set else [0]
            f.write(
                f"{label}\t"
                f"{len(item_set)}\t{len(item_set)/total_items:.4f}\t"
                f"{train_counts[label]}\t{train_counts[label]/total_inter:.4f}\t"
                f"{test_counts[label]}\t"
                f"{min(pops)}\t{max(pops)}\t{sum(pops)/len(pops):.2f}\n"
            )


# ──────────────────────────────────────────────────────
# 5. 处理单个 domain
# ──────────────────────────────────────────────────────

def process_domain(src_root: Path, out_root: Path, domain: str, verbose: bool):
    domain_src = src_root / domain
    train_src  = domain_src / "train.txt"
    test_src   = domain_src / "test.txt"

    if not train_src.exists():
        raise FileNotFoundError(f"找不到 train.txt：{train_src}")
    if not test_src.exists():
        raise FileNotFoundError(f"找不到 test.txt：{test_src}")

    # 统计 popularity、建桶
    pop = count_item_popularity(train_src)
    head_set, middle_set, tail_set = build_buckets(pop)
    bucket_sets = {"head": head_set, "middle": middle_set, "tail": tail_set}

    if verbose:
        n = len(pop)
        print(f"\n[{domain}]  item 总数={n}  "
              f"head={len(head_set)}  middle={len(middle_set)}  tail={len(tail_set)}")

    train_counts, test_counts = {}, {}
    domain_out = out_root / domain

    for label, keep in bucket_sets.items():
        bucket_dir = domain_out / label

        # 生成 train.txt
        n_train = filter_tab_file(train_src, bucket_dir / "train.txt", keep)
        train_counts[label] = n_train

        # 生成 test.txt
        n_test = filter_tab_file(test_src, bucket_dir / "test.txt", keep)
        test_counts[label] = n_test

        # valid.txt = test.txt（原数据集无独立 valid 集）
        import shutil
        shutil.copy2(bucket_dir / "test.txt", bucket_dir / "valid.txt")

        if verbose:
            print(f"  {label:<8} train={n_train:>6}  test={n_test:>6}")

    # 写汇总
    summary_path = domain_out / "bucket_summary.txt"
    write_summary(summary_path, pop,
                  head_set, middle_set, tail_set,
                  train_counts, test_counts)
    if verbose:
        print(f"  summary → {summary_path}")


# ──────────────────────────────────────────────────────
# 6. 入口
# ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="按 item popularity 切分 train/test/valid，生成 head/middle/tail 三组"
    )
    parser.add_argument("--input_dir",  required=True,
                        help="原始数据集根目录，例如 datasets/dual-user-intra/dataset")
    parser.add_argument("--output_dir", required=True,
                        help="输出根目录，例如 datasets/popularity_split")
    parser.add_argument("--domains", nargs="+",
                        default=[
                            "sport_phone", "phone_sport",
                            "electronic_phone", "phone_electronic",
                            "cloth_electronic", "electronic_cloth",
                            "cloth_sport",      "sport_cloth",
                        ],
                        help="要处理的 domain 列表，默认全部 8 个")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    src = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        process_domain(src, out, domain, verbose=not args.quiet)

    print("\n完成。输出目录：", out.resolve())


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════
# 执行方式
# ══════════════════════════════════════════════════════════════════
#
# 【处理全部 8 个 domain】
#   python split_popularity_buckets.py \
#       --input_dir  datasets/dual-user-intra/dataset \
#       --output_dir datasets/popularity_split
#
# 【只处理指定 domain】
#   python split_popularity_buckets.py \
#       --input_dir  datasets/dual-user-intra/dataset \
#       --output_dir datasets/popularity_split \
#       --domains sport_phone phone_sport
#
# ══════════════════════════════════════════════════════════════════
# 输出目录结构（以 sport_phone 为例）
# ══════════════════════════════════════════════════════════════════
#
#   datasets/popularity_split/
#     sport_phone/
#       bucket_summary.txt        ← 各桶 item 数 / 交互数统计
#       head/
#         train.txt               ← 只含 head item 的训练交互
#         test.txt                ← 只含 head item 的测试交互
#         valid.txt               ← 同 test.txt
#       middle/
#         train.txt / test.txt / valid.txt
#       tail/
#         train.txt / test.txt / valid.txt
#     phone_sport/
#       ...
#
# ══════════════════════════════════════════════════════════════════
# bucket_summary.txt 字段说明
# ══════════════════════════════════════════════════════════════════
#
#   bucket           — head / middle / tail
#   item_count       — 该桶的唯一 item 数
#   item_ratio       — 占全部 item 的比例
#   train_inter      — 训练集中该桶 item 的总交互行数
#   train_inter_ratio— 占全部训练交互的比例
#   test_inter       — 测试集中该桶 item 的总交互行数
#   pop_min/max/mean — 该桶内 item 的训练交互数统计
#
# ══════════════════════════════════════════════════════════════════
# 分桶规则
# ══════════════════════════════════════════════════════════════════
#
#   popularity 只基于 train.txt，不包含 test.txt（避免数据泄露）。
#   head   = 交互数排名前 20% 的 item（高频，一般占绝大多数交互）
#   middle = 排名 20%~50%
#   tail   = 排名后 50%（低频，item 数最多但交互数最少）
