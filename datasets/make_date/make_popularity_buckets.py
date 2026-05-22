import argparse
from collections import Counter
from pathlib import Path


BUCKETS = [
    ("head", 0.2),
    ("middle", 0.3),
    ("tail", 0.5),
]


def read_item_popularity(train_path: Path) -> Counter:
    counter = Counter()
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            item = int(parts[1])
            counter[item] += 1
    return counter


def split_items_by_rank(item_popularity: Counter):
    ranked_items = sorted(item_popularity.items(), key=lambda x: (-x[1], x[0]))
    num_items = len(ranked_items)

    head_end = int(num_items * 0.2)
    middle_end = int(num_items * 0.5)

    if head_end == 0 and num_items > 0:
        head_end = 1
    if middle_end < head_end:
        middle_end = head_end

    head_items = ranked_items[:head_end]
    middle_items = ranked_items[head_end:middle_end]
    tail_items = ranked_items[middle_end:]

    return {
        "head": head_items,
        "middle": middle_items,
        "tail": tail_items,
    }


def write_bucket_file(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item, pop in items:
            f.write(f"{item}\t{pop}\n")


def write_summary(path: Path, buckets, total_interactions: int, total_items: int):
    with path.open("w", encoding="utf-8") as f:
        f.write("bucket\titem_count\titem_ratio\tinteraction_count\tinteraction_ratio\n")
        for bucket_name in ["head", "middle", "tail"]:
            items = buckets[bucket_name]
            item_count = len(items)
            interaction_count = sum(pop for _, pop in items)
            item_ratio = item_count / total_items if total_items else 0.0
            interaction_ratio = interaction_count / total_interactions if total_interactions else 0.0
            f.write(
                f"{bucket_name}\t{item_count}\t{item_ratio:.6f}\t{interaction_count}\t{interaction_ratio:.6f}\n"
            )


def process_domain(input_dir: Path, output_dir: Path, domain: str):
    train_path = input_dir / domain / "train.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    item_popularity = read_item_popularity(train_path)
    total_items = len(item_popularity)
    total_interactions = sum(item_popularity.values())
    buckets = split_items_by_rank(item_popularity)

    domain_out = output_dir / domain
    domain_out.mkdir(parents=True, exist_ok=True)

    write_bucket_file(domain_out / "head_items.txt", buckets["head"])
    write_bucket_file(domain_out / "middle_items.txt", buckets["middle"])
    write_bucket_file(domain_out / "tail_items.txt", buckets["tail"])
    write_summary(domain_out / "bucket_summary.txt", buckets, total_interactions, total_items)

    print(f"Processed {domain}")
    print(f"  total items: {total_items}")
    print(f"  total interactions: {total_interactions}")
    for bucket_name in ["head", "middle", "tail"]:
        items = buckets[bucket_name]
        interaction_count = sum(pop for _, pop in items)
        print(f"  {bucket_name}: {len(items)} items, {interaction_count} interactions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to datasets/dual-user-intra/dataset",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save popularity split files",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["sport_phone", "phone_sport"],
        help="Domains to process",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        process_domain(input_dir, output_dir, domain)


if __name__ == "__main__":
    main()


# Explanation:
# 1. This script uses the training set of each target domain to compute item popularity.
# 2. Popularity is defined as the number of interactions of each item in train.txt.
# 3. All items are ranked by popularity in descending order.
# 4. The ranked list is split by item count into three buckets:
#    - head: top 20%
#    - middle: next 30%
#    - tail: bottom 50%
# 5. The script only writes bucket definition files and summary statistics.
# 6. It does not modify the original dataset, model, or evaluation pipeline.
# 7. You can use the generated files to assign each test instance to a bucket according to its ground-truth item.
