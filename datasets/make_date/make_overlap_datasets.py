import argparse
import random
from pathlib import Path


TAB_SPLITS = ['train.txt', 'valid.txt', 'test.txt', 'test_original.txt']


def read_interactions(path):
    data = []
    users = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            u = int(parts[0])
            i = int(parts[1])
            tail = parts[2:] if len(parts) > 2 else []
            data.append((u, i, tail))
            users.add(u)
    return data, users


def get_all_users(domain_dir):
    """Collect user IDs across all tab-splits to correctly identify shared users."""
    all_users = set()
    for split in TAB_SPLITS:
        p = domain_dir / split
        if p.exists():
            _, users = read_interactions(p)
            all_users |= users
    return all_users


def write_interactions(path, data, user_map):
    with path.open('w', encoding='utf-8') as f:
        for u, i, tail in data:
            nu = user_map.get(u, u)
            row = [str(nu), str(i)] + tail
            f.write('\t'.join(row) + '\n')


def build_mapping(shared_users, keep_ratio, seed=42):
    rng = random.Random(seed)
    shared_users = sorted(shared_users)
    keep_n = int(round(len(shared_users) * keep_ratio))
    keep_set = set(rng.sample(shared_users, keep_n)) if keep_n < len(shared_users) else set(shared_users)
    remove_list = [u for u in shared_users if u not in keep_set]
    return keep_set, remove_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Path to datasets/dual-user-intra/dataset')
    parser.add_argument('--output_dir', required=True, help='Where to write modified datasets')
    parser.add_argument('--ratios', nargs='+', type=float, default=[1.0, 0.8, 0.6, 0.4, 0.2],
                        help='Keep ratios of overlapping users, e.g. 1.0 0.8 0.6 0.4 0.2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--move_domain', choices=['sport_phone', 'phone_sport'], default='phone_sport',
                        help='Which domain gets re-indexed for removed shared users')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sport = input_dir / 'sport_phone'
    phone = input_dir / 'phone_sport'

    # Use all splits to correctly identify shared users
    sport_users = get_all_users(sport)
    phone_users = get_all_users(phone)
    shared_users = sport_users & phone_users

    print(f'sport_phone users (all splits): {len(sport_users)}')
    print(f'phone_sport users (all splits): {len(phone_users)}')
    print(f'shared users: {len(shared_users)}')

    for ratio in args.ratios:
        ratio_name = str(int(round(ratio * 100)))
        out_root = output_dir / f'overlap_{ratio_name}' / 'dataset'
        out_sport = out_root / 'sport_phone'
        out_phone = out_root / 'phone_sport'
        out_sport.mkdir(parents=True, exist_ok=True)
        out_phone.mkdir(parents=True, exist_ok=True)

        keep_set, remove_list = build_mapping(shared_users, ratio, seed=args.seed)

        sport_map = {}
        phone_map = {}

        # Removed shared users are re-indexed in one domain so the model no
        # longer treats them as the same entity across domains. The same map
        # is applied to train, valid, and test to keep identity links broken
        # consistently across all splits.
        next_sport_id = max(sport_users) + 1
        next_phone_id = max(phone_users) + 1

        if args.move_domain == 'phone_sport':
            for u in remove_list:
                phone_map[u] = next_phone_id
                next_phone_id += 1
        else:
            for u in remove_list:
                sport_map[u] = next_sport_id
                next_sport_id += 1

        for split in TAB_SPLITS:
            src = sport / split
            if src.exists():
                data, _ = read_interactions(src)
                write_interactions(out_sport / split, data, sport_map)
            src = phone / split
            if src.exists():
                data, _ = read_interactions(src)
                write_interactions(out_phone / split, data, phone_map)

        meta = output_dir / f'overlap_{ratio_name}' / 'mapping.txt'
        with meta.open('w', encoding='utf-8') as f:
            f.write(f'ratio={ratio}\n')
            f.write(f'seed={args.seed}\n')
            f.write(f'shared_users_original={len(shared_users)}\n')
            f.write(f'shared_users_kept={len(keep_set)}\n')
            f.write(f'shared_users_removed={len(remove_list)}\n')
            f.write(f'move_domain={args.move_domain}\n')
            f.write('processed_tab_splits=' + ','.join(TAB_SPLITS) + '\n')

        print(f'Wrote overlap_{ratio_name}: kept {len(keep_set)}, removed {len(remove_list)}')


if __name__ == '__main__':
    main()
