import random
import os


def inject_noise_to_implicit_dataset(train_file, output_file, user_num, item_num, noise_level, seed=42):
    """
    对隐式反馈数据集（txt格式）注入噪声，生成新训练集。

    参数:
        train_file (str): 输入训练集文件路径（如"source_train.txt"）
        output_file (str): 输出噪声训练集文件路径（如"source_train_noisy_0.05.txt"）
        user_num (int): 用户总数
        item_num (int): 物品总数
        noise_level (float): 噪声比例（如0.05表示5%）
        seed (int): 随机种子，默认42

    返回:
        None（直接生成新文件）
    """
    # 设置随机种子
    random.seed(seed)

    # 读取原始训练集
    interactions = []
    with open(train_file, 'r') as f:
        for line in f:
            user_id, item_id = map(int, line.strip().split())
            interactions.append((user_id, item_id, 1))

    # 计算正交互总数
    num_pos = len(interactions)

    # 计算翻转数量（移除和添加的交互数）
    flip_count = int(num_pos * noise_level)
    flip_half = flip_count // 2  # 移除和添加各一半

    # 随机选择要移除的正交互
    remove_indices = random.sample(range(num_pos), flip_half)
    remove_set = set((interactions[i][0], interactions[i][1]) for i in remove_indices)

    # 移除选中的正交互
    noisy_interactions = [(u, i, r) for u, i, r in interactions if (u, i) not in remove_set]

    # 生成非交互对并添加伪交互
    current_pairs = set((u, i) for u, i, _ in interactions)  # 原始正交互
    non_interactions = []
    # 只从活跃用户和物品中采样非交互对，避免全遍历
    active_users = set(u for u, _, _ in interactions)
    active_items = set(i for _, i, _ in interactions)
    for u in active_users:
        for i in active_items:
            if (u, i) not in current_pairs:
                non_interactions.append((u, i))

    # 随机选择要添加的伪交互
    add_indices = random.sample(range(len(non_interactions)), flip_half)
    new_interactions = [(non_interactions[i][0], non_interactions[i][1], 1) for i in add_indices]

    # 合并新交互
    noisy_interactions.extend(new_interactions)

    # 保存到新文件，使用制表符（\t）分隔
    with open(output_file, 'w') as f:
        for u, i, _ in noisy_interactions:
            f.write(f"{u}\t{i}\n")  # 修改为制表符分隔

    print(
        f"Generated noisy dataset: {output_file}, original interactions: {num_pos}, new interactions: {len(noisy_interactions)}")


# 示例使用
if __name__ == "__main__":
    # 参数设置（需根据您的实际数据集调整）
    source_train_file = "../datasets/dual-user-inter/dataset/sport_cloth/train.txt"
    target_train_file = "../datasets/dual-user-inter/dataset/cloth_sport/train.txt"
    source_user_num = 9928  # 替换为实际用户总数
    source_item_num = 30796  # 替换为实际物品总数
    target_user_num = 9928
    target_item_num = 39008
    noise_levels = [0.05]  # 噪声水平：5% 和 10%

    # 处理源域和目标域
    for noise_level in noise_levels:
        # 源域
        source_output_file = f"source_train_noisy_{noise_level}.txt"
        inject_noise_to_implicit_dataset(source_train_file, source_output_file, source_user_num, source_item_num,
                                         noise_level)

        # 目标域
        target_output_file = f"target_train_noisy_{noise_level}.txt"
        inject_noise_to_implicit_dataset(target_train_file, target_output_file, target_user_num, target_item_num,
                                         noise_level)