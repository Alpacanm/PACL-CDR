import random
import os


def reduce_interactions(input_file, output_file, retain_ratio):
    """
    从训练文件中减少交互数据并保存到新文件。

    参数:
        input_file (str): 输入训练文件路径（用户ID和物品ID，\t分隔）
        output_file (str): 输出文件路径
        retain_ratio (float): 保留交互的比例（0到1之间，如0.7表示保留70%）
    """
    # 读取原始交互数据
    with open(input_file, 'r') as f:
        interactions = [line.strip() for line in f if line.strip()]  # 去除空行

    # 计算原始交互数量和目标保留数量
    total_interactions = len(interactions)
    retain_count = int(total_interactions * retain_ratio)

    # 随机采样保留的交互
    random.seed(42)  # 设置随机种子，确保结果可重复
    retained_interactions = random.sample(interactions, retain_count)

    # 保存到新文件
    with open(output_file, 'w') as f:
        f.write('\n'.join(retained_interactions))

    print(f"Processed {input_file}: retained {retain_count}/{total_interactions} interactions, "
          f"saved to {output_file}")


# 主程序
if __name__ == "__main__":
    # 输入文件路径（根据您的实际情况调整）
    input_file = "../datasets/dual-user-inter/dataset/sport_cloth/train.txt"

    # 输出文件路径
    output_dir = "./sparse_data"
    os.makedirs(output_dir, exist_ok=True)

    # 减少30%和50%交互（保留70%和50%）
    reduce_interactions(input_file, os.path.join(output_dir, "sport_train_20.txt"), retain_ratio=0.2)
    reduce_interactions(input_file, os.path.join(output_dir, "sport_train_50.txt"), retain_ratio=0.5)
    reduce_interactions(input_file, os.path.join(output_dir, "sport_train_80.txt"), retain_ratio=0.8)