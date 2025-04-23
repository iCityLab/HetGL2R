# -*- coding: utf-8 -*-
# Create Time  :  2025/4/9 14:59
# Author       :  xjr17
# File Name    :  ndcg.PY
# software     :  PyCharm
# -*- coding: utf-8 -*-
import numpy as np


def read_scores(filename):
    """
    读取节点得分文件，返回一个字典，键为节点ID（小写），值为得分。
    """
    scores = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # 跳过格式错误的行
            node_id = parts[0].lower()
            try:
                score = float(parts[1])
            except ValueError:
                continue  # 跳过得分解析失败的行
            scores[node_id] = score
    return scores



def calculate_ndcg(real_scores, pred_scores):
    """
    计算@NDCG指标（基于共有节点）
    """
    # 对real_scores按照得分排序
    real_ranked = sorted(real_scores.items(), key=lambda x: x[1], reverse=True)
    real_rank = {node_id: rank for rank, (node_id, _) in enumerate(real_ranked, start=1)}
    print(real_rank)

    # 对pred_scores按照得分排序
    pred_ranked = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
    pred_rank = {node_id: rank for rank, (node_id, _) in enumerate(pred_ranked, start=1)}
    print(pred_rank)

    n = len(real_scores)
    if n == 0:
        return 0.0  # 避免除以零

    # 计算理想DCG和预测DCG
    ideal_dcg = 0.0
    pred_dcg = 0.0

    for i, (node_id, _) in enumerate(real_ranked):
        # 理想排序中的增益
        ideal_gain = 2 ** (n - real_rank[node_id]) - 1
        ideal_dcg += ideal_gain / np.log2(i + 2)  # i从0开始，所以+2

        # 预测排序中的增益
        pred_gain = 2 ** (n - pred_rank[node_id]) - 1
        pred_dcg += pred_gain / np.log2(i + 2)

    # 计算NDCG
    if ideal_dcg == 0:
        return 0.0  # 避免除以零
    ndcg = pred_dcg / ideal_dcg
    return ndcg


def main():
    real_filename = r'D:\Project\PythonProject\HetGL2R\data\real_sy.txt'
    pred_filename = r'D:\Project\PythonProject\HetGL2R\code\pre_scores_sy_k7.txt'

    # 读取文件
    real_scores = read_scores(real_filename)
    pred_scores = read_scores(pred_filename)

    # 获取共有节点
    common_nodes = real_scores.keys() & pred_scores.keys()
    if not common_nodes:
        print("没有共有的节点，无法计算指标")
        return

    # 创建共有节点数据集
    real_common = {k: real_scores[k] for k in common_nodes}
    pred_common = {k: pred_scores[k] for k in common_nodes}

    # 计算指标
    ndcg = calculate_ndcg(real_common, pred_common)

    print(f"共有节点数: {len(common_nodes)}")
    print(f"@NDCG: {ndcg:.6f}")


if __name__ == "__main__":
    main()