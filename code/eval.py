# -*- coding: utf-8 -*-
# Create Time  :  2024/5/18 13:20
# Author       :  xjr17
# File Name    :  eval.PY
# software     :  PyCharm
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

result_list = []
result = []
link = []
pre_score = {}
pre_rank = {}
real_score = {}
real_rank = {}

with open(r'D:\Project\PythonProject\HetGL2R\code\pre_scores_sy_k9.txt', 'r', encoding='utf-8') as file1:
    for line in file1:
        name1,score1 = line.strip().split()
        pre_score[name1] = score1
#print(pre_score)

with open(r'D:\Project\PythonProject\HetGL2R\data\real_sy.txt', encoding='utf-8') as file2:
    for line in file2:
        name2, score2 = line.strip().split()
        real_score[name2] = score2
        if name2 not in pre_score.keys():
            pre_score[name2] = 0
#print(real_score)

pr = sorted([float(x) for x in pre_score.values()], reverse=True)
rr = sorted([float(x) for x in real_score.values()], reverse=True)
#print(pr)

for name3 in pre_score.keys():
    #if float(pre_score[name3]) ==0 and float(real_score[name3]) == 0:#3
        #pre_rank[name3] = None
    #else:
    pre_rank[name3] = pr.index(float(pre_score[name3])) + 1
    real_rank[name3] = rr.index(float(real_score[name3])) + 1

for name4 in pre_score.keys():
    if pre_rank[name4] is not None:
        result_list.append((name4,float(pre_rank[name4]),float(real_rank[name4])))
    else:
        result_list.append((name4, pre_rank[name4], float(real_rank[name4])))
    result.append((name4,float(pre_score[name4]),float(real_score[name4])))


P = 0
M = 0
geshu = len(result_list)
for n in range(len(result_list)):
    #if result_list[n][1] is not None :

    if result_list[n][2] < 40:
        M += 1
        if result_list[n][1] >= 40:
           # result_list[n][1] = 40
            P += abs(40 - result_list[n][2])
        else:
            P += abs(result_list[n][1]- result_list[n][2])
diff = P / (M**2/2) #int(len(result_list)**2/2)   #RN为测试集模型，指标用后面这个
print(diff)

re_score,pr_score=[],[]

#acc, fz, fm = 0, 0, 0
#max_real = float(real_score[max(real_score,key=lambda k:float(real_score[k]))])
#min_real = float(real_score[min(real_score,key=lambda k:float(real_score[k]))])
for n in range(len(result)):
    if result_list[n][1] <= geshu:
        pr_score.append(result[n][1])
        re_score.append(result[n][2])
#    fz += abs(result[n][1] - result[n][2])
#    #fm += abs((result[n][2]-min_real)/(max_real-min_real))
#    fm += max_real-min_real
#acc = 1 - fz / fm
#print("acc:",acc)

# 归一化到0到1的范围
scaler = MinMaxScaler()
real_scores_normalized = scaler.fit_transform(np.array(re_score).reshape(-1, 1)).flatten()
pred_scores_normalized = scaler.transform(np.array(pr_score).reshape(-1, 1)).flatten()

# 计算归一化后的Wasserstein距离
wasserstein_dist = wasserstein_distance(real_scores_normalized, pred_scores_normalized)
print("emd:", wasserstein_dist)