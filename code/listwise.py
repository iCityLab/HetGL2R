# -*- coding: utf-8 -*-
# Create Time  :  2024/4/11 14:00
# Author       :  xjr17
# File Name    :  listwise.PY
# software     :  PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ListNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ListNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.fc2(x)
        return x


criterion = nn.MSELoss()


def convert_data(data):
    new_data = []
    for key in data.keys():
        new_data.append(data[key])
    return torch.tensor(new_data)


input_dim = 256  # Example input dimension
hidden_dim = 64  # Example hidden dimension
output_dim = 1  # Example output dimension


def loss_function(input_data, target_data):
    # print(input_data.keys())
    input_data = convert_data(input_data)
    score_data = {key: [value[1]] for key, value in target_data.items()}
    # rank_data = {key: [value[0]] for key, value in target_data.items()}
    data_list = []
    for key in score_data.keys():
        data_list.append(score_data[key][0])
    # print(data_list)
    rank_data_list = sorted(data_list, reverse=True)
    # print(rank_data_list)
    rank_data = torch.tensor([[rank_data_list.index(x) + 1] for x in data_list])
    rank_data = rank_data.float()
    # target_one_hot = F.one_hot(rank_data, num_classes=num_classes)
    # print(rank_data)
    target_data = convert_data(score_data)
    # target_data1 = convert_data(rank_data)
    # Example usage:
    # Suppose you have node embeddings as input

    model2 = ListNet(input_dim, hidden_dim, output_dim)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    optimizer2.zero_grad()
    output = model2(input_data)
    # print(output,target_data)
    loss_score = criterion(output, target_data)
    # print("r2:",r2_score(target_data.detach().numpy(), output.detach().numpy()))
    # print("score:",output)
    output_list = output[:, 0].tolist()
    rank_lst = sorted(output_list, reverse=True)
    rank = torch.LongTensor([[rank_lst.index(x) + 1] for x in output_list])
    # loss_rank = 0.8*(criterion(rank,rank_data))
    loss = loss_score
    optimizer2.step()

    return loss
