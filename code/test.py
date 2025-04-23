import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

from sklearn.metrics import r2_score

from listwise import ListNet


def build_mapping_table(sequence,real):  # Obtain the position mapping of nodes in each sequence
    mapping_table = {}
    for node in sequence.split():
        # 获取节点真实信息（保持原有逻辑）
        real_value = real.get(node.lower(), None)
        if real_value:
            mapped_id = int(real_value[1] + 1)
        else:
            # 根据节点前缀分配ID范围
            if node.startswith('od'):
                base = type_ranges["od"][0]
            elif node.startswith('lj'):
                base = type_ranges["lj"][0]
            elif node.startswith('link'):
                base = type_ranges["link"][0]
            elif node.startswith('f'):
                base = type_ranges["f"][0]
            else:
                base = 0  # 未知类型处理

            # 生成哈希ID（示例逻辑，可自定义）
            hashed = hash(node) % 100
            mapped_id = base + hashed + 1  # +1避免0

        mapping_table[node] = mapped_id

    # value = 1
    # for i in sequence.split():
    # if i.lower() in real.keys():
    # mapping_table[i] = int(real[i.lower()][1]+1)
    # else:
    # mapping_table[i] = value
    # value += 1
    return mapping_table


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                       2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                       2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                       2)  # v_s: [batch_size x n_heads x len_k x d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def position_vector(sequence,real):  # Obtain the position encoding of nodes in each sequence
    mapping_table = build_mapping_table(sequence,real)
    # map_size = len(mapping_table)
    sequence_batch = [[mapping_table[n] for n in sequence.split()]]
    return torch.LongTensor(sequence_batch)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(map_size, d_model)
        self.od_emb = nn.Embedding(type_ranges["od"][1], d_model)
        self.lj_emb = nn.Embedding(type_ranges["lj"][1], d_model)
        self.link_emb = nn.Embedding(type_ranges["link"][1], d_model)
        self.f_emb = nn.Embedding(type_ranges["f"][1], d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        mask_od = (enc_inputs >= type_ranges["od"][0]) & (enc_inputs < type_ranges["od"][1])
        mask_lj = (enc_inputs >= type_ranges["lj"][0]) & (enc_inputs < type_ranges["lj"][1])
        mask_link = (enc_inputs >= type_ranges["link"][0]) & (enc_inputs < type_ranges["link"][1])
        mask_f = (enc_inputs >= type_ranges["f"][0]) & (enc_inputs < type_ranges["f"][1])

        # 分类型进行Embedding
        emb_od = self.od_emb(enc_inputs * mask_od.long())
        emb_lj = self.lj_emb((enc_inputs - type_ranges["lj"][0]) * mask_lj.long())
        emb_link = self.link_emb((enc_inputs - type_ranges["link"][0]) * mask_link.long())
        emb_f = self.f_emb((enc_inputs - type_ranges["f"][0]) * mask_f.long())

        # 合并Embedding结果
        enc_outputs = emb_od + emb_lj + emb_link + emb_f
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # print(enc_inputs)
        # enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  ## 编码层

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        return enc_outputs, enc_self_attns


class myModel(nn.Module):  # 第1个模型
    def __init__(self):
        super(myModel, self).__init__()
        self.transf = Transformer()  # Transformer
        self.listNet = ListNet(256, 64, 1)  # listwise
        self.link_embedding = {}  # 新增嵌入存储属性


    def convert_data(self, data):
        new_data = []
        for key in data.keys():
            new_data.append(data[key])
        return torch.tensor(new_data)

    def forward(self, enc_inputs):
        outputs, enc_self_attns = self.transf(enc_inputs)
        current_sequence = sequence.split()
        for i, node in enumerate(current_sequence):
            if node.startswith('link'):
                self.link_embedding[node] = outputs[0, i].detach().clone()
        outputs_list = outputs.tolist()
        for i in range(len(sequence.split())):
            if sequence.split()[i][:4] == 'link':
                if sequence.split()[i] in self.link_embedding.keys():
                    self.link_embedding[sequence.split()[i]] = outputs_list[0][
                        sequence.split().index(sequence.split()[i])]
                    link_reality[sequence.split()[i]] = links_reality[sequence.split()[i]]
                else:
                    self.link_embedding[sequence.split()[i]] = outputs_list[0][
                        sequence.split().index(sequence.split()[i])]
                    link_reality[sequence.split()[i]] = links_reality[sequence.split()[i]]
        # print(link_embedding)
        input_data = self.convert_data(self.link_embedding)
        # print(input_data.shape)

        output = self.listNet(input_data)

        score_data = {key: [value[1]] for key, value in link_reality.items()}
        #rank_data = {key: [value[0]] for key, value in link_reality.items()}
        data_list = []
        for key in score_data.keys():
            data_list.append(score_data[key][0])
        # print(data_list)
        rank_data_list = sorted(data_list, reverse=True)
        rank_real = [[rank_data_list.index(x) + 1] for x in data_list] #真排名
        pre_score = output.reshape(-1).tolist()
        pre_score_list = sorted(pre_score, reverse=True)
        pre_rank = [[pre_score_list.index(x) + 1] for x in pre_score]
        #计算diff
        sum = 0
        for i in range(len(rank_real)):
            #print(rank_real[i][0])
            sum += abs(rank_real[i][0]-pre_rank[i][0])
        #diff = sum/int(len(rank_real)**2/2)
        #print("diff:",diff)
        # print(rank_data_list)
        #rank_data = torch.tensor([[rank_data_list.index(x) + 1] for x in data_list])
        #rank_data = rank_data.float()
        # target_one_hot = F.one_hot(rank_data, num_classes=num_classes)
        # print(rank_data)
        target_data = self.convert_data(score_data)
        #f = 1 - r2_score(target_data.detach().numpy(), output.detach().numpy())
        #f = r2_score(target_data.detach().numpy(), output.detach().numpy())
        #print("F1:", f)
        return output, target_data



if __name__ == '__main__':
    sequences = []
    with open(r"D:\Project\PythonProject\HetGL2R\code\random_walk.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            sequences.append(line.strip())

    links_reality = {}
    rank = 1
    #with open(r"../data/real_eff.txt", "r") as file:
    with open(r"../data/real_sy.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            key = parts[0]
            links_reality[key.lower()] = [rank, float(parts[1])]
            rank += 1

    # model parameter
    d_model = 256  # Embedding Size
    #map_size = 100
    d_ff = 128  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    type_ranges = {
        "od": (0, 100),  # od类型节点ID范围 1-100
        "lj": (100, 200),  # lj类型节点ID范围 101-200
        "link": (200, 300),  # link类型ID范围 201-300
        "f": (300, 400)  # f系列类型ID范围 301-400
    }
    map_size = 401


    model1 = myModel()
    model_state_dict1 = torch.load(r'D:\Project\PythonProject\HetGL2R\trainRNtestSY\model\best_model_newbest.pth')
    # print(model_state_dict1)
    # pass
    model1.load_state_dict(model_state_dict1)
    model1.eval()

    results = {}  # 用于汇总所有序列的结果
    links_embedding = {}
    with torch.no_grad():  # 预测时不需要计算梯度
        for seq in range(len(sequences)):
            sequence = sequences[seq]
            enc_inputs = position_vector(sequence, links_reality)
            model1.link_embedding = {}
            link_reality = {}
            outputs, target_data = model1(enc_inputs)
            # print(link_embedding, link_reality)
            links_embedding.update(model1.link_embedding)

            # 更新 results 字典
            outputs_list = outputs.detach().numpy().tolist()
            unique_names = []
            seen = set()
            for name in sequence.split():
                if name not in seen:
                    seen.add(name)
                    unique_names.append(name)

            for name in unique_names:
                if name.startswith("link"):
                    results[name] = outputs_list.pop(0)

        # 保存为txt文件
        #np.savetxt(r'./test_sc.txt', outputs.detach().numpy(), fmt='%f')
    #print(target_data)
    torch.save(links_embedding, '../data/sy_node_embeddings.pt')
    print(links_embedding)
    with open(r'../code/pre_sybest.txt', 'w') as f:
        for name, output in results.items():
            f.write(f"{name} {' '.join(map(str, output))}\n")
    print("测试结果已经保存，请在eval.py中进行评估模型~")
