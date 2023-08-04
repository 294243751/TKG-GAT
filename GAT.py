import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import os
import dgl
from torch.autograd import Variable
import sys
import gc

import pickle
from collections import defaultdict
from dgl.nn.pytorch import edge_softmax, GATConv



class GATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1) # 归一化每一条入边的注意力系数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h':h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z # 每个节点的特征
        self.g.apply_edges(self.edge_attention) # 为每一条边获得其注意力系数
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim , num_heads=1, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge


    def forward(self, h):
        head_out = [attn_head(h) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim , out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g , in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim*num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)

        return h


def load_dicts():
    entity_dict_file = 'entity2id.txt'
    relation_dict_file = 'relation2id.txt'
    data_dir = "icews14"
    print('-----Loading entity dict-----')
    entity_df = pd.read_table(os.path.join(data_dir, entity_dict_file), header=None)
    entity_dict = dict(zip(entity_df[0], entity_df[1]))
    n_entity = len(entity_dict)
    entities = list(entity_dict.values())
    print('#entity: {}'.format(n_entity))
    print('-----Loading relation dict-----')
    relation_df = pd.read_table(os.path.join(data_dir, relation_dict_file), header=None)
    relation_dict = dict(zip(relation_df[0], relation_df[1]))
    n_relation = len(relation_dict)
    if rev_set>0: n_relation *= 2
    print('#relation: {}'.format(n_relation))
    return entity_dict, relation_dict, n_entity ,n_relation

def load_triples(entity_dict,relation_dict):
    training_file = 'train.txt'
    validation_file = 'valid.txt'
    test_file = 'test.txt'
    start_date = '2014-01-01'
    start_sec = time.mktime(time.strptime(start_date, '%Y-%m-%d'))
    # n_time = 365
    gran = 3
    data_dir = "icews14"
    train_times = set()
    valid_times = set()
    test_times = set()
    training_triples = []
    validation_triples = []
    test_triples = []
    training_facts = []
    validation_facts = []
    test_facts = []

    print('-----Loading training triples-----')
    training_df = pd.read_table(os.path.join(data_dir, training_file), header=None)
    training_df = np.array(training_df).tolist()
    for triple in training_df:
        end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (gran*24 * 60 * 60))
        training_triples.append([entity_dict[triple[0]],entity_dict[triple[2]],relation_dict[triple[1]],day]) # src dst rel time
        train_times.add(day)
        training_facts.append(
            [entity_dict[triple[0]], entity_dict[triple[2]], relation_dict[triple[1]], triple[3], 0])
    train_times = list(train_times)  # 加
    train_times.sort()  # 加
    n_training_triple = len(training_triples)
    print('#training triple: {}'.format(n_training_triple))

    print('-----Loading validation triples-----')
    validation_df = pd.read_table(os.path.join(data_dir, validation_file), header=None)
    validation_df = np.array(validation_df).tolist()
    for triple in validation_df:
        end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (gran * 24 * 60 * 60))
        valid_times.add(day)  # 加
        validation_triples.append(
            [entity_dict[triple[0]], entity_dict[triple[2]], relation_dict[triple[1]], day])
        validation_facts.append(
            [entity_dict[triple[0]], entity_dict[triple[2]], relation_dict[triple[1]], triple[3], 0])
    n_validation_triple = len(validation_triples)
    print('#validation triple: {}'.format(n_validation_triple))
    valid_times = list(valid_times)  # 加
    valid_times.sort()  # 加

    print('-----Loading test triples------')
    test_df = pd.read_table(os.path.join(data_dir, test_file), header=None)
    test_df = np.array(test_df).tolist()
    for triple in test_df:
        end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (gran * 24 * 60 * 60))
        test_times.add(day)  # 加
        test_triples.append(
            [entity_dict[triple[0]], entity_dict[triple[2]], relation_dict[triple[1]], day])
        test_facts.append(
            [entity_dict[triple[0]], entity_dict[triple[2]], relation_dict[triple[1]], triple[3], 0])
    n_test_triple = len(test_triples)
    test_times = list(test_times)  # 加
    test_times.sort()  # 加
    print('#test triple: {}'.format(n_test_triple))

    return training_triples, train_times, training_facts, validation_triples, valid_times, validation_facts, test_triples, test_times, test_facts

def load_graph(training_triples, train_times):
    graph_dict = {}
    neg_graph_dict = {}
    for tim in train_times:
        data = get_data_with_t(training_triples, tim)
        graph_dict[tim], neg_graph_dict[tim] = get_graph(data, tim)

    return graph_dict, neg_graph_dict

def get_data_with_t(data, tim):
    data = np.array(data)
    x = data[np.where(data[:, 3] == tim)].copy()
    x = np.delete(x, 3, 1)  # drops 3rd column
    return x

def get_graph(data, tim):
    src, dst, rel = data.transpose()
    # load_atise(tim)
    # positive sample with graph
    src, dst, rel = np.append(src, dst), np.append(dst, src), np.append(rel, rel)
    uniq_v, edges = np.unique((src, dst), return_inverse=True)  # the only dict id, the dict id's id
    temp_nodes = torch.torch.randn(len(uniq_v), 500)
    u, v = edges.reshape(2, -1)
    g_pos = dgl.graph((u, v))
    g_pos.edata['h_i'] = torch.tensor(src.reshape(-1, 1))       # dict id
    g_pos.edata['t_i'] = torch.tensor(dst.reshape(-1, 1))       # dict id
    g_pos.edata['src'] = torch.tensor(u.reshape(-1, 1))         # graph id
    g_pos.edata['dst'] = torch.tensor(v.reshape(-1, 1))         # graph id
    g_pos.ndata['Id'] = torch.tensor(uniq_v.reshape(-1, 1))     #
    # for i, j in zip(uniq_v, np.arange(int(len(uniq_v)))):
    #     temp_nodes[j] = entity_embedding[i]
    # temp_nodes = torch.tensor(temp_nodes)
    # temp_nodes = torch.tensor(temp_nodes, requires_grad=True)
    g_pos.ndata['Ag'] = temp_nodes         # the feature of nodes
    # g_pos.ndata['Ag'] = torch.randn(len(uniq_v), 500)
    g_pos.edata['r_i'] = torch.tensor(rel.reshape(-1, 1))

    # create the negative sample
    neg_src = torch.tensor(src)
    neg_src = neg_src.repeat_interleave(4)
    neg_dst = torch.tensor(dst)
    neg_dst = neg_dst.repeat_interleave(4)
    neg_rel = torch.tensor(rel)
    neg_rel = neg_rel.repeat_interleave(4)
    neg_src = neg_src.numpy()
    neg_dst = neg_dst.numpy()
    neg_rel = neg_rel.numpy()

    # temp = neg_src[:int(len(neg_src)/2)]
    # temp = np.random.permutation(temp)
    # temp = torch.randint(n_entity, [int(len(neg_src)/2])
    neg_src[:int(len(neg_src)/2)] = torch.randint(0, n_entity, (1, int(len(neg_src)/2)))

    # temp = neg_dst[int(len(neg_dst) / 2):]
    # temp = np.random.permutation(temp)
    # neg_dst[int(len(neg_dst) / 2):] = temp
    neg_dst[int(len(neg_dst) / 2):] = torch.randint(0, n_entity, (1, int(len(neg_dst)/2)))

    # neg_rel = np.random.permutation(neg_rel)

    # negative sample with graph
    uniq_v, edges = np.unique((neg_src, neg_dst), return_inverse=True)  # the only dict id, the dict id's id
    temp_nodes = torch.torch.randn(len(uniq_v), 500)
    u, v = edges.reshape(2, -1)
    g_neg = dgl.graph((u, v))
    g_neg.edata['h_i'] = torch.tensor(neg_src.reshape(-1, 1))  # dict id
    g_neg.edata['t_i'] = torch.tensor(neg_dst.reshape(-1, 1))  # dict id
    g_neg.edata['src'] = torch.tensor(u.reshape(-1, 1))  # graph id
    g_neg.edata['dst'] = torch.tensor(v.reshape(-1, 1))  # graph id
    g_neg.ndata['Id'] = torch.tensor(uniq_v.reshape(-1, 1))  #
    # for i, j in zip(uniq_v, np.arange(int(len(uniq_v)))):
    #     temp_nodes[j] = entity_embedding[i]  # the feature of nodes
    # temp_nodes = torch.tensor(temp_nodes, requires_grad=True)
    g_neg.ndata['Ag'] = temp_nodes  # the feature of nodes
    g_neg.edata['r_i'] = torch.tensor(neg_rel.reshape(-1, 1))

    return g_pos, g_neg

def get_data(data1, data2, tim):
    data1 = np.array(data1)
    data2 = np.array(data2)
    x1 = []
    x2 = []
    # x = data[np.where(data[:, 3] == tim)].copy()
    for i in np.arange(len(data1)):
        if data1[i, 3] == tim:
            x1.append(data1[i])
            x2.append(data2[i])
    return x1, x2

def log_rank_loss(y_pos, y_neg, temp=0):
    M = y_pos.size(0)
    N = y_neg.size(0)
    y_pos = 120 - y_pos
    y_neg = 120 - y_neg
    C = int(N / M)
    y_neg = y_neg.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * y_neg)
    loss_pos = torch.sum(F.softplus(-1 * y_pos))
    loss_neg = torch.sum(p * F.softplus(y_neg))
    loss = (loss_pos + loss_neg) / 2 / M
    loss.cuda()
    return loss

def get_score(features, g):
    global  tim
    if type(g) == np.ndarray:
        h_i, t_i, r_i, d_i = g[:, 0].astype(np.int64), g[:, 1].astype(np.int64), g[:, 2].astype(np.int64), g[:, 3].astype(np.float32)   # evaluation
    else:
        h_i, t_i, r_i, d_i = g.edata['h_i'].numpy().astype(np.int64), g.edata['t_i'].numpy().astype(np.int64), g.edata['r_i'].numpy().astype(np.int64), torch.tensor(tim)   # the score of agg
        h_i, delete = h_i.reshape(2, -1)
        t_i, delete = t_i.reshape(2, -1)
        r_i, delete = r_i.reshape(2, -1)
    # emb_E = torch.nn.Embedding(self.kg.n_entity, embedding_dim, padding_idx=0)
    # emb_E_var = torch.nn.Embedding(self.kg.n_entity, embedding_dim, padding_idx=0)
    # emb_R = torch.nn.Embedding(self.kg.n_relation, embedding_dim, padding_idx=0)
    # emb_R_var = torch.nn.Embedding(self.kg.n_relation, embedding_dim, padding_idx=0)
    # emb_TE = torch.nn.Embedding(self.kg.n_entity, embedding_dim, padding_idx=0)
    # alpha_E = torch.nn.Embedding(self.kg.n_entity, 1, padding_idx=0)
    # beta_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
    # omega_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
    # emb_TR = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
    # alpha_R = torch.nn.Embedding(self.kg.n_relation, 1, padding_idx=0)
    # beta_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
    # omega_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)

    device = torch.device("cuda")
    checkpoint = torch.load('ATiSE.pkl')
    for key, value in checkpoint.items():
        if key == 'emb_E_var.weight':
            emb_E_var = value
        elif key == 'emb_E.weight':
            emb_E = value
        elif key =='emb_R.weight':
            emb_R = value
        elif key =='emb_R_var.weight':
            emb_R_var = value
        elif key =='emb_TE.weight':
            emb_TE = value
        elif key =='alpha_E.weight':
            alpha_E = value
        elif key =='beta_E.weight':
            beta_E = value
        elif key =='omega_E.weight':
            omega_E = value
        elif key =='emb_TR.weight':
            emb_TR = value
        elif key =='alpha_R.weight':
            alpha_R = value
        elif key =='beta_R.weight':
            beta_R = value
        elif key =='omega_R.weight':
            omega_R = value

    # evaluation
    if type(g) == np.ndarray:
        h_i = Variable(torch.from_numpy(h_i).cuda())  # 把数组转化成张量，且二者共享内存
        t_i = Variable(torch.from_numpy(t_i).cuda())
        r_i = Variable(torch.from_numpy(r_i).cuda())
        d_i = Variable(torch.from_numpy(d_i).cuda())
    else:
        h_i = Variable(torch.from_numpy(h_i).cuda())  # 把数组转化成张量，且二者共享内存
        t_i = Variable(torch.from_numpy(t_i).cuda())
        r_i = Variable(torch.from_numpy(r_i).cuda())
        d_i = d_i.cuda()

    # h_i = Variable(torch.from_numpy(h_i))
    # t_i = Variable(torch.from_numpy(t_i))
    # r_i = Variable(torch.from_numpy(r_i))
    # d_i = Variable(torch.from_numpy(d_i))

    pi = 3.14159265358979323846
    h_mean = emb_E[h_i].view(-1, embedding_dim) + \
             d_i.view(-1, 1) * alpha_E[h_i].view(-1, 1) * emb_TE[h_i].view(-1, embedding_dim) \
             + beta_E[h_i].view(-1, embedding_dim) * torch.sin(
        2 * pi * omega_E[h_i].view(-1, embedding_dim) * d_i.view(-1, 1))

    t_mean = emb_E[t_i].view(-1, embedding_dim) + \
             d_i.view(-1, 1) * alpha_E[t_i].view(-1, 1) * emb_TE[t_i].view(-1, embedding_dim) \
             + beta_E[t_i].view(-1, embedding_dim) * torch.sin(
        2 * pi * omega_E[t_i].view(-1, embedding_dim) * d_i.view(-1, 1))

    r_mean = emb_R[r_i].view(-1, embedding_dim) + \
             d_i.view(-1, 1) * alpha_R[r_i].view(-1, 1) * emb_TR[r_i].view(-1, embedding_dim) \
             + beta_R[r_i].view(-1, embedding_dim) * torch.sin(
        2 * pi * omega_R[r_i].view(-1, embedding_dim) * d_i.view(-1, 1))

    list1 = []
    list2 = []
    if type(g) == np.ndarray:
        for i, j in zip(h_i, t_i):
            list1.append(entity_embedding[i])
            list2.append(entity_embedding[j])
    else:
        for i, j in zip(g.edata['src'], np.arange(int(len(g.edata['src'])/2))):
            i = i.type(torch.long)
            # temp = g[0].ndata['Ag'][i].ravel()
            temp = features[i].ravel()
            # i = i.type(torch.long)
            # temp = features[i].ravel()
            list1.append(temp)
        for i, j in zip(g.edata['dst'], np.arange(int(len(g.edata['dst'])/2))):
            i = i.type(torch.long)
            # temp = g[0].ndata['Ag'][i].ravel()
            temp = features[i].ravel()
            # temp = features[i].ravel()
            list2.append(temp)

    h_mean = h_mean + torch.stack(list1, 0).to(device)
    t_mean = t_mean + torch.stack(list2, 0).to(device)

    h_var = emb_E_var[h_i].view(-1, embedding_dim)
    t_var = emb_E_var[t_i].view(-1, embedding_dim)
    r_var = emb_R_var[r_i].view(-1, embedding_dim)

    out1 = torch.sum((h_var + t_var) / r_var, 1) + torch.sum(((r_mean - h_mean + t_mean) ** 2) / r_var,
                                                             1) - embedding_dim
    out2 = torch.sum(r_var / (h_var + t_var), 1) + torch.sum(((h_mean - t_mean - r_mean) ** 2) / (h_var + t_var),
                                                             1) - embedding_dim
    out = (out1 + out2) / 4
    return out

def load_atise(tim=0):
    global entity_embedding
    checkpoint = torch.load('ATiSE.pkl')
    pi = 3.14159265358979323846
    d_i = tim
    h_emb = []
    for key, value in checkpoint.items():
        if key == 'emb_E_var.weight':
            emb_E_var = value
        elif key == 'emb_E.weight':
            emb_E = value
        elif key == 'emb_R.weight':
            emb_R = value
        elif key == 'emb_R_var.weight':
            emb_R_var = value
        elif key == 'emb_TE.weight':
            emb_TE = value
        elif key == 'alpha_E.weight':
            alpha_E = value
        elif key == 'beta_E.weight':
            beta_E = value
        elif key == 'omega_E.weight':
            omega_E = value
        elif key == 'emb_TR.weight':
            emb_TR = value
        elif key == 'alpha_R.weight':
            alpha_R = value
        elif key == 'beta_R.weight':
            beta_R = value
        elif key == 'omega_R.weight':
            omega_R = value

    for h_i in np.arange(n_entity):
        h = emb_E[h_i].view(-1, embedding_dim) + \
             d_i * alpha_E[h_i].view(-1, 1) * emb_TE[h_i].view(-1, embedding_dim) \
             + beta_E[h_i].view(-1, embedding_dim) * torch.sin(
        2 * pi * omega_E[h_i].view(-1, embedding_dim) * d_i)
        entity_embedding[h_i] = h

def evaluation(epoch):
    global patience
    global mrr_std
    global tim
    global rank
    # global rank1
    # global rank2
    global all_val_rank
    global all_test_rank
    global val_best_rank
    global test_best_rank

    rank = []
    path = r"C:\Users\Administrator\Desktop\ATISE-GAT - AGG - ATISE\result"
    temp_rank1 = rank_left(validation_triples, validation_facts, timedisc = 0, rev_set=rev_set)
    temp_rank2 = rank_right(validation_triples, validation_facts, timedisc=0, rev_set=rev_set)
    temp_rank = temp_rank1 + temp_rank2
    m_rank = mean_rank(temp_rank)
    mean_rr = mrr(temp_rank)
    hit_1 = hit_N(temp_rank, 1)
    hit_3 = hit_N(temp_rank, 3)
    hit_5 = hit_N(temp_rank, 5)
    hit_10 = hit_N(temp_rank, 10)
    print('validation results:')
    print('Mean Rank: {:.0f}'.format(m_rank))
    print('Mean RR: {:.4f}'.format(mean_rr))
    print('Hit@1: {:.4f}'.format(hit_1))
    print('Hit@3: {:.4f}'.format(hit_3))
    print('Hit@5: {:.4f}'.format(hit_5))
    print('Hit@10: {:.4f}'.format(hit_10))
    if mean_rr < mrr_std and patience < 3:
        patience = patience + 1

    if mean_rr >= mrr_std:
        val_best_rank = temp_rank[:]
        mrr_std = mean_rr
        patience = 0
        temp_rank1 = rank_left(test_triples, test_facts, timedisc=0, rev_set=rev_set)
        temp_rank2 = rank_right(test_triples, test_facts, timedisc=0, rev_set=rev_set)
        temp_rank = temp_rank1 + temp_rank2
        test_best_rank = temp_rank[:]

    if epoch == 119 or patience == 3:
        all_val_rank.append(val_best_rank)
        all_test_rank.append(test_best_rank)

        m_rank = mean_rank(test_best_rank)
        mean_rr = mrr(test_best_rank)
        hit_1 = hit_N(test_best_rank, 1)
        hit_3 = hit_N(test_best_rank, 3)
        hit_5 = hit_N(test_best_rank, 5)
        hit_10 = hit_N(test_best_rank, 10)
        print('test results:')
        print('Mean Rank: {:.0f}'.format(m_rank))
        print('Mean RR: {:.4f}'.format(mean_rr))
        print('Hit@1: {:.4f}'.format(hit_1))
        print('Hit@3: {:.4f}'.format(hit_3))
        print('Hit@5: {:.4f}'.format(hit_5))
        print('Hit@10: {:.4f}'.format(hit_10))
        if tim != len(train_times) - 1:
            return 1
        else:
            all_val_rank = [b for a in all_val_rank for b in a]
            m_rank = mean_rank(all_val_rank)
            mean_rr = mrr(all_val_rank)
            hit_1 = hit_N(all_val_rank, 1)
            hit_3 = hit_N(all_val_rank, 3)
            hit_5 = hit_N(all_val_rank, 5)
            hit_10 = hit_N(all_val_rank, 10)
            print('all validation results:')
            # print('Rank bumber: {:.0f}'.format(len(rank)))
            print('Mean Rank: {:.0f}'.format(m_rank))
            print('Mean RR: {:.4f}'.format(mean_rr))
            print('Hit@1: {:.4f}'.format(hit_1))
            print('Hit@3: {:.4f}'.format(hit_3))
            print('Hit@5: {:.4f}'.format(hit_5))
            print('Hit@10: {:.4f}'.format(hit_10))
            f = open(os.path.join(path, 'result{0}_tim{1}.txt'.format(epoch, tim)), 'w')
            f.write('Mean Rank: {:.0f}\n'.format(m_rank))
            f.write('Mean RR: {:.4f}\n'.format(mean_rr))
            f.write('Hit@1: {:.4f}\n'.format(hit_1))
            f.write('Hit@3: {:.4f}\n'.format(hit_3))
            f.write('Hit@5: {:.4f}\n'.format(hit_5))
            f.write('Hit@10: {:.4f}\n'.format(hit_10))
            # f.write('Rank bumber: {:.0f}'.format(len(rank)))
            f.close()

            all_test_rank = [b for a in all_test_rank for b in a]
            m_rank = mean_rank(all_test_rank)
            mean_rr = mrr(all_test_rank)
            hit_1 = hit_N(all_test_rank, 1)
            hit_3 = hit_N(all_test_rank, 3)
            hit_5 = hit_N(all_test_rank, 5)
            hit_10 = hit_N(all_test_rank, 10)
            print('all test result:')
            # print('Rank bumber: {:.0f}'.format(len(rank)))
            print('Mean Rank: {:.0f}'.format(m_rank))
            print('Mean RR: {:.4f}'.format(mean_rr))
            print('Hit@1: {:.4f}'.format(hit_1))
            print('Hit@3: {:.4f}'.format(hit_3))
            print('Hit@5: {:.4f}'.format(hit_5))
            print('Hit@10: {:.4f}'.format(hit_10))
            f = open(os.path.join(path, 'test_result{0}_tim{1}.txt'.format(epoch, tim)), 'w')
            f.write('Mean Rank: {:.0f}\n'.format(m_rank))
            f.write('Mean RR: {:.4f}\n'.format(mean_rr))
            f.write('Hit@1: {:.4f}\n'.format(hit_1))
            f.write('Hit@3: {:.4f}\n'.format(hit_3))
            f.write('Hit@5: {:.4f}\n'.format(hit_5))
            f.write('Hit@10: {:.4f}\n'.format(hit_10))
            # f.write('Rank bumber: {:.0f}'.format(len(rank)))
            f.close()
    return 0


def rank_left(X, facts,timedisc=0, rev_set=0):
    rank = []
    with torch.no_grad():
        if timedisc:
            for triple, fact in zip(X, facts):
                X_i = np.ones([n_entity, 4])  # 生成n行4列矩阵
                i_score = torch.zeros(n_entity)
                if self.gpu:
                    i_score = i_score.cuda()
                for time_index in [triple[3], triple[4]]:
                    for i in range(0,n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = time_index
                    i_score = i_score + get_score(entity_embedding, X_i).view(-1)
                    if rev_set > 0:
                        X_rev = np.ones([n_entity, 4])
                        for i in range(0, n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2] + n_relation // 2
                            X_rev[i, 3] = time_index
                        i_score = i_score + get_score(entity_embedding, X_rev).view(-1)
                filter_out = to_skip_final['lhs'][(fact[1], fact[2], fact[3], fact[4])]
                target = i_score[int(triple[0])].clone()
                i_score[filter_out] = 1e6
                rank_triple = torch.sum((i_score <= target).float()).cpu().item() + 1
                rank.append(rank_triple)

        else:
            for triple, fact in zip(X, facts):
                X_i = np.ones([n_entity, 4])
                for i in range(0, n_entity):
                    X_i[i, 0] = i
                    X_i[i, 1] = triple[1]
                    X_i[i, 2] = triple[2]
                    X_i[i, 3] = triple[3]
                i_score = get_score(entity_embedding, X_i)
                if rev_set > 0:
                    X_rev = np.ones([n_entity, 4])
                    for i in range(0, n_entity):
                        X_rev[i, 0] = triple[1]
                        X_rev[i, 1] = i
                        X_rev[i, 2] = triple[2] + n_relation // 2
                        X_rev[i, 3] = triple[3]
                    i_score = i_score + get_score(entity_embedding, X_rev).view(-1)
                i_score = i_score.cuda()
                filter_out = to_skip_final['lhs'][(fact[1], fact[2], fact[3], fact[4])]
                target = i_score[int(triple[0])].clone()
                i_score = i_score.cpu().numpy()
                for k in filter_out:
                    i_score[int(k)] = 1e6
                i_score = torch.from_numpy(i_score).cuda()
                rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1
                rank.append(rank_triple)
    return rank

def rank_right(X, facts,timedisc=0, rev_set=0):
    rank = []
    with torch.no_grad():
        if timedisc:
            for triple, fact in zip(X, facts):
                X_i = np.ones([n_entity, 4])
                i_score = torch.zeros(n_entity)

                i_score = i_score.cuda()
                for time_index in [triple[3], triple[4]]:
                    for i in range(0, n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = time_index
                    i_score = i_score + get_score(entity_embedding, X_i).view(-1)
                    if rev_set > 0:
                        X_rev = np.ones([n_entity, 4])
                        for i in range(0, n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2] + n_relation // 2
                            X_rev[i, 3] = time_index
                        i_score = i_score + get_score(entity_embedding, X_rev).view(-1)

                filter_out = to_skip_final['rhs'][(fact[0], fact[2], fact[3], fact[4])]
                target = i_score[int(triple[1])].clone()
                i_score[filter_out] = 1e6
                rank_triple = torch.sum((i_score <= target).float()).cpu().item() + 1

                rank.append(rank_triple)
        else:
            for triple, fact in zip(X, facts):
                X_i = np.ones([n_entity, 4])
                for i in range(0, n_entity):
                    X_i[i, 0] = triple[0]
                    X_i[i, 1] = i
                    X_i[i, 2] = triple[2]
                    X_i[i, 3] = triple[3]
                i_score = get_score(entity_embedding, X_i)
                if rev_set > 0:
                    X_rev = np.ones([n_entity, 4])
                    for i in range(0, n_entity):
                        X_rev[i, 0] = i
                        X_rev[i, 1] = triple[0]
                        X_rev[i, 2] = triple[2] + n_relation // 2
                        X_rev[i, 3] = triple[3]
                    i_score = i_score + get_score(entity_embedding, X_rev).view(-1)
                i_score = i_score.cuda()
                filter_out = to_skip_final['rhs'][(fact[0], fact[2], fact[3], fact[4])]
                target = i_score[int(triple[1])].clone()
                i_score = i_score.cpu().numpy()
                for k in filter_out:
                    i_score[int(k)] = 1e6
                i_score = torch.from_numpy(i_score).cuda()
                rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1

                rank.append(rank_triple)

    return rank

def load_filters():
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    facts_pool = [training_facts, validation_facts, test_facts]
    for facts in facts_pool:
        for fact in facts:
            to_skip['lhs'][(fact[1], fact[2], fact[3], fact[4])].add(fact[0])  # left prediction
            to_skip['rhs'][(fact[0], fact[2], fact[3], fact[4])].add(fact[1])  # right prediction

    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))
    # print("data preprocess completed")

def mean_rank(rank):
    m_r = 0
    N = len(rank)
    for i in rank:
        m_r = m_r + i / N

    return m_r

def mrr(rank):
    mrr = 0
    N = len(rank)
    for i in rank:
        mrr = mrr + 1 / i / N

    return mrr

def hit_N(rank, N):
    hit = 0
    for i in rank:
        if i <= N:
            hit = hit + 1

    hit = hit / len(rank)

    return hit


rev_set = 0
embedding_dim = 500
all_val_rank = []
all_test_rank = []
rank1 = []
rank2 = []
rank3 = []
rank4 = []

to_skip_final = {'lhs': {}, 'rhs': {}}
entity_dict, relation_dict, n_entity, n_relation = load_dicts()
entity_embedding = torch.randn(n_entity, 500)
load_atise(tim=0)
all_training_triples, train_times, all_training_facts, all_validation_triples, valid_times, all_validation_facts, all_test_triples, test_times, all_test_facts = load_triples(entity_dict, relation_dict)
pos_g, neg_g = load_graph(all_training_triples, train_times)

for tim in train_times:
    patience = 0
    mrr_std = 0
    next = 0
    training_triples, training_facts = get_data(all_training_triples, all_training_facts, tim)
    validation_triples, validation_facts = get_data(all_validation_triples, all_validation_facts, tim)
    test_triples, test_facts = get_data(all_test_triples, all_test_facts, tim)
    load_filters()
    entity_embedding = entity_embedding.detach()
    for i, j in zip(pos_g[tim].ndata['Id'], np.arange(int(len(pos_g[tim].ndata['Id'])))):
        pos_g[tim].ndata['Ag'][j] = entity_embedding[i.type(torch.long)]
    features = pos_g[tim].ndata['Ag']
    net = GAT(pos_g[tim], features.size()[1], hidden_dim=16, out_dim=500, num_heads=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # train
    for epoch in range(120):
        logits = net(features)
        # indx = pos_g[0].ndata['Id'][0].type(torch.long)
        # print(entity_embedding[indx][0][0])
        for i, j in zip(pos_g[tim].ndata['Id'], logits):
            entity_embedding[i.type(torch.long)] = j    # refresh the embedding
            for q, k in zip(neg_g[tim].ndata['Id'], np.arange(int(len(neg_g[tim].ndata['Id'])))) :
                if i == q:
                    neg_g[tim].ndata['Ag'][k] = j
        # neg_logits = neg_g.ndata['Ag'].data
        pos_score = get_score(logits, pos_g[tim])
        # neg_score = get_score(neg_logits.data, neg_g)
        neg_score = get_score(neg_g[tim].ndata['Ag'].detach(), neg_g[tim])
        loss = log_rank_loss(pos_score, neg_score, temp = 0.5)
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Tim:{:d} | Epoch:{:d} | Loss {:.5f}".format(tim, epoch, loss.item()))
        if epoch >= 70:
            next = evaluation(epoch)
        if next:
            break
    print("tim {0} done".format(tim))
