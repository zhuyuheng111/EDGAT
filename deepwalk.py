import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, average_precision_score
# Step 1: 加载数据并进行清洗
file_path = 'suoyoujiyin.xlsx'  # 替换为你的文件路径
# file_path = 'gene_cirvsun_cir_deletepvalue_DE.xlsx'  # 替换为你的文件路径
genes_df = pd.read_excel(file_path)

# 清理空值或NaN的基因名称
genes_df = genes_df.dropna(subset=['gene_name'])

# Step 2: 构建图
G = nx.Graph()

# 将基因添加为图的节点
for index, row in genes_df.iterrows():
    G.add_node(row['gene_name'], Readcount_cir=row['Readcount_cir'], Readcount_un_cir=row['Readcount_un_cir'])

# 假设基于log2FoldChange的相关性来构建边
threshold = 1.5  # 设置阈值
for i, gene1 in genes_df.iterrows():
    for j, gene2 in genes_df.iterrows():
        if i != j and abs(gene1['log2FoldChange'] - gene2['log2FoldChange']) > threshold:
            G.add_edge(gene1['gene_name'], gene2['gene_name'])

# Step 3: 生成随机游走序列，DeepWalk的主要部分
def generate_random_walks(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, node, walk_length))
    return walks

def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if len(neighbors) > 0:
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
        else:
            break
    return walk

# 参数设置
num_walks = 10  # 每个节点的随机游走次数
walk_length = 5  # 随机游走的长度

# 生成随机游走序列
walks = generate_random_walks(G, num_walks, walk_length)

# Step 4: 使用Word2Vec进行节点嵌入（即DeepWalk的核心思想）
model = Word2Vec(walks, vector_size=64, window=5, min_count=0, sg=1, workers=4)

# Step 5: 获得节点的嵌入向量
def get_node_embeddings(model, graph):
    embeddings = {}
    for node in graph.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node]
    return embeddings

node_embeddings = get_node_embeddings(model, G)

# Step 6: 链路预测任务
# 创建正样本（现有边）和负样本（不存在的边）
positive_edges = list(G.edges())
non_edges = list(nx.non_edges(G))

# 为了平衡正负样本，随机选择与正样本数相同的负样本
positive_edges = np.array(positive_edges)
negative_edges = np.array(non_edges)
negative_edges = negative_edges[np.random.choice(len(negative_edges), len(positive_edges), replace=True)]

# 将正负样本分成训练集和测试集
pos_train, pos_test = train_test_split(positive_edges, test_size=0.3, random_state=42)
neg_train, neg_test = train_test_split(negative_edges, test_size=0.3, random_state=42)

# Step 7: 训练链路预测模型
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def link_prediction_score(embedding1, embedding2):
    return np.dot(embedding1, embedding2)

# 训练集和测试集上的链路预测
def evaluate_link_prediction(embeddings, edges, true_labels):
    predictions = []
    for edge in edges:
        if edge[0] in embeddings and edge[1] in embeddings:
            pred_score = link_prediction_score(embeddings[edge[0]], embeddings[edge[1]])
            predictions.append(sigmoid(pred_score))  # 将得分映射到0-1之间
        else:
            predictions.append(0)  # 如果没有嵌入，则预测为0
    auc = roc_auc_score(true_labels, predictions)  # 计算AUC
    ap = average_precision_score(true_labels, predictions)  # 计算AP
    return auc, ap

# 训练集上的正样本标签和负样本标签
train_edges = np.concatenate([pos_train, neg_train])
train_labels = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])

# 测试集上的正样本标签和负样本标签
test_edges = np.concatenate([pos_test, neg_test])
test_labels = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])

# Step 8: 评估链路预测性能
train_auc, train_ap = evaluate_link_prediction(node_embeddings, train_edges, train_labels)
test_auc, test_ap = evaluate_link_prediction(node_embeddings, test_edges, test_labels)

print(f'Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}')
print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
