import torch
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
# 使用 CPU
device = torch.device('cpu')

# 读取数据
data = pd.read_excel('suoyoujiyin.xlsx')
# data = pd.read_excel('gene_cirvsun_cir_deletepvalue_DE.xlsx')


# 创建 NetworkX 图
G = nx.Graph()

# 添加节点，节点特征包括 log2fc, readcount_cirrhosis, readcount_non_cirrhosis
for idx, row in data.iterrows():
    G.add_node(row['gene_name'], features=[row['log2FoldChange'], row['Readcount_cir'], row['Readcount_un_cir']])

# 定义边的阈值（log2fc差异小于 1.5 则添加边）
threshold = 1.5
for i in range(len(data)):
    for j in range(i + 1, len(data)):
        if abs(data.loc[i, 'log2FoldChange'] - data.loc[j, 'log2FoldChange']) < threshold:
            G.add_edge(data.loc[i, 'gene_name'], data.loc[j, 'gene_name'])

# 转换为 PyTorch Geometric 格式
data_geometric = from_networkx(G)

# 添加节点特征并移动到 CPU
data_geometric.x = torch.tensor([G.nodes[n]['features'] for n in G.nodes], dtype=torch.float).to(device)

# 分割数据集用于链路预测 (训练集、验证集、测试集)
data_geometric = train_test_split_edges(data_geometric)

# 生成负样本
data_geometric.train_neg_edge_index = negative_sampling(
    edge_index=data_geometric.train_pos_edge_index,  # 使用正样本边来生成负样本
    num_nodes=data_geometric.num_nodes,
    num_neg_samples=data_geometric.train_pos_edge_index.size(1)
).to(device)


# 定义带有 GAT 的链路预测模型 (Encoder-Decoder 结构)
class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super(GATLinkPredictor, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=heads, concat=True)  # 第一层有多个 heads
        self.conv2 = GATConv(32 * heads, out_channels, heads=1, concat=False)  # 最后一层无 concat

        # 定义解码器
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, 64),  # 输入为两个节点的嵌入
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)  # 输出为边的预测值
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        edge_features = torch.cat([z_i, z_j], dim=-1)  # 合并节点嵌入
        return self.decoder(edge_features).squeeze(-1)  # 输出为预测值


# 初始化 GAT 模型和优化器并将模型移动到 CPU
model = GATLinkPredictor(data_geometric.x.shape[1], 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data_geometric.x, data_geometric.train_pos_edge_index.to(device))
    pos_pred = model.decode(z, data_geometric.train_pos_edge_index.to(device),
                            data_geometric.train_pos_edge_index.to(device))
    neg_pred = model.decode(z, data_geometric.train_pos_edge_index.to(device),
                            data_geometric.train_neg_edge_index.to(device))
    loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred)) + \
           F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss.backward()
    optimizer.step()
    return loss.item()


# 评估函数，计算 AUC-ROC
def test():
    model.eval()
    with torch.no_grad():
        # 使用测试集边索引来生成嵌入
        z = model(data_geometric.x, data_geometric.test_pos_edge_index.to(device))

        # 正样本和负样本的预测值
        pos_pred = model.decode(z, data_geometric.test_pos_edge_index.to(device),
                                data_geometric.test_pos_edge_index.to(device))
        neg_pred = model.decode(z, data_geometric.test_neg_edge_index.to(device),
                                data_geometric.test_neg_edge_index.to(device))

        # 合并预测值
        pred = torch.cat([pos_pred, neg_pred])

        # 构造标签
        label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        # 计算 AUC-ROC 和 AP
        auc_roc = roc_auc_score(label.cpu().numpy(), pred.cpu().numpy())
        ap_score = average_precision_score(label.cpu().numpy(), pred.cpu().numpy())

        return auc_roc, ap_score


# 训练模型
for epoch in range(300):
    loss = train()
    if (epoch + 1) % 10 == 0:
        auc, ap = test()
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, AUC-ROC: {auc:.4f}, AP: {ap:.4f}')
