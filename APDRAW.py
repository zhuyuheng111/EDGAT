import matplotlib.pyplot as plt

# 定义模型名称和对应的AUC值
models = ['DeepWalk', 'Node2Vec', 'GAT', 'EDgAT']
auc_values = [0.583, 0.485, 0.669, 0.806]

# 创建柱状图，使用相同色系
plt.figure(figsize=(8, 6))
color = '#6495ED'  # 选用相同的颜色

plt.bar(models, auc_values, color=[color] * len(models))

# 添加标题和标签
plt.title('Comparison of AUC Values for Different Models (Same Color Scheme)')
plt.xlabel('Model')
plt.ylabel('AP Value')

# 显示数值
for i, value in enumerate(auc_values):
    plt.text(i, value + 0.01, f'{value:.3f}', ha='center')

# 显示图表
plt.show()

