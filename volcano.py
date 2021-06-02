import pandas as pd  # Data analysis
import numpy as np  # Scientific computing
import seaborn as sns  # Statistical visualization
import sys
import matplotlib.pyplot as plt  # Plotting
import matplotlib.colors as colors  # Coloring
import csv
import math
import matplotlib

matplotlib.use('TKAgg')
for k in range(0, 9):
    print(k)
    pd_csv = pd.read_csv('E:/cluster' + str(k) + '-case-DEGs.csv')
    result = pd.DataFrame()

    result['x'] = pd_csv['avg_log2FC']
    result['y'] = pd_csv['p_val_adj']

    l = []
    for i in result['y']:
        if i > 0:
            l.append(-math.log(i, 10))
        else:
            l.append(i)

    # print(result['y'])

    cut_off_pvalue = 0.05  # 统计显著性
    cut_off_logFC = 0.1  # 差异倍数值

    # 分组为up, down, stable
    result.loc[(result.x >= cut_off_logFC) & (result.y < cut_off_pvalue), 'group'] \
        = 'red'
    result.loc[(-result.x >= cut_off_logFC) & (result.y < cut_off_pvalue), 'group'] \
        = 'blue'
    result.loc[(abs(result.x) < cut_off_logFC) | (result.y >= cut_off_pvalue), 'group'] = 'dimgrey'

    # 绘制散点图
    fig = plt.figure(figsize=plt.figaspect(7 / 6))  # 确定fig比例（h/w）
    ax = fig.add_subplot()

    ax.scatter(result['x'], l, s=2, c=result['group'])
    ax.spines['right'].set_visible(False)  # 去掉右边框
    ax.spines['top'].set_visible(False)  # 去掉上边框

    # ax.set_xticks(range(-2,2,1)) #设置x轴刻度起点和步长
    # ax.set_yticks(range(-4,4,1)) #设置y轴刻度起点和步长
    # plt.show()
    fig.savefig('E:/plot' + str(k) + '.jpg', dpi=600, format='jpg')  # 保存为eps矢量图

print("OK")

# # 绘制散点图
# ax = sns.scatterplot(x="x", y="y",
#                      hue='group',
#                      hue_order=('down', 'normal', 'up'),
#                      palette=("#377EB8", "grey", "#E41A1C"),
#                      alpha=0.5,
#                      s=15,
#                      data=result)
#
# # 确定坐标轴显示范围
# xmin = -6
# xmax = 10
# ymin = 7
# ymax = 13
#
# ax.spines['right'].set_visible(False)  # 去掉右边框
# ax.spines['top'].set_visible(False)  # 去掉上边框
#
# ax.vlines(-cut_off_logFC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # 画竖直线
# ax.vlines(cut_off_logFC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # 画竖直线
# ax.hlines(-np.log10(cut_off_pvalue), xmin, xmax, color='dimgrey', linestyle='dashed', linewidth=1)  # 画竖水平线
# ax.set_xticks(range(xmin, xmax, 4))  # 设置x轴刻度
# ax.set_yticks(range(ymin, ymax, 2))  # 设置y轴刻度
# ax.set_ylabel('-log10(pvalue)', fontweight='bold')  # 设置y轴标签
# ax.set_xlabel('log2(fold change)', fontweight='bold')  # 设置x轴标签

print('1111')
