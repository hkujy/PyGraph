from numpy.core.numeric import identity
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.DiGraph()
num = 9
nodes = list(range(num))  # [0,1,2,3,4,5]
# 将节点添加到网络中
G.add_nodes_from(nodes)  # 从列表中加点
edges = []  # 存放所有的边，构成无向图（去掉最后一个结点，构成一个环）

labels={}
for idx, node in enumerate(G.nodes()):
    labels[node] = str(idx+1)

edges.append((0,1))
# edges.append((1,0))
edges.append((1,2))
edges.append((0,3))
edges.append((1,4))
edges.append((2,5))
edges.append((3,4))
edges.append((4,5))
edges.append((3,6))
edges.append((4,7))
edges.append((5,8))
edges.append((6,7))
edges.append((7,8))
# for idx in range(num - 1):
#     edges.append((idx, idx + 1))
# edges.append((num - 1, 0))
G.add_edges_from(edges)

coordinates = [[1, 3], [2, 3], [3, 3], [1, 2], [2, 2], [3, 2],[1, 1], [2, 1], [3, 1]]

G[0][1]['weight'] = 2
# G[1][0]['weight'] = 2
G[1][2]['weight'] = 2
G[0][3]['weight'] = 2
G[1][4]['weight'] = 2
G[2][5]['weight'] = 2
G[3][4]['weight'] = 2
G[4][5]['weight'] = 2
G[3][6]['weight'] = 2
G[4][7]['weight'] = 2
G[5][8]['weight'] = 2
G[6][7]['weight'] = 2
G[7][8]['weight'] = 2

edge_labels = dict([((u,v), round(d['weight'], 1))
             for u,v,d in G.edges(data=True)])


# for u,v,d in G.edges(data=True):
#     d['label']=str(d['weight'])

vnode= np.array(coordinates)
npos = dict(zip(nodes, vnode))  # 获取节点与坐标之间的映射关系，用字典表示
 # 若显示多个图，可将所有节点放入该列表中
# pos = {} 
# pos.update(npos)
nlabels = dict(zip(nodes, nodes))  # 标志字典，构建节点与标识点之间的关系

x_max,y_max = vnode.max(axis=0)  # 获取每一列最大值
x_min, y_min = vnode.min(axis=0)  # 获取每一列最小值
x_num = (x_max - x_min) / 10
y_num = (y_max - y_min) / 10
# print(x_max, y_max, x_min, y_min)
plt.xlim(x_min - x_num, x_max + x_num)
plt.ylim(y_min - y_num, y_max + y_num)
fig=plt.gcf()
ax=plt.gca()

# pos=nx.circular_layout(G, center=(0, 0))
nx.draw_networkx_labels(G, npos, labels=labels, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
nx.draw_networkx_nodes(G, npos,node_color='skyblue',cmap=plt.get_cmap('jet'))
# nx.draw_networkx_edges(G, npos, arrows=True, arrowstyle = '-|>',  connectionstyle='arc3, rad = 0.15')
nx.draw_networkx_edges(G, npos, arrows=True, arrowstyle = '-|>')
# D = nx.drawing.nx_agraph.to_agraph(G)

# pos_attrs = {}
# for node, coords in nx.spring_layout(G).items():
    # pos_attrs[node] = (coords[0], coords[1] + 0.08)

nx.draw_networkx_edge_labels(G,npos, edge_labels=edge_labels,font_size=10)
# nx.draw_networkx_edge_labels(G,npos, font_size=10)
# position=nx.drawing.nx_agraph.graphviz_layout(G,prog='twopi',args='')

for key, spine in ax.spines.items():
    # 'left', 'right', 'bottom', 'top'
    # if key == 'right' or key == 'top':
    spine.set_visible(False)

# nx.draw(G, npos, with_labels=True,
#         connectionstyle='arc3, rad = 0.15',
#         node_size=800)

plt.show()