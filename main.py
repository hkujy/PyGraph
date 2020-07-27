from numpy.core.numeric import identity, ufunc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class LinkClass(object):
    """
        ini class for the graph 
    """
    def __init__(self, _id, _tail, _head):
        self.id = _id
        self.tail = _tail
        self.head = _head
        self.weight = 0
          

def plot_Dial_net(_links):
    """
        plot the network flow for the dial network
    """

    G = nx.DiGraph()
    num = 9
    nodes = list(range(num))  # [0,1,2,3,4,5]
    # 将节点添加到网络中
    G.add_nodes_from(nodes)  # 从列表中加点
    edges = []  # 存放所有的边，构成无向图（去掉最后一个结点，构成一个环）
    labels={}
    for idx, node in enumerate(G.nodes()):
        labels[node] = str(idx+1)

    for l in links:
        tail = l.tail
        head = l.head
        if l.weight > 0:
            edges.append((tail,head))
    # edges.append((1,2))
    # edges.append((0,3))
    # edges.append((1,4))
    # edges.append((2,5))
    # edges.append((3,4))
    # edges.append((4,5))
    # edges.append((3,6))
    # edges.append((4,7))
    # edges.append((5,8))
    # edges.append((6,7))
    # edges.append((7,8))
    G.add_edges_from(edges)

    coordinates = [[1, 3], [2, 3], [3, 3], [1, 2], [2, 2], [3, 2],[1, 1], [2, 1], [3, 1]]

    for l in _links:
        tail = l.tail
        head = l.head
        # if l.weight == 0:
        #     G[tail][head]['weight'] = 0
        #     G[tail][head]['color'] = 'r'
        if l.weight > 0:
            G[tail][head]['weight'] = l.weight
            G[tail][head]['color'] = 'r'

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
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, npos, arrows=True, arrowstyle = '-|>', edge_color=colors, width=weights)
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
    plt.axis('off')
    plt.show()
if __name__ == "__main__":
    # execute only if run as a script
    links = []
    links.append(LinkClass(0,0,1))
    links.append(LinkClass(1,1,2))
    links.append(LinkClass(2,0,3))
    links.append(LinkClass(3,1,4))
    links.append(LinkClass(4,2,5))
    links.append(LinkClass(5,3,4))
    links.append(LinkClass(6,4,5))
    links.append(LinkClass(7,3,6))
    links.append(LinkClass(8,4,7))
    links.append(LinkClass(9,5,8))
    links.append(LinkClass(10,6,7))
    links.append(LinkClass(11,7,8))
    for l in links:
        l.weight = 2 
    links[-1].weight= 0
    links[-5].weight= 0

    plot_Dial_net(links)
 



