import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_k_peak(G,n):


    full_node_list = np.arange(0, n, 1, dtype=int)
    print("full",len(full_node_list))

    edge_of_core = []

    core_num_dict = nx.core_number(G)
    max_core_num = max(list(core_num_dict.values()))
    counter = max_core_num

    numberof_kpeak = 0
    k_list = {}
    core_number_list = {}
    while len(G.nodes) > 0:

#         print(len(list(nx.connected_components(G))))
#         print(list(nx.connected_components(G)))

        core_num_dict = nx.core_number(G)
        if len(core_num_dict.values()) != 0:
            max_core_num = max(list(core_num_dict.values()))
            k_core_graph = nx.k_core(G, k=max_core_num, core_number=core_num_dict)
        else:
            max_core_num = 0
            k_core_graph = nx.k_core(G, k=max_core_num, core_number=core_num_dict)



        for edge in k_core_graph.nodes():
            core_number_list[edge]=max_core_num

        from networkx import connected_components
        if max_core_num not in k_list.keys():
            k_list[max_core_num] = len(list(nx.connected_components(k_core_graph)))
        else:
            k_list[max_core_num] += len(list(nx.connected_components(k_core_graph)))

        nodes = k_core_graph.nodes


        numberof_kpeak  += 1

        edge_of_core.extend(list(k_core_graph.edges()))

        G.remove_nodes_from(nodes)
#         print(len(G.nodes()))

#     print(edge_of_core)


    print("unique contours numbers",len(k_list))
    print("max contour", max(k_list))
    print("total contours ", sum(k_list.values()))

    Gxxx = nx.Graph()
    Gxxx.add_edges_from(edge_of_core)
    Gxxx.add_nodes_from(full_node_list)
#     print(Gxxx.nodes)

    graphs = list(nx.connected_components(Gxxx))
    print("nuber of subgraphs",len(graphs))

    import collections
    core_number_list = dict(sorted(core_number_list.items()))
#     print(core_number_list)
    return list(core_number_list.values())



for i in range(10):
    a = 9 - i
    print("##############")
    print(a)
    path = "/media/nuri/E/datasets/QTMT/UE/adj2/"+str(a)+".txt"
    G1=nx.Graph()
    G1 = nx.read_edgelist(path, delimiter=' ',nodetype= int)
    print(len(G1.edges))
    y_pred = generate_k_peak(G1, 7771)
    print("##############")
    print(len(y_pred))
