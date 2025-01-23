from utils import *
import pathpy as pp


def add_digraph(G, dfg_str) -> nx.DiGraph:
    for i, node in enumerate(dfg_str):
        G.has_node(node) or G.add_node(node)
        if i > 0:
            G = renew_edge_attribute(G, dfg_str[i - 1], dfg_str[i])
    return G


def renew_edge_attribute(G, edge_src, edge_dst) -> nx.DiGraph:
    G.has_edge(edge_src, edge_dst) or G.add_edge(edge_src, edge_dst)
    G.edges[edge_src, edge_dst]['activities'] = G.edges[edge_src, edge_dst].get('activities', 0) + 1
    return G


if __name__ == '__main__':
    js_addr = '/home/hy24/akkanmore/logs/DFGJsons/test.json'
    G = nx.DiGraph()
    dfg = load_json(js_addr)
    for i in range(len(dfg)):
        dfg_str = dfg[f'DFS{i}']
        G = add_digraph(G, dfg_str)

    # draw_graph_nx(G)
    draw_graph_pp(G, 'test.html')
