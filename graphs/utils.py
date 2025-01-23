import json
import networkx as nx
import pathpy as pp
import matplotlib.pyplot as plt

color_map = ['#EAAA00', '#FB5607', '#FF006E', '#8338EC', '#3A86FF', '#857437', '#FF0000']


def load_json(file_path) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def json_dict_parser(json_dict: dict) -> dict:
    d = {}
    for i, k in enumerate(json_dict.keys()):
        if k.__contains__('tar'):
            tar = int(k[k.find('tar') + 3:].replace('\'', ''))
            d[i] = {'target': tar, 'value': json_dict[k]}
        elif k.__contains__('DFS'):
            flag = 'OU-0 of Router-'
            ctx = json_dict[k][-1]
            assert ctx.find(flag) != -1, f'Unexpected context:{ctx} from key:{k}'
            tar = int(ctx[ctx.find(flag) + len(flag):ctx.find(']')])
            d[i] = {'target': tar, 'value': json_dict[k]}
    return d


def get_node_name_lst(num_node):
    num_ports = 5
    base_names = ['IU-xPORTx of Router-xPEx', 'OU-xPORTx of Router-xPEx', 'SA of Router-xPEx', 'NI-xPEx-toRouter',
                  'NI-xPEx-toPE', 'LUT of PE-xPEx', 'WeightSRAM of PE-xPEx', 'AER of PE-xPEx', 'Injector']
    short_names = ['IxPORTxRxPEx', 'OxPORTxRxPEx', 'SARxPEx', 'N2RxPEx', 'N2PxPEx', 'LUTxPEx', 'RAMxPEx',
                   'AERxPEx', 'INJ']
    node_names_tmp = []
    for name in base_names:
        if name.__contains__('xPORTx'):
            for i in range(num_ports):
                node_names_tmp.append(name.replace('xPORTx', str(i)))
    base_names.extend(node_names_tmp)
    node_names = []
    for name in base_names[2:]:
        if name.__contains__('xPEx'):
            for i in range(num_node):
                node_names.append(name.replace('xPEx', str(i)))
        else:
            assert name == 'Injector', f'Unexpected node name:{name}'
            node_names.append('Injector')

    node_names_tmp = []
    for name in short_names:
        if name.__contains__('xPORTx'):
            for i in range(num_ports):
                node_names_tmp.append(name.replace('xPORTx', str(i)))
    short_names.extend(node_names_tmp)
    node_names_short = []
    for name in short_names[2:]:
        if name.__contains__('xPEx'):
            for i in range(num_node):
                node_names_short.append(name.replace('xPEx', str(i)))
        else:
            assert name == 'INJ', f'Unexpected node name:{name}'
            node_names_short.append('INJ')
    return node_names, node_names_short


def draw_graph_nx(G: nx.DiGraph, node_color=1, font_size=8, node_size=50):
    nx.draw(G, nx.spring_layout(G), with_labels=True, node_color=color_map[node_color],
            font_size=font_size, node_size=node_size)
    plt.show()


def draw_graph_pp(G: nx.DiGraph, html_name):
    g = pp.Network(directed=True)
    for edge in G.edges:
        g.add_edge(edge[0], edge[1], weight=G.edges[edge[0], edge[1]]['activities'])
    edge_color_param = {}
    edge_width_param = {}
    node_color_param = {}
    for n in g.nodes:
        edge_width_param[n] = 1
        edge_color_param[n] = color_map[0]
        node_color_param[n] = color_map[1]
    params = {'label_color': 'black', 'width': 1200, 'height': 1200, 'node_color': node_color_param,
              'edge_color': edge_color_param, 'edge_width': edge_width_param}
    pp.visualisation.plot(g, **params)
    print(g)
    pp.visualisation.export_html(g, filename=html_name, **params)


if __name__ == '__main__':
    js_addr = 'E:\\prj\\akkanMore\\logs\\DFGJsons\\test.json'
    print(load_json(js_addr)['DFS50'])
