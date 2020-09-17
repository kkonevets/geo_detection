import tools
import os
from os.path import join
import numpy as np
import pandas as pd
from numpy import genfromtxt
import torch
import multiprocessing
from cymatrix import CSRMatrix
import data
from itertools import groupby
from stlmap import MapU64U32
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt


def print_ix(cond, edges, weights, nodes):
    col = edges[:, cond[0]]
    print("%d -> %d" % (col[0], col[1]))
    print("%d -> %d" % (nodes[col[0]], nodes[col[1]]))
    print(weights[cond])


def print_reverse_edges(ix, edge_index, edge_map):
    print(edge_index[:, ix])
    print(edge_index[:, edge_map[ix]])


def get_id2ix(nodes):
    id2ix = MapU64U32()
    id2ix.reserve(len(nodes))
    for ix, _id in enumerate(nodes):
        id2ix[_id] = ix
    return id2ix


def check_russian():
    nodes = tools.load_vector('../data/facebook/nodes.bin', 'L')
    id2ix = get_id2ix(nodes)
    labels = tools.to_numpy(
        tools.load_vector('../data/facebook/labels.bin', 'i'))

    ixs = []
    with open('../data/facebook/queue2.txt') as f:
        for s in f:
            try:
                ixs.append(id2ix[int(s)])
            except:
                continue

    # я не взял профили без ребер, это логично

    sub = labels[ixs]
    sub[sub > -1].shape


def get_city(extid, u64_nodes, labels, geography):
    try:
        ix = tools.binary_search(u64_nodes, extid)
    except:
        return None
    l = labels[ix]
    if l > -1:
        return geography[l]
    else:
        return None


def query(extid, u64_nodes, labels, geography, group=True):
    import requests

    res = []

    for t in ('edges_rev_count', 'edges_count'):
        is_in = t == 'edges_rev_count'
        for link_type in ('comment', 'mention', 'share'):
            response = requests.get(
                'http://graph_base.company/%s/%s?vertex=%d@facebook.com' %
                (t, link_type, extid))
            resp = ((int(s.split('@')[0]), c) for s, c in response.json())
            resp = ((i, c, get_city(i, u64_nodes, labels, geography))
                    for i, c in resp if i != extid)
            resp = [(i, c, '.'.join(ci.split('.')[1:]), is_in)
                    for i, c, ci in resp if ci]
            if resp:
                if group:
                    res += resp
                else:
                    print(t, link_type)
                    resp = sorted(resp, key=lambda x: x[2])
                    pprint(resp)
                    print()

    def __sum__(a):
        l = [int(v[0]) for v in a[1]]
        return {'w': sum(l), 'n': len(l)}

    if group:
        res = sorted(res, key=lambda t: (t[0], t[3]))

        res1 = []
        for k, g in groupby(res, lambda t: (t[0], t[3])):
            g = list(g)
            res1.append([sum(gi[1] for gi in g), g[0][2]])

        res1 = sorted(res1, key=lambda t: t[1])
        counts = list(
            map(lambda a: (a[0], __sum__(a)), groupby(res1, lambda t: t[1])))
        pprint(counts)


def get_neib_counts(extid, x, u64_nodes, geo_splited):
    conc = np.concatenate([geo_splited, geo_splited])
    row = x[np.array([tools.binary_search(u64_nodes, extid)], 'I')][0]
    ixs = np.where(row)[0]
    ziped = zip(conc[ixs], row[ixs])
    ziped = sorted(ziped, key=lambda x: x[0])
    counts = list(
        map(lambda a: (a[0], [int(v[1]) for v in a[1]]),
            groupby(ziped, lambda t: t[0])))
    return counts


def show():
    u64_nodes = tools.load_vector('../data/facebook/nodes.bin', 'L')
    edge_index = tools.load_2d_vec(
        '../data/facebook/colrow.bin',  # edgelist_double.bin
        nrows=2,
        typecode='i',
        order='F')

    # ======================================================================

    edge_map = tools.load_vector("../data/facebook/reverse_edge_map.bin", 'I')

    print_reverse_edges(2345, edge_index, edge_map)

    # ======================================================================

    # id2ix = get_id2ix(u64_nodes)
    x = CSRMatrix('../data/facebook/train/neib_ftrs_data.bin',
                  '../data/facebook/train/neib_ftrs_indices.bin',
                  '../data/facebook/train/neib_ftrs_indptr.bin',
                  multiprocessing.cpu_count())

    labels = tools.to_numpy(
        tools.load_vector("../data/facebook/labels.bin", 'i'))

    geo_splited = np.genfromtxt('../data/facebook/geography_splited.csv', str)
    geography = np.genfromtxt('../data/facebook/geography.csv', dtype=str)

    extid = 100001046296473
    query(extid, u64_nodes, labels, geography, group=True)
    print()
    pprint(get_neib_counts(extid, x, u64_nodes, geo_splited))

    # ======================================================================

    weights = CSRMatrix('../data/facebook/edge_ftrs_data.bin',
                        '../data/facebook/edge_ftrs_indices.bin',
                        '../data/facebook/edge_ftrs_indptr.bin',
                        multiprocessing.cpu_count())

    # cond = np.all(weights, axis=0)
    # eall = edge_index[:, cond]
    # wall = weights[:, cond]

    cond = np.array([23144], dtype='I')
    print_ix(cond, edge_index, weights, u64_nodes)

    ix1 = tools.binary_search(u64_nodes, 100009116587703)
    ix2 = tools.binary_search(u64_nodes, 100001989967439)

    cond = np.where(np.logical_and(edge_index[0] == ix1,
                                   edge_index[1] == ix2))[0].astype('I')
    print_ix(cond, edge_index, weights, u64_nodes)

    cond = np.where(np.logical_and(edge_index[0] == ix2,
                                   edge_index[1] == ix1))[0].astype('I')
    print_ix(cond, edge_index, weights, u64_nodes)

    # 100003289482076  # co_Ukraine.r1_KievObl.ci_Kiev
    100009116587703
    100001989967439


def plot_stat(base_dir, ylim):
    def _show(data, ylim=None):
        sub = data[:50]
        plt.bar(sub[:, 0], height=sub[:, 1])
        plt.xticks(sub[:, 0], sub[:, 0])
        if ylim:
            plt.ylim(0, ylim)
        plt.show()

    def _precents(data, l):
        nums = []
        for i in l:
            n = round(100 * data[i, 1] / data[:, 1].sum(), 2)
            nums.append(n)
        return nums

    fcomment = join(base_dir, "in_degree_comment.bin")
    fshare = join(base_dir, "in_degree_share.bin")
    comment = genfromtxt(fcomment, delimiter=',', dtype=np.int64)
    share = genfromtxt(fshare, delimiter=',', dtype=np.int64)

    _show(comment, ylim)
    _show(share, ylim)

    l = range(10)
    print('comment')
    print(_precents(comment, l))
    print('share')
    print(_precents(share, l))


def show_city_coincide(base_dir, xlabels):
    counts_eq = []
    counts_neq = []
    for i in range(len(xlabels)):
        eqs = tools.load_2d_vec(join(base_dir, "counts_eq_%d.bin" % i),
                                typecode='I',
                                order='F')
        counts_eq.append(eqs)
        neqs = tools.load_2d_vec(join(base_dir, "counts_neq_%d.bin" % i),
                                 typecode='I',
                                 order='F')
        counts_neq.append(neqs)

    def plot_counts_pair(counts, xlabel, title):
        fname = os.path.join(base_dir, "%s.pdf" % xlabel)

        plt.clf()
        plt.bar(counts[0][0], counts[0][1], lw=1, alpha=0.5, label="equal")
        plt.bar(counts[1][0], counts[1][1], lw=1, alpha=0.5, label="not equal")

        plt.xlim(0, 15)
        plt.legend(fontsize='small', loc='upper right')
        plt.xlabel(xlabel)
        plt.ylabel("counts")
        plt.title(title)
        plt.savefig(fname, dpi=900)

    for i, eq_neq in enumerate(zip(counts_eq, counts_neq)):
        plot_counts_pair(eq_neq, xlabels[i], base_dir.split('/')[-1])


if __name__ == "__main__":
    # show_city_coincide('../data/facebook', ['comment', 'mention', 'share'])
    # show_city_coincide('../data/twitter', ['comment', 'share'])
    # plot_stat(base_dir='../data/twitter', ylim=900000)
    plot_stat(base_dir='../data/facebook', ylim=4000000)