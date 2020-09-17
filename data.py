import tools
import math
import os
import csv
import pandas as pd
import numpy as np
from os.path import join, exists
from tools import load_vector, load_2d_vec, to_numpy, natural_keys
from sklearn.model_selection import train_test_split
from cymatrix import CSRMatrix
import array
from ctypes import cdll
from glob import glob
from tqdm import tqdm
import multiprocessing
import torch
from torch_geometric.data import Data
from sampler_geometric import NeighborSampler


class MultilabelTarget:
    def __init__(self, y, geomap):
        self.y = y
        self.geomap = geomap
        self.shape = (len(y), geomap.shape[1])

    def __getitem__(self, selection):
        return self.geomap[self.y[selection]]


class Config:
    def __init__(self,
                 base_dir,
                 dev_mode,
                 directed=False,
                 multilabel=True,
                 degree_threshold=30,
                 degree_city_threshold=5,
                 city_freq_threshold=1000,
                 precision_min1=0.95,
                 precision_min2=0.90,
                 test_size=100000,
                 max_train_size=4 * 10**6,
                 nsample=None,
                 test_every=500,
                 steps=4000,
                 continue_train=False):
        self.multilabel = multilabel
        self.steps = steps
        self.continue_train = continue_train
        self.test_every = test_every
        self.directed = directed
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.precision_min1 = precision_min1
        self.precision_min2 = precision_min2
        self.degree_threshold = degree_threshold
        self.degree_city_threshold = degree_city_threshold
        self.city_freq_threshold = city_freq_threshold
        self.dev_mode = dev_mode
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.base_dir = base_dir
        if directed:
            self.save_dir = self.join('train/')
        else:
            self.save_dir = self.join('train/' if dev_mode else 'predict/')
        self.extids_fname = join(self.save_dir, 'extids.csv')
        self.probs_fname = join(self.save_dir, 'probs.csv')
        self.cixs_fname = join(self.save_dir, 'city_ixs.csv')
        if exists(self.extids_fname):
            os.remove(self.extids_fname)
        if exists(self.probs_fname):
            os.remove(self.probs_fname)
        if exists(self.cixs_fname):
            os.remove(self.cixs_fname)

        self.part_dirs = glob(self.save_dir + 'part*')
        self.part_dirs.sort(key=natural_keys)

        self.cpu_count = multiprocessing.cpu_count()  # 8
        self.node_csr_data_path = self.save_dir + "neib_ftrs_data.bin"
        self.node_csr_indices_path = self.save_dir + "neib_ftrs_indices.bin"
        self.node_csr_indptr_path = self.save_dir + "neib_ftrs_indptr.bin"
        if directed:
            self.edge_csr_data_path = self.join("edge_ftrs_data.bin")
            self.edge_csr_indices_path = self.join("edge_ftrs_indices.bin")
            self.edge_csr_indptr_path = self.join("edge_ftrs_indptr.bin")
            self.reverse_edge_map_path = self.join("reverse_edge_map.bin")

        self.postfix = '_mlabel' if multilabel else ''
        self.pt_data_path = self.join('data%s.pt' % self.postfix)
        self.stat_path = self.join('stat.pt')

        self.nsample = nsample
        if nsample is None:
            fnsample = self.join('nsample.conf')
            if exists(fnsample):
                s = open(fnsample).read()
                self.nsample = list(map(int, s[1:-1].split(',')))

    def join(self, s):
        return os.path.join(self.base_dir, s)


def train_test_partition(config, labels, nodes=None, anew=False):
    """ Sample active users with cities, filter nonfrequent cities, return train/test split """
    ftest = config.join("test_index.bin")
    ftrain = config.join("train_index.bin")
    fpredict = config.join("predict_index.bin")

    # sample only users having at least 30 friends with cities
    if anew:
        degrees_cities = to_numpy(
            load_vector(config.join("degrees_with_cities.bin"), 'f'))
        is_active_cities = degrees_cities > config.degree_city_threshold

        if config.directed:
            is_node_active = is_active_cities
        else:
            degrees = to_numpy(load_vector(config.join("degrees.bin"), 'I'))
            is_active = np.logical_and(degrees > config.degree_threshold,
                                       is_active_cities)
            is_node = np.zeros(len(labels), dtype=np.bool)
            is_node[nodes] = True
            is_node_active = is_node & is_active

        is_city = to_numpy(load_vector(config.join("is_city.bin"), 'B'))
        train_test = labels[(is_city == 1) & is_node_active]
        # predict = labels[~is_city & is_node_active]
        predict = labels[is_node_active]

        city_freq = train_test.value_counts()
        is_valid = np.zeros(city_freq.index.max() + 1, dtype=np.bool)
        # exclude very infrequent
        valid_cities = city_freq[city_freq > config.city_freq_threshold].index
        is_valid[valid_cities] = True
        train_test = train_test[is_valid[train_test]]
        train_size = min(config.max_train_size,
                         len(train_test) - config.test_size)
        train, test = train_test_split(train_test,
                                       stratify=train_test,
                                       test_size=config.test_size,
                                       train_size=train_size,
                                       random_state=11)
        with open(ftest, 'wb') as f1, \
             open(ftrain,'wb') as f2, \
             open(fpredict,'wb') as f3:
            test = array.array('I', test.index)
            test.tofile(f1)
            train = array.array('I', train.index)
            train.tofile(f2)
            predict = array.array('I', predict.index)
            predict.tofile(f3)
    else:
        test = load_vector(ftest, 'I')
        train = load_vector(ftrain, 'I')
        predict = load_vector(fpredict, 'I')

    return train, test, predict


def gen_sub_paths(path, exclude_continent=True):
    splited = path.split('.')
    if exclude_continent:
        splited = splited[1:]  # exclude continent totaly
    for i, s in enumerate(splited[1:], 1):  # exclude countries
        assert s.startswith(('r1_', 'r2_', 'r3_', 'ci_'))
        yield '.'.join(splited[:i + 1])


def get_target(config, labels, train_test):
    def _split(paths, fsave):
        """
        Split geography paths by levels, then sort by name and return the list
        e.g.
        co_Russia.r1_CentralRussia.r2_Belgorod.ci_Belgorod
        becomes
        [co_Russia.r1_CentralRussia,
        co_Russia.r1_CentralRussia.r2_Belgorod,
        co_Russia.r1_CentralRussia.r2_Belgorod.ci_Belgorod]
        """
        res = list(set([sub for c in paths for sub in gen_sub_paths(c)]))
        res = list(sorted(res))
        pd.Series(res).to_csv(fsave, index=False, header=False)
        return res

    def build_geomap(geo, geo_splited, fsave):
        """
        Build a map {"geography path index" : "list of subpath indexes"}
        e.g.
        "co_Kazakhstan.ci_Astana" -> [co_Kazakhstan, co_Kazakhstan.ci_Astana]
        becomes
        56 -> [11,12]
        """
        geo2ix = {s: i for i, s in enumerate(geo_splited)}
        geomap = []
        for c in geo:
            geomap.append([geo2ix[sub] for sub in gen_sub_paths(c)])

        with open(fsave, 'w') as f:
            writer = csv.writer(f)
            for row in geomap:
                writer.writerow(row)

        ret = torch.zeros((len(geomap), max(geo2ix.values()) + 1),
                          dtype=torch.float32)
        for i, ixs in enumerate(geomap):
            ret[i, ixs] = 1
        return ret

    geo = pd.read_csv(config.join("geography.csv"), header=None)[0].values
    geo_splited = _split(geo, config.join("geography_splited.csv"))
    _ = build_geomap(geo, geo_splited, config.join('geomap.csv'))

    city_labels = labels[train_test].unique()
    cities = geo[city_labels]
    pd.Series(cities).to_csv(config.join("cities.csv"),
                             index=False,
                             header=False)
    cities_splited = _split(cities, config.join("cities_splited.csv"))
    geomap_cities = build_geomap(cities, cities_splited,
                                 config.join('geomap_cities.csv'))

    ######################################################################

    # set noncities to -1
    old2new = [-1] * (max(city_labels) + 1)
    for i, l in enumerate(city_labels):
        old2new[l] = i
    ttl = labels[train_test].apply(lambda l: old2new[l])
    labels[:] = -1
    labels.loc[train_test] = ttl
    y = torch.from_numpy(labels.values)

    if config.multilabel:
        num_classes = len(cities_splited)
        names = cities_splited
        y = MultilabelTarget(y, geomap_cities)
    else:
        num_classes = len(city_labels)
        names = cities

    return y, num_classes, names


def save_data(config, anew=True):
    # веса VK не используем, так как 98% из них одинаковы

    labels = pd.Series(
        to_numpy(load_vector(config.join("labels.bin"), 'i'), 'l'))
    num_nodes = len(labels)

    nodes = None
    if not config.directed:
        nodes = to_numpy(load_vector(config.join("nodes.bin"), 'I'))

    train, test, _ = train_test_partition(config, labels, nodes, anew=anew)
    train_test = np.concatenate([train, test])
    y, num_classes, cities = get_target(config, labels, train_test)
    print("num_classes = %s" % num_classes)

    data = Data(y=y)

    data.num_nodes = num_nodes
    data.num_classes = num_classes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.uint8)
    data.train_mask[train] = 1
    data.test_mask = torch.zeros(num_nodes, dtype=torch.uint8)
    data.test_mask[test] = 1
    data.cities = cities

    torch.save(data, config.join('data%s.pt' % config.postfix))


def compute_mean_std(config, data):
    def compute(x, ixs, chunk_size):
        device = config.device
        total = math.ceil(len(ixs) / chunk_size)
        s1 = torch.zeros(x.shape[1], dtype=torch.float32).to(device)
        s2 = torch.zeros(x.shape[1], dtype=torch.float32).to(device)
        counts = torch.zeros(x.shape[1], dtype=torch.int64).to(device)
        for ch_ixs in tqdm(tools.grouper(ixs, chunk_size),
                           total=total,
                           desc="mean/std"):
            ch_ixs = np.array(ch_ixs, dtype='I')
            chunk = torch.from_numpy(x[ch_ixs]).to(device)
            s1 += chunk.sum(axis=0)
            s2 += torch.square(chunk).sum(axis=0)
            counts += (chunk != 0).sum(dim=0)

        mu = s1 / x.shape[0]
        var = s2 / x.shape[0] - torch.square(mu)
        std = torch.pow(var, 0.5)
        std[std == 0] = 1.
        return counts, mu, std

    if config.directed:
        ixs = range(data.x.shape[0])
    else:
        ixs = load_vector(config.join("nodes.bin"), "I")

    counts_node, mu_node, std_node = compute(data.x, ixs, 10**6)
    lessthen = (counts_node < config.city_freq_threshold).sum()
    if lessthen:
        print("node features having less then %d counts: %d" %
              (config.city_freq_threshold, lessthen))

    counts_edge, mu_edge, std_edge = None, None, None
    if config.directed:
        ixs = range(data.edge_attr.shape[0])
        counts_edge, mu_edge, std_edge = compute(data.edge_attr, ixs, 10**7)

    stat = Data(counts_node=counts_node,
                mu_node=mu_node,
                std_node=std_node,
                counts_edge=counts_edge,
                mu_edge=mu_edge,
                std_edge=std_edge)
    torch.save(stat, config.stat_path)


def load_data(config: Config):
    Data.num_node_features = property(lambda self: self.x.shape[1])
    data = torch.load(config.pt_data_path)
    data.x = CSRMatrix(config.node_csr_data_path, config.node_csr_indices_path,
                       config.node_csr_indptr_path, config.cpu_count)

    if config.directed:
        Data.num_edge_features = property(lambda self: self.edge_attr.shape[1])
        data.edge_attr = CSRMatrix(config.edge_csr_data_path,
                                   config.edge_csr_indices_path,
                                   config.edge_csr_indptr_path,
                                   config.cpu_count)
        data.reverse_edge_map = to_numpy(
            load_vector(config.reverse_edge_map_path, 'I'))

    if not os.path.exists(config.stat_path):
        compute_mean_std(config, data)
    stat = torch.load(config.stat_path)
    data.stat = stat
    if config.directed:
        data.double_mu_edge = torch.cat([stat.mu_edge, stat.mu_edge])
        data.double_std_edge = torch.cat([stat.std_edge, stat.std_edge])

    for part_path in config.part_dirs:
        print(part_path)
        torch.cuda.empty_cache()
        if not config.dev_mode:
            data.predict_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            if config.directed:
                findex = join(part_path, "predict_index.bin")
            else:
                findex = join(part_path, "nodes.bin")
            data.predict_mask[load_vector(findex, 'I')] = 1
        data.edge_index = torch.from_numpy(
            load_2d_vec(join(part_path, "colrow.bin"),
                        nrows=2,
                        typecode='i',
                        order='F'))
        deg = None
        if config.directed:
            deg = torch.from_numpy(
                to_numpy(load_vector(join(part_path, "degrees.bin"), 'I'),
                         'l'))

        nbr_sampler = NeighborSampler(config, data, deg=deg, batch_size=1000)

        yield nbr_sampler
