from os.path import join, exists
import importlib
import sys
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import tools
from tools import load_vector, load_2d_vec, to_numpy, save_labels, chunks
import array
from scipy import sparse
from itertools import islice, groupby, chain
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from cymatrix import CSRMatrix
import multiprocessing
import math
import argparse
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import inf
import sage_conv
from sage_conv import SAGEConv, SAGEConvWithEdges
from data import Config, gen_sub_paths, save_data, load_data

# TODO:
# проверить возможность multiGPU


class SAGENet(torch.nn.Module):
    def __init__(self, n_node_ftrs, n_classes, n_hidden=1500):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(n_node_ftrs, n_hidden)
        self.conv2 = SAGEConv(n_hidden, n_classes)

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = self.conv1(x, block.res_size, block.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        block = data_flow[1]
        x = self.conv2(x, block.res_size, block.edge_index)
        return F.log_softmax(x, dim=1)


class SAGENetWithEdges(torch.nn.Module):
    def __init__(self, n_node_ftrs, n_edge_ftrs, n_classes, n_hidden=1500):
        super(SAGENetWithEdges, self).__init__()
        self.conv1 = SAGEConvWithEdges(n_node_ftrs, n_edge_ftrs, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.conv2 = SAGEConvWithEdges(n_hidden, n_edge_ftrs, n_classes)

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = self.conv1(x, block.res_size, block.edge_index, block.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.bn(x)

        block = data_flow[1]
        x = self.conv2(x, block.res_size, block.edge_index, block.edge_attr)
        return F.log_softmax(x, dim=1)


def slice_arr(arr, nids, device='cpu'):
    if type(arr) == CSRMatrix:
        if type(nids) == np.ndarray:
            x = arr[nids]
        else:
            x = arr[nids.numpy().astype(np.uint32)]
    else:
        x = arr[nids]

    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    elif type(x) == sparse.csr.csr_matrix:
        x = torch.from_numpy(x.A)

    return x.to(device)


def slice_data_flow(config, data, data_flow):
    def std_norm(x, mean, std):
        return (x.sub(mean)).true_divide(std)

    def slice_edge_attr(block, device):
        edge_attr_0 = slice_arr(data.edge_attr, block.e_id, device=device)
        rev_e_id = data.reverse_edge_map[block.e_id]
        edge_attr_1 = slice_arr(data.edge_attr, rev_e_id, device=device)

        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)
        # self loop edges as ones
        edge_attr = F.pad(input=edge_attr,
                          pad=(0, 0, 0, block.res_size),
                          value=1)
        if config.directed:
            edge_attr = std_norm(edge_attr, data.double_mu_edge,
                                 data.double_std_edge)
        return edge_attr

    x = slice_arr(data.x, data_flow[0].n_id, device=config.device)
    if config.directed:
        x = std_norm(x, data.stat.mu_node, data.stat.std_node)
        for block in data_flow:
            block.edge_attr = slice_edge_attr(block, config.device)
    return x


def init_model(config, data):
    if config.directed:
        model = SAGENetWithEdges(data.num_features, 2 * data.num_edge_features,
                                 data.num_classes)
    else:
        model = SAGENet(data.num_features, data.num_classes)

    model.to(config.device)
    # weight_decay=5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    return model, optimizer


def load_model(config, data):
    checkpoint = torch.load(config.join('checkpoint%s.pt' % config.postfix))
    model, optimizer = init_model(config, data)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def tqdm_total(mask, batch_size):
    total = mask.sum().item()
    return math.ceil(total / batch_size)


def transductive_sage(nbr_sampler, config):
    gc.collect()
    torch.cuda.empty_cache()
    data = nbr_sampler.data
    device = config.device
    if config.resume:
        model, optimizer = load_model(config, data)
    else:
        model, optimizer = init_model(config, data)
    print(model)

    topn = 5

    def train(global_step, epoch):
        model.train()

        total = tqdm_total(data.train_mask, nbr_sampler.batch_size)
        total_loss = 0
        max_test_score = 0
        for i, data_flow in tqdm(enumerate(nbr_sampler(data.train_mask), 1),
                                 total=total,
                                 position=0):
            optimizer.zero_grad()
            x = slice_data_flow(config, data, data_flow)
            # break

            out = model(x, data_flow.to(device))
            y = slice_arr(data.y, data_flow.n_id, device=device)
            if config.multilabel:
                loss = 1000 * \
                    F.binary_cross_entropy_with_logits(out, y)
            else:
                loss = F.nll_loss(out, y)  # weight=weight
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data_flow.batch_size

            if global_step % config.test_every == 0 or global_step >= config.steps:
                # break  # !!!!!!
                test_score = evaluate()
                if test_score > max_test_score:
                    max_test_score = test_score
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(checkpoint,
                               config.join('checkpoint%s.pt' % config.postfix))
                print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
                    epoch, total_loss / (i * data_flow.batch_size),
                    test_score))
                if global_step >= config.steps:
                    return global_step
                model.train()

            global_step += 1

        return global_step

    # def preds_by_level(out, nested, level):
    #     """
    #     nested = levels[asorted]
    #     """
    #     nz = (nested == level).nonzero()
    #     gr = nz[:, 0]
    #     first = (gr[1:] - gr[:-1]).nonzero().squeeze() + 1
    #     nz = nz[torch.cat([torch.tensor([0]).to(device), first])]
    #     probs = torch.exp(out[nz[:, 0], nz[:, 1]])
    #     return probs

    def evaluate():
        model.eval()
        correct = 0
        recall_at_k = 0
        with torch.no_grad():
            for data_flow in nbr_sampler(data.test_mask):
                x = slice_data_flow(config, data, data_flow)
                y = slice_arr(data.y, data_flow.n_id, device=device)
                out = model(x, data_flow.to(device))
                if config.multilabel:
                    asorted = out.argsort(descending=True)[:, :topn]
                    matches = torch.gather(y, 1, asorted)
                    recall_at_k += (matches.sum(axis=1) /
                                    y.sum(axis=1)).sum().item()
                else:
                    pred = out.max(1)[1]
                    correct += pred.eq(y).sum().item()

            if config.multilabel:  # average recall@k
                metric = recall_at_k / data.test_mask.sum().item()
            else:  # accuracy
                metric = correct / data.test_mask.sum().item()
        return metric

    global_step, epoch = 1, 1
    while True:
        global_step = train(global_step, epoch)
        if global_step >= config.steps:
            break
        epoch += 1


def test(nbr_sampler, config):
    data = nbr_sampler.data
    multilabel, device = config.multilabel, config.device
    model, _ = load_model(config, data)
    model.eval()

    # _, prior = torch.unique(data.y[data.train_mask == 1], return_counts=True)
    # prior = prior.to(torch.float32)/prior.sum()
    # prior = prior.to(device)

    city2ix = {c: i for i, c in enumerate(data.cities)}
    if multilabel:
        test_cities = [
            'co_Russia.r1_CentralRussia',
            'co_Russia.r1_CentralRussia.r2_MosObl',
            'co_Russia.r1_CentralRussia.r2_MosObl.ci_Moscow',
            'co_Russia.r1_NorthWest',
            'co_Russia.r1_NorthWest.r2_SpbObl',
            'co_Russia.r1_NorthWest.r2_SpbObl.ci_SaintPetersburg',
            'co_Russia.r1_Urals',
            'co_Russia.r1_Urals.r2_Sverdlovsk',
            'co_Russia.r1_Urals.r2_Sverdlovsk.ci_Yekaterinburg',
            'co_Russia.r1_FarEast',
            'co_Russia.r1_FarEast.r2_Yakutia',
            'co_Russia.r1_FarEast.r2_Yakutia.ci_Yakutsk',
            'co_Russia.r1_Siberia',
            'co_Russia.r1_Siberia.r2_Chita',
            'co_Russia.r1_Siberia.r2_Chita.ci_Chita',
        ]
    else:
        test_cities = [
            'p_Europe.co_Russia.r1_CentralRussia.r2_MosObl.ci_Moscow',
            'p_Europe.co_Russia.r1_NorthWest.r2_SpbObl.ci_SaintPetersburg',
            'p_Europe.co_Russia.r1_Urals.r2_Sverdlovsk.ci_Yekaterinburg',
            'p_Europe.co_Russia.r1_Siberia.r2_Novosibirsk.ci_Novosibirsk',
            'p_Europe.co_Belorussia.r1_MinskObl.ci_Minsk',
            'p_Europe.co_Russia.r1_CentralRussia.r2_Tula.ci_Tula',
            'p_Europe.co_Russia.r1_Siberia.r2_Chita.ci_Chita',
            'p_Europe.co_Russia.r1_CentralRussia.r2_Bryansk.ci_Bryansk',
            'p_Europe.co_Russia.r1_FarEast.r2_Yakutia.ci_Yakutsk',
            # 'p_Europe.co_Russia.r1_CentralRussia.r2_Orel',
        ]

    ixs = [city2ix[c] for c in test_cities]
    if multilabel:
        plt_ixs = list(chunks(ixs, 3))

    probs, preds, y = [], [], []
    with torch.no_grad():
        for data_flow in nbr_sampler(data.test_mask):
            if multilabel:
                y_cur = data.y[data_flow.n_id]
            else:
                y_cur = data.y[data_flow.n_id].numpy()
            x = slice_data_flow(config, data, data_flow)
            logits = model(x, data_flow.to(device))
            probs.append(torch.exp(logits).cpu().detach().numpy())
            pred = logits.max(1)[1]
            preds.append(pred.cpu().detach().numpy())
            y.append(y_cur)

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    y = np.concatenate(y)

    if not multilabel:
        ntest = data.test_mask.sum().item()
        accuracy = (preds == y).sum() / ntest
        print(accuracy)
        Y = label_binarize(y, classes=ixs)

        # ids = load_vector("../data/vk/nodes.bin", 'I')
        # df = pd.DataFrame(probs)
        # df.columns = data.cities
        # df.index = [ids[i] for i in data_flow.n_id]
        # df.to_csv('../data/vk/preds_example.csv')
    else:
        Y = y

        df = pd.DataFrame()
        df['geo'] = data.cities
        # choose thresholds suche that precision >= precision_min
        for th in [config.precision_min1, config.precision_min2]:
            recalls, thresholds = [], []
            for i in range(Y.shape[1]):
                precision, recall, _thresholds = precision_recall_curve(
                    Y[:, i], probs[:, i])
                wix = np.where(precision >= th)[0][0]
                if wix == len(_thresholds):
                    thresholds.append(1)
                    recalls.append(0)
                else:
                    thresholds.append(_thresholds[wix])
                    recalls.append(recall[wix])

            df['recall_%.2f' % th] = recalls
            df['threshold_%.2f' % th] = thresholds

        df.to_csv(config.join('thresholds.csv'), index=False)

    def plot_precision_recall(Y, probs, ixs, fname):
        plt.clf()
        for i in ixs:
            precision, recall, _ = precision_recall_curve(Y[:, i], probs[:, i])
            plt.plot(recall,
                     precision,
                     lw=1,
                     label=data.cities[i].split('.')[-1])

        plt.legend(fontsize='small', loc='lower left')
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision vs. recall curve")
        plt.savefig(fname, dpi=900)

    fname = config.join('precision_recall%s_all.pdf' % config.postfix)
    plot_precision_recall(Y, probs, ixs, fname)
    if multilabel:
        for i, sub_ixs in enumerate(plt_ixs):
            fname = config.join('precision_recall%s_%s.pdf' %
                                (config.postfix, i))
            plot_precision_recall(Y, probs, sub_ixs, fname)


def __writer(config, queue, geos, city_mask):
    geo2ix = {g: i for i, g in enumerate(geos)}
    ix2path = []
    for g in geos:
        sub = [geo2ix[sub_g] for sub_g in gen_sub_paths(g, False)]
        ix2path.append(sub)

    def denestify(geo_ixs):
        """
        geo_ixs = [1,2,5,8,9,10]
        ix2path = {1: [1], 2: [2], 9: [2,4,9], 5: [2,5], 
                    8: [2,5,8], 10: [2,5,10]}
        return:
            [True, False, False, True, True, True]
            # [1,8,9,10]
        """
        if len(geo_ixs) == 0:
            return []

        ret = []
        groups = [(l, list(g)) for l, g in groupby(
            sorted([ix2path[ix] for ix in geo_ixs], key=len), key=len)]
        to_delete = []
        for gi, (l_small, g_small) in enumerate(groups):
            for _, g_big in groups[gi + 1:]:
                for p_small in g_small:
                    last = p_small[-1]
                    for p_big in g_big:
                        if last == p_big[l_small - 1]:
                            to_delete.append(last)
                            break
        return [ix not in to_delete for ix in geo_ixs]

    with open(config.extids_fname, 'a') as f, \
            open(config.cixs_fname, 'a') as g, \
            open(config.probs_fname, 'a') as h:
        extids_writer = csv.writer(f, delimiter=' ')
        ixs_writer = csv.writer(g, delimiter=' ')
        probs_writer = csv.writer(h, delimiter=' ')

        while True:
            msg = queue.get()
            # msg = (data_flow.n_id.detach().numpy(), ixs.cpu().detach().numpy(),
            #        logits.cpu().detach().numpy())
            if msg is None:
                return

            for extid, row_ixs, row_logits in zip(*msg):
                ixs_primary, logits_primary = [], []
                ixs_secondary, logits_secondary = [], []
                ncities_secondary = 0
                for ix, logit in zip(row_ixs, row_logits):
                    if logit <= 0:
                        ixs_primary.append(ix)
                        logits_primary.append(logit)
                    elif logit < np.inf and ncities_secondary == 0:
                        # search for first city under threshold
                        ncities_secondary += int(city_mask[ix])
                        ixs_secondary.append(ix)
                        logits_secondary.append(-logit)
                    else:
                        break

                # denestify geographies
                mask_primary = denestify(ixs_primary)
                mask_secondary = denestify(ixs_secondary)

                try:
                    index = mask_secondary.index(True) + 1
                    mask = mask_primary + mask_secondary[:index]
                except:
                    mask = mask_primary

                ixs_out = chain(ixs_primary, ixs_secondary)
                ixs_out = [v for b, v in zip(mask, ixs_out) if b]
                logits_out = chain(logits_primary, logits_secondary)
                logits_out = [v for b, v in zip(mask, logits_out) if b]
                ixs_out.insert(0, sum(mask_primary))

                extids_writer.writerow([extid])
                ixs_writer.writerow(ixs_out)
                probs_writer.writerow(np.exp(logits_out))


def load_thresholds(thresholds, precision_min, device):
    thresholds = torch.log(
        torch.from_numpy(thresholds['threshold_%.2f' %
                                    precision_min].values.astype(
                                        np.float32)).to(device))
    return thresholds


def predict(nbr_sampler, config):
    data = nbr_sampler.data
    model, _ = load_model(config, data)
    model.eval()
    device = config.device
    thresholds = pd.read_csv(config.join('thresholds.csv'))
    thresholds1 = load_thresholds(thresholds, config.precision_min1, device)
    thresholds2 = load_thresholds(thresholds, config.precision_min2, device)

    geos = open(config.join('cities_splited.csv')).read().split()
    if config.directed:
        nodes = to_numpy(load_vector(config.join("nodes.bin"), 'L'))

    def last_starts_with(prefix):
        return torch.tensor(
            [s.split('.')[-1].startswith(prefix) for s in geos]).to(device)

    r2_mask = last_starts_with('r2_')
    r3_mask = last_starts_with('r3_')
    city_mask = tools.get_city_mask(geos, exclude_continent=False)
    city_mask = torch.tensor(city_mask).to(device)

    no_interest = ~(r2_mask | r3_mask | city_mask)

    output_queue = multiprocessing.Queue()
    writer_process = multiprocessing.Process(target=__writer,
                                             args=(config, output_queue, geos,
                                                   city_mask.cpu().numpy()))
    writer_process.start()
    with torch.no_grad():
        total = tqdm_total(data.predict_mask, nbr_sampler.batch_size)
        for data_flow in tqdm(nbr_sampler(data.predict_mask), total=total):
            x = slice_data_flow(config, data, data_flow)
            logits = model(x, data_flow.to(device))
            logits[:, no_interest] = -inf
            logits[logits < thresholds2] = -inf
            logits[logits < thresholds1] *= -1
            cond = logits.min(dim=1)[0] < inf
            logits, ixs = torch.sort(logits[cond], dim=1)
            # break

            n_id = data_flow.n_id[cond].numpy()
            if config.directed: n_id = nodes[n_id]
            output_queue.put_nowait(
                (n_id, ixs.cpu().numpy(), logits.cpu().numpy()))

    # _id = 527
    # _ixs = ixs[_id].cpu().detach().numpy()
    # _probs = torch.exp(logits[_id]).cpu().detach().numpy()

    # df = pd.DataFrame([np.array(data.cities)[_ixs], _probs]).T
    # df.columns = ['geo', 'prob']
    # df['extid'] = data_flow.n_id[_id].detach().numpy()
    # df['th'] = torch.exp(thresholds)[_ixs].cpu().detach().numpy()
    # df['diff'] = df['prob'] - df['th']
    # df = df[['extid', 'geo', 'prob', 'th', 'diff']]
    # df.to_csv('../data/vk/example.csv', index=False)

    # put_nowait
    output_queue.put(None)
    writer_process.join()


if __name__ == "__main__":
    """
    Берем бо'льшую часть размеченных вершин для тренировки, остальные из
    размеченных (1%) для теста. Во время тренировки неразмеченные узлы и 
    тестовые используются для распространения сигнала, причем разметка 
    тестовых узлов не участвует в формировании фич на основе соседей, то есть 
    тестовые узлы берутся без разметки - учитывается только геометрия.
    Ситуация такая же как при label spreading - сигнал распростараняется от
    размеченных через неразмеченные по ребрам графа.
    """

    parser = argparse.ArgumentParser(
        description='Prepare data and train model')
    parser.add_argument('socialnet',
                        choices={"vk", "facebook", "twitter"},
                        help='name of a social network')
    parser.add_argument("--email",
                        action='store_true',
                        help="send email on completion")
    parser.add_argument("--train", action='store_true', help="train mode")
    parser.add_argument("--resume",
                        action='store_true',
                        help="continue training")
    parser.add_argument("--test", action='store_true', help="evaluate mode")
    parser.add_argument("--predict", action='store_true', help="predict mode")
    parser.add_argument('--labels', action='store_true', help="extract labels")
    parser.add_argument('--data', action='store_true', help="prepare data")

    args = parser.parse_args()
    # args = parser.parse_args(['twitter'])

    if (args.predict or args.data) and args.test:
        print("can't test while predict or data prepare", file=sys.stderr)
        exit(1)

    dev_mode = args.train or args.test
    base_dir = '../data/%s/' % args.socialnet
    if args.socialnet == 'vk':
        config = Config(base_dir=base_dir,
                        dev_mode=dev_mode,
                        degree_threshold=30,
                        degree_city_threshold=5,
                        city_freq_threshold=1000,
                        max_train_size=8 * 10**6,
                        steps=32_000,
                        test_every=500)
    elif args.socialnet == 'facebook':
        config = Config(base_dir=base_dir,
                        dev_mode=dev_mode,
                        directed=True,
                        degree_city_threshold=2,
                        city_freq_threshold=500,
                        nsample=[30, 5],
                        steps=5_000,
                        test_every=500)
        config.part_dirs = [base_dir]
    elif args.socialnet == 'twitter':
        config = Config(base_dir=base_dir,
                        dev_mode=dev_mode,
                        directed=True,
                        degree_city_threshold=2,
                        city_freq_threshold=500,
                        nsample=[30, 5],
                        steps=2_000,
                        test_every=100)
        config.part_dirs = [base_dir]
    else:
        raise NotImplemented

    config.resume = args.resume

    if args.labels or args.data:
        if args.labels:
            typecode = 'I' if args.socialnet == 'vk' else 'L'
            save_labels(config.base_dir, typecode)
        if args.data:
            save_data(config, anew=True)
    else:
        # nbr_sampler = next(load_data(config))
        importlib.reload(sage_conv)
        from sage_conv import SAGEConv, SAGEConvWithEdges

        for nbr_sampler in load_data(config):
            if args.train:
                transductive_sage(nbr_sampler, config)
            if args.test:
                test(nbr_sampler, config)
            if args.predict:
                predict(nbr_sampler, config)

    if args.email:
        tools.send_email(body="%s completed" % __file__)