from ccmg_cluster import query_all, ccmgCluster, listify_aids, request
from itertools import islice, chain, groupby
from collections import Counter
from tqdm import tqdm
import csv
import tools
from tools import *
from extractor import extract_city
from itertools import dropwhile
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import argparse
from os.path import join, isfile
import asyncio
import pickle
from functools import partial
from stlmap import MapU64U32
import array


def make_query(shard_n, account_ids, org):
    account_ids = listify_aids(account_ids, org)
    SQL = """
    SELECT ae.account_id, unnest(text_rubrs) AS rub, COUNT(1)
    FROM shard_{0}.account_event AS ae
    JOIN shard_{0}.event_post_geo  AS eg
    ON ae.event_id = eg.id AND ae.account_id in {1}
        AND ae.event = 'event_post_geo'
        AND ae.ptime > '2019-01-01'::DATE
        -- AND ae.ptime BETWEEN '2019-01-01'::DATE AND '2019-07-18'::DATE
    GROUP BY ae.account_id, rub
    """.format(shard_n, str(account_ids))
    # print(SQL)
    return SQL


def account_rubrs(pool, shard_n, account_ids, org='vk.com'):
    SQL = make_query(shard_n, account_ids, org)
    return request(pool, SQL)


def rec_fn(rows):
    return [[rec['rub'], str(rec['count'])] for rec in rows]


def analyze(basedir, typecode):
    # basedir='../data/twitter'
    # typecode = 'L'
    cities = pd.read_csv(join(basedir, "geography.csv"), header=None)[0]
    labels = to_numpy(load_vector(join(basedir, "labels.bin"), 'i'))
    nodes = tools.load_vector(join(basedir, "nodes.bin"), typecode)
    id2ix = MapU64U32()
    for ix, _id in enumerate(nodes):
        id2ix[_id] = ix

    found = array.array('f')
    with open(join(basedir, "post_geo.csv")) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for line in tqdm(reader):
            ix = id2ix[int(line[0])]
            l = labels[ix]
            if l > -1:
                city = cities[l]
                splited = line[1].split(',')
                try:
                    index = splited[::2].index(city)
                except:
                    continue
                counts = np.array(splited[1::2], dtype='I')
                # r = float(counts[index]) / counts.sum()
                r = counts[index]
                found.append(r)

    found = to_numpy(found)
    qs = [[.2, .3, .4, .5, .6, .7, .8, .9]]
    qs.append(np.quantile(found, qs[0]))
    qs = np.array(qs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download post geo data')
    parser.add_argument('socialnet',
                        choices={"vk", "facebook", "twitter"},
                        help='name of a social network')
    parser.add_argument("--email",
                        action='store_true',
                        help="send email on completion")
    args = parser.parse_args()

    basedir = '../data/%s/' % args.socialnet

    fcluster = join(basedir, 'cluster.pkl')
    fname = join(basedir, 'post_geo.csv')
    fnodes = join(basedir, 'nodes.bin')

    if isfile(fcluster):
        with open(fcluster, 'rb') as f:
            cl = pickle.load(f)
    else:
        typecode = 'I' if args.socialnet == 'vk' else 'L'
        cl = ccmgCluster(load_vector(fnodes, typecode), args.socialnet)
        cl.save(fcluster)

    asyncio.run(
        query_all(cl,
                  partial(account_rubrs, org='%s.com' % args.socialnet),
                  rec_fn,
                  fname,
                  chunk_size=500,
                  threads_per_host=8))

    if args.email:
        tools.send_email(body="%s completed" % __file__)