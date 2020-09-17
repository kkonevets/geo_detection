from tools import *
from ccmg_cluster import *
from tqdm import tqdm
import csv
from itertools import islice, chain, groupby
import pickle
import asyncio
import asyncpg


def make_query(shard_n, account_ids, table, field, org='vk.com'):
    account_ids = listify_aids(account_ids, org)
    SQL = '''
        SELECT ae.account_id, ec.{1} AS com_id, COUNT(ec.{1})
        FROM shard_{2}.account_event AS ae
        JOIN shard_{2}.{0} AS ec
        ON ae.event_id = ec.id AND ae.account_id in {3}
            AND ae.event = '{0}'
            AND ec.{1} NOT LIKE '-%%'
            AND ae.ptime BETWEEN '2019-01-01'::DATE AND '2019-07-18'::DATE
        GROUP BY ae.account_id, ec.{1}
        '''.format(table, field, shard_n, str(account_ids))
    return SQL


def commented_by(pool, shard_n, account_ids):
    SQL = make_query(shard_n, account_ids, 'event_comment_rev', 'src')
    return request(pool, SQL)


def commented_to(pool, shard_n, account_ids):
    SQL = make_query(shard_n, account_ids, 'event_comment', 'dst')
    return request(pool, SQL)


def rec_fn(rows):
    return [[rec['com_id'].split('@')[0], str(rec['count'])] for rec in rows]


if __name__ == "__main__":
    fby = '../data/vk/commented_by.csv'
    fto = '../data/vk/commented_to.csv'
    fcluster = '../data/vk/cluster.pkl'
    fnodes = '../data/vk/active_vk_accounts.csv'

    # cl = ccmgCluster(pd.read_csv(fnodes, header=None, dtype=np.uint32)[0])
    # cl.save(fcluster)

    with open(fcluster, 'rb') as f:
        cl = pickle.load(f)

    ###################################################################################

    asyncio.run(
        query_all(cl,
                  commented_by,
                  fby,
                  rec_fn,
                  chunk_size=500,
                  threads_per_host=15))
    asyncio.run(
        query_all(cl,
                  commented_to,
                  rec_fn,
                  fto,
                  chunk_size=500,
                  threads_per_host=15))

    ###################################################################################
