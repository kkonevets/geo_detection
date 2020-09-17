import ccmg_pgc_driver
import ccmg_pgc_scheme
import log
from tools import *
from tqdm import tqdm
from itertools import islice, chain, groupby
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
import struct
import asyncio
import asyncpg
from pprint import pprint
import time
import json
import requests
from requests.auth import HTTPBasicAuth

logger = log.log(None)  # or logger = log.log(open('/tmp/my.log', 'w'))
pgc = ccmg_pgc_driver.ccmg_pgc_driver(log=logger)


def rabbit_total_messages():
    resp = requests.get('http://blabla.company:15672/api/overview',
                        auth=('afdb', 'afdb01'))
    return json.loads(resp.text)['queue_totals']['messages']


async def request(pool, SQL):
    async with pool.acquire() as conn:
        rows = await conn.fetch(SQL)
        return rows


async def produce(queue, pool, it):
    start = time.time()
    for drive, shard_n, nodes in it:
        if time.time() - start > 5 * 60:
            if rabbit_total_messages() > 2 * 10**6:
                await asyncio.sleep(10 * 60)
            start = time.time()
        await queue.put((pool, shard_n, nodes))


async def consume(queue, fn, rec_fn, fname, pbar):
    while True:
        pool, shard_n, nodes = await queue.get()
        rows = await fn(pool, shard_n, nodes)
        with open(fname, 'a') as f:
            writer = csv.writer(f)
            for account_id, g in groupby(rows, lambda rec: rec['account_id']):
                coms = rec_fn(g)
                coms = ','.join(chain(*coms))
                writer.writerow([account_id.split('@')[0], coms])

        pbar.update(len(nodes))
        queue.task_done()


def listify_aids(account_ids, org):
    account_ids = tuple(['%d@%s' % (i, org) for i in account_ids])
    if len(account_ids) == 1:
        account_ids = (account_ids[0], account_ids[0])
    return account_ids


class ccmgCluster:
    """
    cl.host2drives = {1: ['drive1', 'drive2', 'drive3'], 2: ['drive4', 'drive5']}
    cl.drive2shards = {'drive1': [1, 2, 3], 'drive2': [4, 5, 6, 7], 'drive3': [8, 9],
                       'drive4': [10], 'drive5': [11, 12]}
    cl.shard2nodes = {1: [1, 2, 3], 2: [4, 5], 3: [6], 4: [7, 8, 9, 10], 5: [11],
                      6: [12], 7: [13, 14], 8: [15, 16], 9: [17, 18, 19, 20], 10: [21],
                      11: [22, 23], 12: [24, 25]}

    for host in cl.host2drives:
        pprint([host, list(cl.iter_host(host, chunk_size=2))])

    [1,
     [('drive1', 1, [1, 2]),
      ('drive2', 4, [7, 8]),
      ('drive3', 8, [15, 16]),
      ('drive1', 1, [3]),
      ('drive2', 4, [9, 10]),
      ('drive3', 9, [17, 18]),
      ('drive1', 2, [4, 5]),
      ('drive2', 5, [11]),
      ('drive3', 9, [19, 20]),
      ('drive1', 3, [6]),
      ('drive2', 6, [12]),
      ('drive2', 7, [13, 14])]]
    [2, [('drive4', 10, [21]), ('drive5', 11, [22, 23]), ('drive5', 12, [24, 25])]]
    """
    host2drives = defaultdict(lambda: set())
    drive2shards = defaultdict(lambda: list())
    shard2nodes = defaultdict(lambda: list())

    def __init__(self, nodes, socialnet, skip=False):
        if skip:
            return

        for node in tqdm(nodes):
            shard_n = pgc.get_people_shard('%d@%s.com' % (node, socialnet))
            self.shard2nodes[shard_n].append(node)

        scheme = ccmg_pgc_scheme.ccmg_pgc_scheme()
        for shard_n in self.shard2nodes:
            keys = scheme.sharding_keys(shard_n, 'r')
            self.drive2shards[keys['drive']].append(shard_n)
            self.host2drives[keys['ip']].update([keys['drive']])

    @staticmethod
    def column_iterator(row_iterators):
        while len(row_iterators):
            for i, (v1, it) in enumerate(row_iterators):
                try:
                    v2 = next(it)
                except StopIteration:
                    row_iterators.pop(i)
                    continue

                yield v1, v2

    def iter_drive(self, drive, chunk_size):
        for shard in self.drive2shards[drive]:
            for nodes in chunks(self.shard2nodes[shard], chunk_size):
                yield shard, nodes

    def iter_host(self, host, chunk_size=1000):
        iterators = [(drive, self.iter_drive(drive, chunk_size))
                     for drive in self.host2drives[host]]
        for drive, (shard, nodes) in self.column_iterator(iterators):
            yield drive, shard, nodes

    def save(self, fcluster):
        self.host2drives = dict(self.host2drives)
        self.drive2shards = dict(self.drive2shards)
        self.shard2nodes = dict(self.shard2nodes)
        with open(fcluster, 'wb') as f:
            pickle.dump(self, f)


async def query_all(cl,
                    fn,
                    rec_fn,
                    fname,
                    chunk_size=1000,
                    threads_per_host=36):
    pools = {}
    for host in cl.host2drives:
        pools[host] = await asyncpg.create_pool(user='pgc_user',
                                                password='',
                                                database='pgc',
                                                host=host,
                                                min_size=threads_per_host,
                                                max_size=36)

    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['extid', 'commented'])

    host2queue = {}
    for host in cl.host2drives:
        host2queue[host] = asyncio.Queue(maxsize=threads_per_host)

    pbars = []
    consumers = []
    for i, (host, queue) in enumerate(sorted(host2queue.items())):
        total = sum(
            [len(nodes) for _, _, nodes in cl.iter_host(host, chunk_size)])
        pbar = tqdm(total=total, position=i, desc=host)
        pbars.append(pbar)
        for i in range(threads_per_host):
            consumers.append(
                asyncio.create_task(consume(queue, fn, rec_fn, fname, pbar)))

    producers = []
    for host, queue in host2queue.items():
        it = cl.iter_host(host, chunk_size)
        producers.append(asyncio.create_task(produce(queue, pools[host], it)))

    await asyncio.wait(producers)
    await asyncio.wait([queue.join() for queue in host2queue.values()])
    for consumer in consumers:
        consumer.cancel()
    for pbar in pbars:
        pbar.close()
    for pool in pools.values():
        await pool.close()
