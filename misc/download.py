from select_star import select_star
from tqdm import tqdm
import aiohttp
import asyncio
from tools import grouper, to_numpy
import array
import json
import os
from itertools import chain
import numpy as np
import tools
import traceback
import argparse


def sort_people(fin, fout, anew=False):
    if not anew:
        with open(fout, 'rb') as f:
            return np.load(f)

    num_lines = int(os.popen('wc -l %s' % fin).read().split()[0])
    ids = array.array('L', [])
    print('reading %s' % fin)
    with open(fin) as f:
        next(f)  # header
        for line in tqdm(f, total=num_lines):
            ids.append(int(line.split(',', 1)[0]))

    ids = to_numpy(ids)
    ids = np.unique(ids)  # unique and sorted
    with open(fout, 'wb') as f:
        np.save(f, ids)

    return ids


def tsar_download(ids, edge_type, platform):
    async def response_get(response, vxs):
        data = await response.read()
        assert response.status == 200, "%s, %s" % (response.status, vxs)

        nodes = array.array('L')
        counts = array.array('f')
        for vx, ec in zip(vxs, json.loads(data)):
            if ec:
                pairs = ((e.split('@')[0], c) for e, c in ec)
                pairs = ((int(e), c) for e, c in pairs
                         if e.isdigit() and int(e) != vx)
                ziped = list(zip(*pairs))
                if ziped:
                    _nodes, _counts = ziped
                    nodes.extend(chain((len(_nodes), vx), _nodes))
                    counts.extend(_counts)

        return nodes, counts

    async def fetch_one(session, vxs, platform):
        vxs_str = ','.join(('%d@%s' % (vx, platform) for vx in vxs))
        pattern = "http://graph_base.company/%s/%s?vertices=%s"

        url = pattern % ("edges_rev_count", edge_type, vxs_str)
        try:
            async with session.get(url) as response:
                return await response_get(response, vxs)
        except aiohttp.ServerDisconnectedError:
            async with session.get(url) as response:
                return await response_get(response, vxs)

    async def create_chunk(session, chunk, platform):
        return [fetch_one(session, vxs, platform) for vxs in chunk]

    async def fetch(chunk_size=10, batch_size=100):
        prefix = platform.split('.')[0]
        f_n = '../data/%s/%s_edjlist.bin' % (prefix, edge_type)
        f_w = '../data/%s/%s_weights.bin' % (prefix, edge_type)

        async with aiohttp.ClientSession() as session:
            with open(f_n, 'wb') as out_n, open(f_w, 'wb') as out_w:
                chunks = grouper(grouper(tqdm(ids), batch_size), chunk_size)
                for chunk in chunks:
                    tasks = await create_chunk(session, chunk, platform)
                    data = await asyncio.gather(*tasks)
                    for nodes, counts in data:
                        nodes.tofile(out_n)
                        counts.tofile(out_w)

    print(edge_type, platform)
    asyncio.run(fetch(10, 200))


def download_people(fname, platform, min_fields=2):
    while select_star(
        [
            "extid",
            'profile_proc_location',
            'profile_proc_city',
            'profile_proc_hometown',
            # 'prev_profile_proc_location',
            # 'prev_profile_proc_city',
            # 'prev_profile_proc_hometown',
        ],
            'people',
            min_fields=min_fields,
            limit=1000,
            timeout=60,
            threads_cnt=8,
            platform=platform,
            filename=fname):
        pass


def download_vk():
    platform = 'vk.com'
    while select_star(["extid", 'friends'],
                      'connections',
                      limit=500,
                      timeout=60,
                      threads_cnt=8,
                      platform=platform,
                      filename='../data/vk/connections.csv'):
        pass

    download_people('../data/vk/people.csv', platform)


def download_facebook():
    fin = '../data/facebook/people.csv'
    fout = '../data/facebook/ids_sorted.npy'
    platform = 'facebook.com'
    download_people(fin, platform, min_fields=1)

    ids = sort_people(fin, fout, anew=1)

    tsar_download(ids, 'mention', platform)
    tsar_download(ids, 'comment', platform)
    tsar_download(ids, 'share', platform)


def download_twitter():
    fin = '../data/twitter/people.csv'
    fout = '../data/twitter/ids_sorted.npy'
    platform = 'twitter.com'
    download_people(fin, platform, min_fields=1)

    ids = sort_people(fin, fout, anew=1)

    tsar_download(ids, 'comment', platform)
    tsar_download(ids, 'share', platform)


def download():
    download_vk()
    # download_facebook()
    # download_twitter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email",
                        action='store_true',
                        help="send email on completion/error")
    args = parser.parse_args()

    try:
        download()
    except KeyboardInterrupt:
        exit()
    except:
        if args.email:
            tools.send_email(body=traceback.format_exc())
        raise

    if args.email:
        tools.send_email(body="%s completed" % __file__)
