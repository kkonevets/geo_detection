import multiprocessing
import ccmg_cassandra
from concurrent.futures import ThreadPoolExecutor
from cassandra import OperationTimedOut
from tqdm import tqdm
import os
import signal
import numpy as np

TOKEN_BEGIN = -2**63
TOKEN_END = 2**63 - 1
TIMEOUT_OCCURRED = False

OKGREEN = '\033[92m'
ENDC = '\033[0m'


def partition(min_id, max_id, buckets_cnt):
    total = max_id - min_id
    buckets_cnt = min(total, buckets_cnt)
    step = float(total) / buckets_cnt
    buckets = []
    current = min_id
    for i in range(buckets_cnt):
        left = int(current)
        if i == buckets_cnt - 1:
            right = max_id
        else:
            right = int(current + step)
        buckets.append((left, right))
        current += step
    return buckets


def select_star(fields,
                table,
                min_fields=2,
                limit=1000,
                timeout=60,
                threads_cnt=5,
                begin=TOKEN_BEGIN,
                end=TOKEN_END,
                platform=None,
                filename='output.csv'):
    cassandra = ccmg_cassandra.cassandra(request_timeout=60)

    global TIMEOUT_OCCURRED
    TIMEOUT_OCCURRED = False
    print("saving %s" % filename)

    splited = os.path.split(filename)
    fparts = splited[0] + '/parts_%s' % os.path.splitext(splited[1])[0]
    parts = []
    if os.path.exists(fparts):
        parts = np.genfromtxt(fparts, delimiter=' ',
                              dtype=np.int64).reshape(-1, 2).tolist()
        if len(parts) == 0:
            return TIMEOUT_OCCURRED

    anew = len(parts) == 0
    if anew:
        if os.path.exists(filename):
            os.remove(filename)
        parts = partition(begin, end, threads_cnt)

    STOP_LINE = 'sdfghdghjfdghjsfg(20s)'
    output_queue = multiprocessing.Queue()
    writer_process = multiprocessing.Process(target=__writer,
                                             args=(filename, output_queue,
                                                   STOP_LINE, fparts))
    writer_process.start()
    if anew:
        output_queue.put((','.join(fields), None))

    def _func(partn_from_to):
        partn, from_to = partn_from_to
        return select_from_to(partn, output_queue, from_to[0], from_to[1],
                              cassandra, fields, table, min_fields, limit,
                              timeout, platform)

    with ThreadPoolExecutor(max_workers=threads_cnt) as executor:
        results = executor.map(_func, enumerate(parts))
        for _ in results:
            pass

    output_queue.put((STOP_LINE, None))
    writer_process.join()

    return TIMEOUT_OCCURRED


def select_from_to(partn, output_queue, token_from, token_to, cassandra,
                   fields, table, min_fields, limit, timeout, platform):
    global TIMEOUT_OCCURRED
    bar = tqdm(total=token_to - token_from,
               bar_format='{l_bar}{bar}[{elapsed}<{remaining}]')
    current_token = token_from
    query = prepare_select_star(cassandra, fields, table, platform, limit)
    while current_token < token_to and not TIMEOUT_OCCURRED:
        args = (current_token, )
        try:
            rows = select(cassandra, query, args, timeout)
        except OperationTimedOut as e:
            try:
                rows = select(cassandra, query, args, timeout)
            except OperationTimedOut as e:
                TIMEOUT_OCCURRED = True
                bar.close()
                return

        prev_current_token = current_token
        if len(rows) == 0:
            current_token = token_to
        else:
            current_token = max(map(lambda row: row['a_token'], rows)) + 1
        if current_token >= token_to:
            rows = filter(lambda row: row['a_token'] < token_to, rows)
            current_token = token_to

        for row in rows:
            if not row['extid'].isdigit():
                continue

            del row['a_token']
            row = {k: v for k, v in row.items() if v}
            if len(row) >= min_fields:  # extid and platform
                v = row.get('friends')
                if v:
                    row['friends'] = v.replace('\n', ' ')
                s = ','.join((row.get(k, "") for k in fields))
                part_info = (partn, current_token, token_to)
                output_queue.put((s, part_info))

        bar.update(current_token - prev_current_token)

    bar.close()


################################################################################


def prepare_select_star(cassandra, fields, table, platform, limit=1000):
    field_list_str = ','.join(fields)
    query = "SELECT token(extid) AS a_token, {} FROM {} \
            WHERE token(extid) >= ?  and platform = '{}' LIMIT {} allow filtering"

    query = query.format(field_list_str, table, platform, limit)
    return cassandra.session().prepare(query)


def select(cassandra, query, args, timeout=60):
    future = cassandra.session().execute_async(query, args, timeout=timeout)
    return [dict(r._asdict()) for r in future.result()]


def __writer(filename, queue, stop_token, fparts):
    states = {}

    try:
        with open(filename, 'a') as f:
            while True:
                line, part_info = queue.get()
                if line == stop_token:
                    return

                f.write(line + '\n')
                if part_info:
                    partn, current_token, token_to = part_info
                    states[partn] = (current_token, token_to)
    except KeyboardInterrupt as e:
        os.kill(os.getppid(), signal.SIGTERM)
    finally:
        print(f"\n{OKGREEN}saving current progress, "\
            f"closing writer{ENDC}")
        parts = [states[k] for k in sorted(states)]
        np.savetxt(fparts, parts, fmt='%i')
