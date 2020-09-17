import os
import pika
import orjson as json
from datetime import datetime
from datetime import timezone
from os.path import join
from tqdm import tqdm
import ccmg_cassandra
import argparse
import time
from cassandra.cluster import BatchStatement
from cassandra import ConsistencyLevel
import cassandra
from itertools import islice
import time


def message_pattern(now):
    msg = {
        "account_id": None,
        "ptime": now,
        "ctime": now,
        "event_type": None,
        "event": None
    }
    return msg


def cassanda_connection():
    return ccmg_cassandra.cassandra(executor_threads=4, request_timeout=60)


def channel_basic_publish(msg):
    body = json.dumps(msg).decode()
    channel.basic_publish(exchange=msg['event_type'],
                          routing_key='',
                          body=body)


# собственно вот это событие: https://redmine.company.ru/projects/dev-documentation/wiki/ccmg_event_profile_change
# его надо посылать в рэббит всегда, когда вносятся какие-либо изменения в профиль. Создание профиля тоже считаем его изменением

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('socialnet',
                        choices={"vk", "facebook", "twitter"},
                        help='name of a social network')
    parser.add_argument("--cassandra",
                        help="publish to cassandra",
                        action="store_true")
    parser.add_argument("--ccmg", help="publish to ccmg", action="store_true")
    parser.add_argument("--all",
                        help="publish to cassandra and ccmg",
                        action="store_true")
    parser.add_argument("--line",
                        help="start from line _",
                        type=int,
                        default=0)
    parser.add_argument("--top", help="take top _", type=int, default=0)

    args = parser.parse_args()
    # args = parser.parse_args(['facebook', '--all'])

    if args.all:
        args.cassandra, args.ccmg = True, True
    elif not args.cassandra and not args.ccmg:
        parser.error('No action requested')

    params = pika.URLParameters('amqp://afdb:afdb01@blabla.company:5672')
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    casconn = cassanda_connection()
    batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)

    MAX_INT = 2**31 - 1
    scale = 10**10
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    now = now.replace('+00:00', 'Z')
    platform = '%s.com' % args.socialnet
    base = '../data/%s/' % args.socialnet
    if args.socialnet == 'vk':
        base = os.path.join(base, 'predict')
    else:
        base = os.path.join(base, 'train')

    fname = join(base, 'extids.csv')
    num_lines = int(os.popen('wc -l %s' % fname).read().split()[0])
    cities = open(join(base, 'cities_splited.csv')).read().split()
    query = """
        UPDATE people SET profile_proc_location_pred = ? WHERE extid = ? and platform = ?
        """
    query = casconn.session().prepare(query)
    msg = message_pattern(now)

    with open(join(base, 'extids.csv')) as f1, \
        open(join( base, 'city_ixs.csv')) as f2, \
        open(join(base, 'probs.csv')) as f3:
        it = enumerate(tqdm(zip(f1, f2, f3), total=num_lines))
        if args.top:
            it = islice(it, args.top)
        for i, (l1, l2, l3) in it:
            if i < args.line:
                continue
            events = []
            extid = l1.strip()
            cixs = [int(v) for v in l2.split()]
            nprimary = cixs[0]
            probs = [float(v) for v in l3.split()]
            for j, (ix, prob) in enumerate(zip(cixs[1:], probs)):
                rubrs = [cities[ix]]
                score = int(prob * scale)
                assert 0 <= score <= MAX_INT
                primary = int(j < nprimary)

                event = {"rubrs": rubrs, "score": score, "primary": primary}

                # assert 0

                # print(body)
                if args.ccmg:
                    msg['account_id'] = "%s@%s" % (extid, platform)

                    msg['event_type'] = 'event_profile_field_location_pred'
                    msg['event'] = event
                    channel_basic_publish(msg)

                    msg['event_type'] = 'event_profile_change'

                    msg['event'] = {"profile_field_id": 'extid'}
                    channel_basic_publish(msg)

                    msg['event'] = {
                        "profile_field_id": 'profile_proc_location_pred'
                    }
                    channel_basic_publish(msg)
                elif args.cassandra:
                    events.append(event)

            # send events to cassandra
            if args.cassandra:
                txt = json.dumps(events).decode()
                batch.add(query, (txt, extid, platform))
                if len(batch) == 100:
                    try:
                        casconn.session().execute(batch, timeout=60)
                    except (cassandra.OperationTimedOut,
                            cassandra.WriteTimeout) as e:
                        time.sleep(30)  # important, really useful
                        casconn = cassanda_connection()
                        casconn.session().execute(batch, timeout=60)

                    batch.clear()

        # finally
        if len(batch):
            casconn.session().execute(batch, timeout=60)
