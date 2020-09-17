import pandas as pd
from extractor import extract_city
import csv
from tqdm import tqdm
import struct
import numpy as np
import os
import logging
import array
import re
import itertools
from stlmap import MapU64U32
import os.path
from bisect import bisect_left
from cryptography.fernet import Fernet


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def fromfile(a, fname, chunk_size=10**8):
    fsize = os.path.getsize(fname)
    assert fsize % a.itemsize == 0
    total = fsize // a.itemsize
    with open(fname, 'rb') as f:
        for i in range(0, total, chunk_size):
            dif = total - i
            n = chunk_size if dif >= chunk_size else dif
            a.fromfile(f, n)


def load_vector(fname, typecode='I'):
    a = array.array(typecode)
    fromfile(a, fname)
    return a


def to_numpy(buff: array.array, typecode=None, shape=None, order='C'):
    if shape is None:
        shape = len(buff)

    a = np.ndarray(shape,
                   buffer=memoryview(buff),
                   order=order,
                   dtype=buff.typecode)

    if typecode is None:
        typecode = buff.typecode
    if typecode != buff.typecode:
        a = a.astype(typecode)

    return a


def load_2d_vec(fname, nrows=2, typecode='L', order='C'):
    """
    buff=array.array('i', [1,2,3,4,5,6])

    order='C'
    v1 = [1,2,3]
    v2 = [4,5,6]

    order='F'
    v1 = [1,3,5]
    v2 = [2,4,6]

    a = [v1, v2]
    """

    buff = load_vector(fname, typecode=typecode)
    assert len(buff) % nrows == 0
    a = to_numpy(buff, typecode, (nrows, len(buff) // nrows), order)
    return a


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def binary_search(a, x):
    ix = bisect_left(a, x)
    if ix != len(a) and a[ix] == x:
        return ix
    raise ValueError


def iter_csv(fname: str):
    num_lines = int(os.popen('wc -l %s' % fname).read().split()[0])
    with open(fname) as f:
        next(f, None)  # header
        base_name = os.path.basename(fname)
        for row in tqdm(f, total=num_lines, desc='reading %s' % base_name):
            row = row.split(',')
            try:
                row[0] = int(row[0])
            except ValueError:
                continue
            yield row


def get_cities(fin_people, node2label, id2ix):
    city2label = {}
    for row in iter_csv(fin_people):
        _id = row[0]
        if id2ix:
            try:
                _id = id2ix[_id]
            except KeyError:
                continue
        elif _id >= len(node2label):
            continue

        city = extract_city(row)
        if city is not None:
            label = city2label.get(city)
            if label is None:
                label = len(city2label)
                city2label[city] = label
            node2label[_id] = label

    return city2label


def get_city_mask(geo, exclude_continent=True):
    """ selects cities starting with ci_ and cities which start with r1_ """

    sng = {'co_Kazakhstan', 'co_Belorussia', 'co_Ukraine', 'co_Azerbaijan'}

    cities = set()
    exclude = set()
    for s in geo:
        splited = s.split('.')
        country = splited[1] if exclude_continent else splited[0]
        if splited[-1].startswith('ci_'):
            cities.update([s])
            for i in range(1, len(splited)):
                exclude.update(['.'.join(splited[:i])])
        elif country in sng and splited[-1].startswith('r1_'):
            cities.update([s])

    cities -= exclude

    city_labels = []
    for i, s in enumerate(geo):
        if s in cities:
            city_labels.append(i)

    mask = np.zeros(len(geo), dtype=np.bool)
    mask[city_labels] = True
    return mask


def save_labels(base_dir, typecode='I'):
    assert typecode in ('I', 'L')

    fin_nodes = os.path.join(base_dir, 'nodes.bin')
    fin_people = os.path.join(base_dir, 'people.csv')
    fout_geo = os.path.join(base_dir, 'geography.csv')
    fout_labels = os.path.join(base_dir, 'labels.bin')
    fout_is_city = os.path.join(base_dir, 'is_city.bin')

    nodes = load_vector(fin_nodes, typecode)
    if typecode == 'I':
        nnodes = max(nodes) + 1
        id2ix = None
    else:
        nnodes = len(nodes)
        id2ix = MapU64U32()
        id2ix.reserve(nnodes)
        for ix, _id in enumerate(nodes):
            id2ix[_id] = ix

    node2label = array.array('i', (-1 for _ in range(nnodes)))
    city2label = get_cities(fin_people, node2label, id2ix)
    geo = []
    with open(fout_geo, 'w') as f:
        for city, i in city2label.items():
            geo.append(city)
            f.write('%s\n' % city)

    with open(fout_labels, 'wb') as f:
        node2label.tofile(f)

    node2label = to_numpy(node2label)
    mask = get_city_mask(geo)
    is_city = mask[node2label]
    is_city[node2label == -1] = False
    is_city = array.array('B', is_city)
    with open(fout_is_city, 'wb') as f:
        is_city.tofile(f)


def save_encrypted(password, key, filename):
    """
    Encrypts a password and saves it to file
    """
    encrypted_password = Fernet(key).encrypt(password.encode())
    with open(filename, "wb") as f:
        f.write(encrypted_password)


def decrypt_password(encrypted_password, key):
    """
    Decrypts an encrypted password
    """
    decrypted_password = Fernet(key).decrypt(encrypted_password)
    return decrypted_password.decode()


def send_email(sender='yourname@bk.ru',
               receivers=['myname@gmail.com'],
               subject='terminal notification',
               body=''):
    import smtplib
    from os.path import expanduser

    filename = '/usr/share/email.pass'
    if os.path.isfile(filename):
        # key = Fernet.generate_key()
        key = b'FMfaBOYAoB4I7DE4VCqLEJQMCsDB65VjogOkIvhbDsY='
        with open(filename, 'rb') as f:
            password = decrypt_password(f.read(), key)
    else:
        logging.error('file %s does not exist' % filename)
        return

    msg = "\r\n".join([
        "From: %s" % sender,
        "To: %s" % (','.join(receivers)),
        "Subject: %s" % subject,
        "%s" % body
    ])

    server = smtplib.SMTP('smtp.mail.ru:587')
    server.ehlo()
    server.starttls()

    server.login('guyos@bk.ru', password)
    server.sendmail(sender, receivers, msg)
    server.quit()
