
import csv
import json
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import shutil

def read_data():
    files = ['click', 'collect', 'cart', 'buy']
    for file in files:
        with open('trn_' + file, 'rb') as fs, open('./taobao/' + file + '.txt', 'w') as wt:
            mat = pickle.load(fs)
            indic = mat.nonzero()
            coordinates = list(zip(indic[0], indic[1]))
            for i in coordinates:
                wt.write(str(i[0] + 1) + ' ' + str(i[1] + 1) + '\n')
            print('x')

def read_test():
    with open('tst_int', 'rb') as fs, open('./taobao/test.txt', 'w') as wt:
        mat = np.array(pickle.load(fs))
        # items = (mat != None)
        users = np.reshape(np.argwhere(mat != None), -1)
        for u in users:
            # print(u, mat[u])
            for i in mat[u]:
                wt.write(str(u + 1) + ' ' + str(i + 1) + '\n')

def split_items():
    with open('buy_old.txt') as fs, open('buy.txt', 'w') as w1, open('validation.txt', 'w') as w2:
        dict = {}
        data = fs.readlines()
        for line in data:
            line = line.strip('\n').strip().split()
            u, i = line[0], line[1]
            i = int(i)
            if str(u) in dict:
                dict[str(u)].append(int(i))
            else:
                dict[str(u)] = [i]
        buy = {}
        val = {}
        for k, v in dict.items():
            if len(v) > 2:
                buy[k] = v[: -1]
                val[k] = v[-1]
            else:
                buy[k] = v
        for k, v in buy.items():
            for i in v:
                w1.write(k + ' ' + str(i) + '\n')
        for k, v in val.items():
            w2.write(k + ' ' + str(v) + '\n')


def split_test():
    with open('taobao/buy0.txt') as fs, open('taobao/test0.txt') as ft:
        dict = {}
        data = fs.readlines()
        for line in data:
            line = line.strip('\n').strip().split()
            u, i = line[0], line[1]
            i = int(i)
            if str(u) in dict:
                dict[str(u)].append(i)
            else:
                dict[str(u)] = [i]
        test = ft.readlines()
        for line in test:
            line = line.strip('\n').strip().split()
            u, i = line[0], line[1]
            dict[u].append(int(i))

    with open('taobao/buy_old.txt', 'w') as b, open('taobao/test.txt', 'w') as t, open('taobao/validation.txt', 'w') as v:
        for k, value in dict.items():
            if len(value) > 4:
                i = value.pop()
                t.write(k + ' ' + str(i) + '\n')
                i = value.pop()
                v.write(k + ' ' + str(i) + '\n')
            elif len(value) > 3:
                i = value.pop()
                v.write(k + ' ' + str(i) + '\n')
            for i in value:
                b.write(k + ' ' + str(i) + '\n')


def get_test_valid_list(path, files):
    with open(path + 'count.txt') as r:
        data = json.load(r)
        n_user = data['user']
    for file in files:
        with open(path + file + '.txt') as fs, open(path + file + '_int', 'wb') as w:
            dict = {}
            data = fs.readlines()
            for line in data:
                line = line.strip('\n').strip().split()
                u, i = line[0], line[1]
                i = int(i)
                if str(u) in dict:
                    dict[str(u)].append(i)
                else:
                    dict[str(u)] = [i]

            items = []
            for i in range(1, n_user + 1):
                item = dict.get(str(i), None)
                if item is not None:
                    item = item[0] - 1
                items.append(item)
            pickle.dump(items, w)


def get_csr_matris(path, files):
    with open(path + 'count.txt') as r:
        data = json.load(r)
        n_user = data['user']
        n_item = data['item']
    for file in files:
        with open(path + file + '.txt') as r, open(path + 'trn_' + file, 'wb') as w:
            data = r.readlines()
            row, col = [], []
            for line in data:
                u, v = line.strip('\n').strip().split()
                u, v = int(u), int(v)
                row.append(u - 1)
                col.append(v - 1)
            row = np.array(row)
            col = np.array(col)
            values = np.ones(len(row), dtype=float)
            mat = sp.csr_matrix((values, (row, col)), shape=(n_user, n_item))
            pickle.dump(mat, w)


def generate_dict(path, file):
    user_interaction = {}
    with open(os.path.join(path, file)) as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            user, item = int(user), int(item)

            if user not in user_interaction:
                user_interaction[user] = [item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
    return user_interaction


def generate_interact(path):
    buy_dict = generate_dict(path, 'buy.txt')
    with open(os.path.join(path, 'buy_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(buy_dict))

    cart_dict = generate_dict(path, 'cart.txt')
    with open(os.path.join(path, 'cart_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(cart_dict))

    click_dict = generate_dict(path, 'view.txt')
    with open(os.path.join(path, 'view_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(click_dict))


    for dic in [buy_dict, cart_dict]:
        for k, v in dic.items():
            if k not in click_dict:
                click_dict[k] = v
            item = click_dict[k]
            item.extend(v)
    for k, v in click_dict.items():
        item = click_dict[k]
        item = list(set(item))
        click_dict[k] = sorted(item)

    shutil.copyfile('buy_dict.txt', 'train_dict.txt')

    validation_dict = generate_dict(path, 'validation.txt')
    with open(os.path.join(path, 'validation_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(validation_dict))

    test_dict = generate_dict(path, 'test.txt')
    with open(os.path.join(path, 'test_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict))


def generate_all_interact(path):
    all_dict = {}
    files = ['buy', 'cart', 'view']
    for file in files:
        with open(os.path.join(path, file+'_dict.txt')) as r:
            data = json.load(r)
            for k, v in data.items():
                if all_dict.get(k, None) is None:
                    all_dict[k] = v
                else:
                    total = all_dict[k]
                    total.extend(v)
                    all_dict[k] = sorted(list(set(total)))
        with open(os.path.join(path, 'all.txt'), 'w') as w1, open(os.path.join(path, 'all_dict.txt'), 'w') as w2:
            for k, v in all_dict.items():
                for i in v:
                    w1.write('{} {}\n'.format(int(k), i))
            w2.write(json.dumps(all_dict))


def pos_sampling(path):
    behaviors = ['buy', 'cart', 'view']
    with open(os.path.join(path, 'pos_sampling.txt'), 'w') as f:
        for index, file in enumerate(behaviors):
            with open(os.path.join(path, file + '_dict.txt'), encoding='utf-8') as r:
                tmp_dict = json.load(r)
                for k in tmp_dict:
                    for v in tmp_dict[k]:
                        f.write('{} {} {} 1\n'.format(k, v, index))

def item_inter(path, behaviors):
    for behavior in behaviors:
        all_inter = set()
        with open(os.path.join(path, behavior + '_dict.txt')) as f:
            data = json.load(f)
            for v in data.values():
                i = len(v)
                m = 0
                while m < i:
                    n = 0
                    while n < i:
                        all_inter.add((v[m], v[n]))
                        n += 1
                    m += 1
        row = []
        col = []
        for item in all_inter:
            row.append(item[0])
            col.append(item[1])
        indict = len(row)
        item_graph = sp.coo_matrix((np.ones(indict), (row, col)), shape=[39494, 39494])
        item_graph_degree = item_graph.toarray().sum(axis=0).reshape(-1, 1)
        info = {'row': row, 'col': col, 'degree': item_graph_degree.tolist()}
        with open(os.path.join(path, behavior+'_item_graph.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(info))

if __name__ == '__main__':
    path = './'
    # files = ['click', 'collect', 'cart', 'buy']
    files = ['buy', 'cart', 'view']
    # read_data()
    # read_test()
    # split_test()
    # split_items()
    # get_csr_matris(path, files)
    # files = ['validation', 'test']
    # get_test_valid_list(path, files)



    generate_interact(path)
    generate_all_interact(path)
    pos_sampling(path)
    # item_inter(path, files)



    # with open('validation_int', 'rb') as fs:
    #     mat = pickle.load(fs)
    #     indic = mat.nonzero()
    #     coordinates = list(zip(indic[0], indic[1]))
    #     print('x')
