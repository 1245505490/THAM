import os
import numpy as np


def parse_icd9_range(range_: str) -> (str, str, int, int):
    # 001 - 009
    # 282
    # V01 - V09
    # E840 - E845
    ranges = range_.lstrip().split('-')
    # 001,009
    # V01,V09
    # E840,E845
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        # start:01 end:09
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        # start:840 end:845
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            # start: 282 ,end : 283
            start = int(ranges[0])
            end = start + 1
        else:
            # start:1,end:9
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (1, 1, 1)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    level4 = 1
    code_level = dict()
    for code in code_map:
        three_level_code = code.split('.')[0]
        if three_level_code in three_level_dict:
            three_level = three_level_dict[three_level_code]
            code_level[code] = three_level + [level4]
            level4 += 1
        else:
            print(three_level_code)
            code_level[code] = [0, 0, 0, 0]

    code_level_matrix = np.zeros((len(code_map) + 1, 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix


def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num, threshold=0.01):
    print('generating code code adjacent matrix ...')
    n = code_num
    adj = np.zeros((n + 1, n + 1), dtype=float)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] += 1
                    adj[c_j, c_i] += 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    norm_adj = normalize_adj(adj)
    a = norm_adj < threshold
    b = adj.sum(axis=-1, keepdims=True) > (1 / threshold)
    adj[np.logical_and(a, b)] = 0
    return adj


def generate_drug_code_adjacent(pids, patient_admission, admission_codes_encoded, admission_drugs_encoded, code_num,
                                drug_num):
    adj = np.zeros((drug_num, code_num + 1), dtype=int)
    print('generating code and drug adjacent matrix ...')
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        for admission in patient_admission[pid]:
            hadm = admission['adm_id']
            if hadm not in admission_drugs_encoded:
                continue
            drugs = admission_drugs_encoded[admission['adm_id']]
            codes = admission_codes_encoded[admission['adm_id']]
            for row in range(len(drugs)):
                for col in range(len(codes)):
                    c_i = drugs[row]
                    c_j = codes[col]
                    adj[c_i, c_j] += 1

    print('\r\t%d / %d' % (len(pids), len(pids)))
    return adj


def normalize_adj(adj):
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result
