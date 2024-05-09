from datetime import datetime
import numpy as np


def split_patients(patient_admission, admission_codes, code_map, train_num, valid_num, seed=18):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission['adm_id']]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num,
                  max_code_num_in_a_visit):
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    times = []
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        t = [0] * max_admission_num
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            x[i, k, :len(codes)] = codes
            t[k] = datetime.fromtimestamp(datetime.timestamp(admission['adm_id']))
        codes = np.array(admission_codes_encoded[admissions[-1]['adm_id']]) - 1
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
        times.append(t)
    times = np.array(times)
    intervals = np.zeros_like(times, dtype=float)
    for i in range(len(times)):
        for j in range(lens[i]):
            if j > 0:
                intervals[i][j] = (times[i][j] - times[i][j - 1]).days
        intervals[i][0] = 0
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens, intervals


def build_heart_failure_y(hf_prefix: str, codes_y: np.ndarray, code_map: dict) -> np.ndarray:
    print('building train/valid/test heart failure labels ...')
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list - 1] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y
