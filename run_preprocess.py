import os
import _pickle as pickle

from preprocess.auxiliary import *
from preprocess.build_dataset import split_patients, build_code_xy, build_heart_failure_y
from preprocess.encode import encode_code, encode_drug
from preprocess import save_sparse, save_data

if __name__ == '__main__':
    conf = {
        'mimic3': {
            'train_num': 6000,
            'valid_num': 500,
            'threshold': 0.01
        },
        'mimic4': {
            'train_num': 8000,
            'valid_num': 1000,
            'threshold': 0.01,
            'sample_num': 10000
        }
    }
    data_path = 'data'
    dataset = 'mimic4'
    seed = 18

    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    standard_path = os.path.join(dataset_path, 'standard')
    parsed_path = os.path.join(dataset_path, 'parsed')
    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()

    parsed_path = os.path.join(dataset_path, 'parsed')
    patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
    admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
    admission_drugs = pickle.load(open(os.path.join(parsed_path, 'admission_drugs.pkl'), 'rb'))

    patient_num = len(patient_admission)
    # 最大就诊次数
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    # 平均就诊次数
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    # 单次就诊最多疾病
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    # 就诊平均疾病数
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)

    max_visit_drug_num = max([len(codes) for codes in admission_drugs.values()])
    avg_visit_drug_num = sum([len(codes) for codes in admission_drugs.values()]) / len(admission_drugs)

    print('patient num: %d' % patient_num)
    print('max admission num: %d' % max_admission_num)
    print('mean admission num: %.2f' % avg_admission_num)
    print('max code num in an admission: %d' % max_visit_code_num)
    print('mean code num in an admission: %.2f' % avg_visit_code_num)
    print('max drug num in an admission: %d' % max_visit_drug_num)
    print('mean drug num in an admission: %.2f' % avg_visit_drug_num)
    print('encoding code ...')
    admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
    admission_drugs_encoded, drug_map = encode_drug(patient_admission, admission_drugs)
    code_num = len(code_map)
    print('There are %d codes' % code_num)
    drug_num = len(drug_map)
    print('There are %d drugs' % drug_num)

    train_pids, valid_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=conf[dataset]['train_num'],
        valid_num=conf[dataset]['valid_num']
    )
    print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))

    code_code_adj = generate_code_code_adjacent(train_pids, patient_admission, admission_codes_encoded, code_num)
    drug_code_adj = generate_drug_code_adjacent(train_pids, patient_admission, admission_codes_encoded,
                                                admission_drugs_encoded, code_num, drug_num)
    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num, max_visit_code_num]
    (train_code_x, train_codes_y, train_visit_lens, train_intervals) = build_code_xy(train_pids,
                                                                                     *common_args)
    (valid_code_x, valid_codes_y, valid_visit_lens, valid_intervals) = build_code_xy(valid_pids,
                                                                                     *common_args)
    (test_code_x, test_codes_y, test_visit_lens, test_intervals) = build_code_xy(test_pids,
                                                                                 *common_args)
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)
    code_levels = generate_code_levels(data_path, code_map)
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    pickle.dump({
        'code_levels': code_levels,
        'drug_code_adj': drug_code_adj,
        'code_code_adj': code_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))

    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump(admission_drugs_encoded, open(os.path.join(encoded_path, 'drugs_encoded.pkl'), 'wb'))
    pickle.dump(drug_map, open(os.path.join(encoded_path, 'drug_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)
    print('\tsaving training data')
    save_data(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y, train_intervals)
    print('\tsaving valid data')
    save_data(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y, valid_intervals)
    print('\tsaving test data')
    save_data(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y, test_intervals)
