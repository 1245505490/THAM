from collections import OrderedDict

def encode_code(patient_admission, admission_codes):
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission['adm_id']]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map) + 1
    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map


def encode_drug(patient_admission, admission_drugs):
    drug_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            hadm = admission['adm_id']
            if hadm not in admission_drugs:
                continue
            drugs = admission_drugs[hadm]
            for drug in drugs:
                if drug not in drug_map:
                    drug_map[drug] = len(drug_map)
    admission_drugs_encoded = {
        admission_id: list(set(drug_map[drug] for drug in drugs))
        for admission_id, drugs in admission_drugs.items()
    }
    return admission_drugs_encoded, drug_map

