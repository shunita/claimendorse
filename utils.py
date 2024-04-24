import math

import numpy as np
import scipy as sp
import pandas as pd
import os
import dotenv
from constants import *

def safe_median_pvalue_test(a, b):
    if len(set(a).union(set(b))) == 1:
    #In this case we want to make the pvalue to be 1.0
        a = [1]
        b = [2]
    return sp.stats.median_test(a, b)

mean_list = [np.mean, "mean", sp.stats.ttest_ind]
median_list = [np.median, "median", safe_median_pvalue_test]
count_list = [len, "count", lambda x,y: 0]
agg_type_to_agg_list = {"mean": mean_list, "median": median_list, "count": count_list}


def safe_translate(k, trans_dict):
    # k is either a field name (string) or a (field, value) tuple
    if k in trans_dict:
        return trans_dict[k]
    if type(k) == tuple:
        if k[1] is None or (type(k[1]) != str and math.isnan(k[1])):
            return "Not available"
        if type(k[1]) == str and k[1].isdigit() and (k[0], int(k[1])) in trans_dict:
            return trans_dict[(k[0], int(k[1]))]
        if (k[0], 'other') in trans_dict:
            return trans_dict[(k[0], 'other')]
        # 'other' option not available for this field - return the value as is (without mapping to a string)
        # print(f"Could not translate: {k}")
        return str(k[1])
    # k is not a (field, value) tuple - only a field name without a mapping. Return it as is.
    # print(f"Could not translate: {k}")
    return k


def less_than_cmp(val1, val2):
    return val1 < val2


def read_ACS_fields_data(year=2018):
    varlist = open(f"data/Folkstable/cepr_acs_{year}_varlist.log", "r").read()
    parts = varlist.split("-" * 73)
    titles = ["variable name", "storage type", "display format", "value label", "variable label"]
    data = parts[3].splitlines()[1:]
    processed = []
    for data_line in data:
        vname = data_line[:16].strip()
        storage_type = data_line[16:24].strip()
        display_format = data_line[24:35].strip()
        value_label = data_line[35:44].strip()
        var_label = data_line[44:].strip()
        if not vname and not storage_type and not display_format and not value_label:
            processed[-1]["variable label"] += f" {var_label}"
        elif not vname:
            processed[-1] = {"variable name": processed[-1]["variable name"], "storage type": storage_type,
                             "display format": display_format,
                             "value label": value_label, "variable label": var_label}
        else:
            processed.append({"variable name": vname, "storage type": storage_type, "display format": display_format,
                              "value label": value_label, "variable label": var_label})
    df = pd.DataFrame.from_records(processed)
    return df


def read_ACS_value_data():
    text = open('data/Folkstable/cepr_acs_2018_varlabels_plus.log', 'r').read().split("-" * 73)[2]
    lines = text.split("\n")
    field_to_value_dict = {}  # field name-> {field value code-> string}
    d = {}
    field = None
    for i, line in enumerate(lines):
        line_text = line.strip()
        if line_text == '':
            continue
        if line_text.endswith(":"):  # new field
            if field is not None:
                field_to_value_dict[field] = d
            field = line_text[:-1]
            d = {}
        elif line_text.startswith("> "):  # continuation of previous line
            d[k] += line_text[2:]
        else:
            try:
                parts = line_text.split()
                k = parts[0]
                v = " ".join(parts[1:])
                d[k] = v
            except:
                print(f"line num {i}: {line_text}")
                return
    if field is not None:
        field_to_value_dict[field] = d
    return field_to_value_dict


def make_translation_for_ACS(fields_list):
    # df = read_ACS_fields_data(year=2018)
    df = pd.read_csv('data/Folkstable/field_map.tsv', sep='\t')
    df['field label'] = df['field label'].apply(lambda s: s.strip('* '))
    trans = {}
    unmatched = []
    matched = 0
    unmatched_value_mapping = []
    field_to_value_dict = read_ACS_value_data()
    for field in fields_list:
        subset = df[df['field name'] == field]
        if len(subset) == 1:
            trans[field] = subset.iloc[0]['field label']
            matched += 1
        else:
            unmatched.append(field)
            continue
        # look for value mapping
        value_map_needed = subset.iloc[0]['field value map needed?']
        exclude = subset.iloc[0]['exclude?']
        if value_map_needed == 'no' or exclude == 'yes':
            continue
        elif value_map_needed == 'binary':
            if field == 'SEX':
                trans[(field, 1)] = 'Man'
                trans[(field, 2)] = 'Woman'
            else:
                trans[(field, 1)] = 'yes'
                trans[(field, 2)] = 'no'
        elif value_map_needed == 'binary(0,1)':
            trans[(field, 1)] = 'yes'
            trans[(field, 0)] = 'no'
        elif field.lower() in field_to_value_dict:
            value_to_meaning = field_to_value_dict[field.lower()]
            for v, meaning in value_to_meaning.items():
                value_for_trans = v
                if value_for_trans.isdigit():
                    value_for_trans = int(value_for_trans)
                trans[(field, value_for_trans)] = meaning
        else:  # mapping needed but not found
            unmatched_value_mapping.append(field)
    print(f"matched field names: {matched}/{len(fields_list)}. \nUnmatched: {unmatched}")
    print(f"missing value mapping for: {unmatched_value_mapping}")
    return trans


def prepare_for_regression(df, data_path, attrs):
    if "Folkstable" not in data_path:
        return df
    field_df = pd.read_csv('data/Folkstable/field_map.tsv', sep='\t')
    field_df['field label'] = field_df['field label'].apply(lambda s: s.strip('* '))
    for field in attrs:
        subset = field_df[field_df['field name'] == field]
        value_map_needed = subset.iloc[0]['field value map needed?']
        # is_binary = False
        if value_map_needed == 'binary':  # 1- yes, 2 - no
            binary_mapping = {1: 1, 2: -1}
            # is_binary = True
        elif value_map_needed == 'binary(0,1)':  # 1- Yes, 0 - No
            binary_mapping = {1: 1, 0: -1}
            # is_binary = True
        else:
            continue
        df[field] = df[field].apply(binary_mapping.get)
        df[field].fillna({field: 0}, inplace=True)
    return df




def check_ACS_fields():
    fields_list = ['RT', 'SERIALNO', 'DIVISION', 'SPORDER', 'PUMA', 'REGION', 'ST', 'ADJINC', 'PWGTP', 'AGEP', 'CIT', 'CITWP', 'COW', 'DDRS', 'DEAR', 'DEYE', 'DOUT', 'DPHY', 'DRAT', 'DRATX', 'DREM', 'ENG', 'FER', 'GCL', 'GCM', 'GCR', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7', 'INTP', 'JWMNP', 'JWRIP', 'JWTR', 'LANX', 'MAR', 'MARHD', 'MARHM', 'MARHT', 'MARHW', 'MARHYP', 'MIG', 'MIL', 'MLPA', 'MLPB', 'MLPCD', 'MLPE', 'MLPFG', 'MLPH', 'MLPI', 'MLPJ', 'MLPK', 'NWAB', 'NWAV', 'NWLA', 'NWLK', 'NWRE', 'OIP', 'PAP', 'RELP', 'RETP', 'SCH', 'SCHG', 'SCHL', 'SEMP', 'SEX', 'SSIP', 'SSP', 'WAGP', 'WKHP', 'WKL', 'WKW', 'WRK', 'YOEP', 'ANC', 'ANC1P', 'ANC2P', 'DECADE', 'DIS', 'DRIVESP', 'ESP', 'ESR', 'FOD1P', 'FOD2P', 'HICOV', 'HISP', 'INDP', 'JWAP', 'JWDP', 'LANP', 'MIGPUMA', 'MIGSP', 'MSP', 'NAICSP', 'NATIVITY', 'NOP', 'OC', 'OCCP', 'PAOC', 'PERNP', 'PINCP', 'POBP', 'POVPIP', 'POWPUMA', 'POWSP', 'PRIVCOV', 'PUBCOV', 'QTRBIR', 'RAC1P', 'RAC2P', 'RAC3P', 'RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACNUM', 'RACPI', 'RACSOR', 'RACWHT', 'RC', 'SCIENGP', 'SCIENGRLP', 'SFN', 'SFR', 'SOCP', 'VPS', 'WAOB', 'FAGEP', 'FANCP', 'FCITP', 'FCITWP', 'FCOWP', 'FDDRSP', 'FDEARP', 'FDEYEP', 'FDISP', 'FDOUTP', 'FDPHYP', 'FDRATP', 'FDRATXP', 'FDREMP', 'FENGP', 'FESRP', 'FFERP', 'FFODP', 'FGCLP', 'FGCMP', 'FGCRP', 'FHICOVP', 'FHINS1P', 'FHINS2P', 'FHINS3C', 'FHINS3P', 'FHINS4C', 'FHINS4P', 'FHINS5C', 'FHINS5P', 'FHINS6P', 'FHINS7P', 'FHISP', 'FINDP', 'FINTP', 'FJWDP', 'FJWMNP', 'FJWRIP', 'FJWTRP', 'FLANP', 'FLANXP', 'FMARP', 'FMARHDP', 'FMARHMP', 'FMARHTP', 'FMARHWP', 'FMARHYP', 'FMIGP', 'FMIGSP', 'FMILPP', 'FMILSP', 'FOCCP', 'FOIP', 'FPAP', 'FPERNP', 'FPINCP', 'FPOBP', 'FPOWSP', 'FPRIVCOVP', 'FPUBCOVP', 'FRACP', 'FRELP', 'FRETP', 'FSCHGP', 'FSCHLP', 'FSCHP', 'FSEMP', 'FSEXP', 'FSSIP', 'FSSP', 'FWAGP', 'FWKHP', 'FWKLP', 'FWKWP', 'FWRKP', 'FYOEP', 'PWGTP1', 'PWGTP2', 'PWGTP3', 'PWGTP4', 'PWGTP5', 'PWGTP6', 'PWGTP7', 'PWGTP8', 'PWGTP9', 'PWGTP10', 'PWGTP11', 'PWGTP12', 'PWGTP13', 'PWGTP14', 'PWGTP15', 'PWGTP16', 'PWGTP17', 'PWGTP18', 'PWGTP19', 'PWGTP20', 'PWGTP21', 'PWGTP22', 'PWGTP23', 'PWGTP24', 'PWGTP25', 'PWGTP26', 'PWGTP27', 'PWGTP28', 'PWGTP29', 'PWGTP30', 'PWGTP31', 'PWGTP32', 'PWGTP33', 'PWGTP34', 'PWGTP35', 'PWGTP36', 'PWGTP37', 'PWGTP38', 'PWGTP39', 'PWGTP40', 'PWGTP41', 'PWGTP42', 'PWGTP43', 'PWGTP44', 'PWGTP45', 'PWGTP46', 'PWGTP47', 'PWGTP48', 'PWGTP49', 'PWGTP50', 'PWGTP51', 'PWGTP52', 'PWGTP53', 'PWGTP54', 'PWGTP55', 'PWGTP56', 'PWGTP57', 'PWGTP58', 'PWGTP59', 'PWGTP60', 'PWGTP61', 'PWGTP62', 'PWGTP63', 'PWGTP64', 'PWGTP65', 'PWGTP66', 'PWGTP67', 'PWGTP68', 'PWGTP69', 'PWGTP70', 'PWGTP71', 'PWGTP72', 'PWGTP73', 'PWGTP74', 'PWGTP75', 'PWGTP76', 'PWGTP77', 'PWGTP78', 'PWGTP79', 'PWGTP80']
    df = read_ACS_fields_data(year=2018)
    field_to_value_dict = read_ACS_value_data()
    for field in fields_list:
        subset = df[df['variable name'] == field.lower()]
        line_to_print = f"{field}\t"
        if len(subset) == 1:
            line_to_print += f"{subset.iloc[0]['variable label']}\t"
        else:
            line_to_print += "\t"
        if field.lower() in field_to_value_dict:
            line_to_print += "yes"
        else:
            line_to_print += "no"
        print(line_to_print)


def estimate_is_numeric(data_df):
    c_to_num_values = {c: data_df[c].nunique() for c in data_df.columns}
    maybe_numeric = []
    for c, v in c_to_num_values.items():
        if v > 20:
            maybe_numeric.append(c)
    print(maybe_numeric)
    return maybe_numeric


def get_chatgpt_key():
    dotenv.load_dotenv(dotenv_path="data/chatgpt.env")
    key = os.getenv("CHATGPT_KEY")
    return key


def calc_t_stat(mean1, n1, s1, mean2, n2, s2):
    x1 = (s1 ** 2) / n1
    x2 = (s2 ** 2) / n2
    return (mean1-mean2)/np.sqrt(x1+x2)


def calc_mean_diff_degrees_freedom(n1, s1, n2, s2):
    x1 = (s1 ** 2) / n1
    x2 = (s2 ** 2) / n2
    return ((x1+x2)**2) / (x1*x1/(n1-1) + x2*x2/(n2-1))


def calc_chi_squared_stat(v1_over, v2_over, v1_under, v2_under, yates=True):
    over = v1_over + v2_over
    under = v1_under + v2_under
    v1 = v1_under + v1_over
    v2 = v2_under + v2_over
    total = (over + under)*1.0
    e1_over = over*v1/total
    e1_under = under*v1/total
    e2_over = over*v2/total
    e2_under = under*v2/total
    d = 0.5 if yates else 0
    chi_sq_stat = ((np.abs(v1_over - e1_over) - d) ** 2) / e1_over + \
                  ((np.abs(v1_under - e1_under) - d) ** 2) / e1_under + \
                  ((np.abs(v2_over - e2_over) - d) ** 2) / e2_over + \
                  ((np.abs(v2_under - e2_under) - d) ** 2) / e2_under
    return chi_sq_stat


def calc_median_diff_test(v1_over, v2_over, v1_under, v2_under):
    # as formulated in Sprent 2007
    nom = (v1_over + v1_under + v2_over + v2_under)*(2*v1_over - (v1_over + v1_under))**2
    denom = (v1_over + v1_under)*(v2_over + v2_under)
    return nom/denom


def get_attr_and_value_fields(num_atoms):
    attr_value_fields = []
    for i in range(num_atoms):
        attr_value_fields.append(f"Attr{i + 1}")
        attr_value_fields.append(f"Value{i + 1}")
    return attr_value_fields


def should_exclude(column_tuple, exclude_list):
    for c in column_tuple:
        if c in exclude_list:
            return True
    return False


def find_index_in_list(element, lst):
    if element in lst:
        return lst.index(element)
    return -1


def unite_attr_names_to_tuple_field(row, max_atoms):
    res = [row['Attr1']]
    for i in range(1, max_atoms):
        a = f'Attr{i+1}'
        if a in row:
            next_attr = row[a]
            if type(next_attr) == str:
                res.append(next_attr)
    return tuple(res)


def measure_score_recall(reference_result, result_so_far, k, score_name=DF_METRICS_AVERAGE):
    reference_result = reference_result.sort_values(by=score_name, ascending=False)
    top_k_total_score = reference_result[score_name].head(k).sum()

    result_subset = result_so_far.sort_values(by=score_name, ascending=False).head(k)
    result_top_k_total_score = result_subset[score_name].sum()
    score_recall = result_top_k_total_score / top_k_total_score
    return score_recall


def safe_is_nan(x):
    if x is None:
        return True
    if type(x) == str:
        return False
    return np.isnan(x)