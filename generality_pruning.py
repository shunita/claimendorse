import itertools
from tqdm import tqdm
import pandas as pd


def make_attr_set_without_nans(sequence):
    return set([x for x in sequence if type(x) == str])


def prune_by_generality(res_df, max_atoms):
    attr_cols = [f"Attr{i + 1}" for i in range(max_atoms)]
    # 1. get the set of attrs in each row, and size of set
    res_df['Attr_set'] = res_df[attr_cols].apply(make_attr_set_without_nans, axis=1)
    res_df['Attr_set_size'] = res_df['Attr_set'].apply(len)
    # 2. start from smallest sets
    res_df = res_df.sort_values(by='Attr_set_size')
    # prune all sets that strictly contain the current one.
    # remaining = res_df.copy()
    attr_sets = res_df['Attr_set'].values
    i = 0
    while i < len(attr_sets):
        print(len(attr_sets))
        attr_set = attr_sets[i]
        attr_sets = [s for s in attr_sets if not (len(attr_set) < len(s) and attr_set.issubset(s))]
        i += 1
    res_df['keep_by_generality'] = res_df.Attr_set.isin(attr_sets)
    return res_df


def is_next_level_needed(res_df, split_cols, max_atoms):
    multi_atom_combinations = [set(x) for x in itertools.combinations(split_cols, max_atoms+1)]
    print(f"found {len(multi_atom_combinations)}")
    most_gen_sets = res_df[res_df['keep_by_generality']]['Attr_set'].values
    most_gen_sets = [set(x) for x in set([tuple(s) for s in most_gen_sets])]
    # i = 0
    for attr_set in tqdm(most_gen_sets):
        # print(len(multi_atom_combinations))
        # before = len(multi_atom_combinations)
        # attr_set = most_gen_sets[i]
        multi_atom_combinations = [s for s in multi_atom_combinations if not attr_set.issubset(s)]
        #after = len(multi_atom_combinations)
        # if after != before:
        #     print(after)
        # i += 1
    return multi_atom_combinations


if __name__ == '__main__':
    res = pd.read_csv("data/Folkstable/SevenStates/results/ACS7_numeric_mean_2atoms_F_gt_M_original_order_guided_reference.csv", index_col=0)
    res = prune_by_generality(res, 2)
    # ACS columns, minus the excluded
    cols = ['RT', 'DIVISION', 'SPORDER', 'PUMA', 'REGION', 'ST', 'AGEP', 'CIT', 'CITWP', 'COW', 'DDRS', 'DEAR', 'DEYE', 'DOUT', 'DPHY', 'DRAT', 'DRATX', 'DREM', 'ENG', 'FER', 'GCL', 'GCM', 'GCR', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7', 'JWMNP', 'JWRIP', 'JWTR', 'LANX', 'MAR', 'MARHD', 'MARHM', 'MARHT', 'MARHW', 'MARHYP', 'MIG', 'MIL', 'MLPA', 'MLPB', 'MLPCD', 'MLPE', 'MLPFG', 'MLPH', 'MLPI', 'MLPJ', 'MLPK', 'NWAB', 'NWAV', 'NWLA', 'NWLK', 'NWRE', 'RELP', 'SCH', 'SCHG', 'SCHL', 'SEX', 'WKHP', 'WKL', 'WKW', 'WRK', 'YOEP', 'ANC', 'ANC1P', 'ANC2P', 'DECADE', 'DIS', 'DRIVESP', 'ESP', 'ESR', 'FOD1P', 'FOD2P', 'HICOV', 'HISP', 'INDP', 'JWAP', 'JWDP', 'LANP', 'MIGPUMA', 'MIGSP', 'MSP', 'NAICSP', 'NATIVITY', 'NOP', 'OC', 'OCCP', 'PAOC', 'POBP', 'POWPUMA', 'POWSP', 'PRIVCOV', 'PUBCOV', 'QTRBIR', 'RAC1P', 'RAC2P', 'RAC3P', 'RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACNUM', 'RACPI', 'RACSOR', 'RACWHT', 'RC', 'SCIENGP', 'SCIENGRLP', 'SFN', 'SFR', 'SOCP', 'VPS', 'WAOB', 'FHINS3C', 'FHINS4C', 'FHINS5C', 'OCCP_grouped', 'NAICSP_grouped']
    mac = is_next_level_needed(res, cols, 2)
    mac = [tuple(sorted(list(x))) for x in mac]
