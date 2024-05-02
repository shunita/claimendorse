import os
import random
from folktables import BasicProblem, ACSDataSource, ACSEmployment, adult_filter
from ClaimEndorseFunctions import *
from analyze_output import *

ACSRawIncome = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
        'PINCP',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

codes_df = pd.read_csv("data/Folkstable/Code-Translation.csv")
ACSIncome_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
    "DIS": {1.0: "With a disability", 2.0: "Without a disability"},
    "CIT": {1.0: "Born in the U.S.",
            2.0: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
            3.0: "Born abroad of American parent(s)", 4.0: "U.S. citizen by naturalization",
            5.0: "Not a citizen of the U.S."}
}


def add_country_codes_translation():
    place_of_birth_df = pd.read_csv("data/Folkstable/Place_of_birth.csv")
    country_translation_dict = {}
    for index in place_of_birth_df["Code"].index:
        # print(index)
        combined_str = place_of_birth_df.iloc[index]["Code"]
        if pd.isna(combined_str):
            continue
        split_list = combined_str.split(".")
        code = int(split_list[0].strip())
        country = split_list[1]
        country_translation_dict[code] = country
    ACSIncome_categories["POBP"] = country_translation_dict


def add_RELP_translation():
    RELP_df = pd.read_csv("data/Folkstable/Relation to House Owner.csv")
    RELP_translation_dict = {}
    for index in RELP_df["Relationship"].index:
        # print(index)
        combined_str = RELP_df.iloc[index]["Relationship"]
        if pd.isna(combined_str):
            continue
        split_list = combined_str.split(".")
        code = int(split_list[0].strip())
        country = split_list[1]
        RELP_translation_dict[code] = country

    # print(RELP_translation_dict)
    ACSIncome_categories["RELP"] = RELP_translation_dict


def get_CA_2018_income_data():
    add_country_codes_translation()
    add_RELP_translation()
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    ca_features, ca_labels, _ = ACSRawIncome.df_to_pandas(ca_data,
                                                          # categories=ACSIncome_categories
                                                          )
    ca_features["Income>50k"] = ca_labels
    ca_features["Work Rate"]=ca_features["PINCP"]/ca_features["WKHP"]
    return ca_features


def get_CA_2018_All():
    add_country_codes_translation()
    add_RELP_translation()
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    # for k in ACSIncome_categories:
    #     ca_data[k] = ca_data[k].apply(ACSIncome_categories[k].get)
    return ca_data

def get_Multiple_States_2018_All(state_code_list):
    #Last Done for states:["CA","TX","FL","NY","PA","IL","OH"]
    add_country_codes_translation()
    add_RELP_translation()
    df_list = []
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    for state_code in state_code_list:
        print(state_code)
        state_file_string=f"data/Folkstable/2018_{state_code}_data.csv"
        if not os.path.exists(state_file_string):
            state_data = data_source.get_data(states=[state_code], download=True)
            state_data.to_csv()
        else:
            state_data = pd.read_csv(state_file_string)
        #print(state_data.columns)
        df_list.append(state_data)
    # for k in ACSIncome_categories:
    #     ca_data[k] = ca_data[k].apply(ACSIncome_categories[k].get)
    return pd.concat(df_list).reset_index(drop=True, inplace=False)



def get_CA_2018_employment_data():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    ca_features, ca_labels, _ = ACSEmployment.df_to_pandas(ca_data, categories=ACSIncome_categories)
    #print(ca_features)
    #print(ca_labels)


def generalize_all_occp_codes(df):
    df["Generalized OCCP"] = df["OCCP"].map(generalize_occp_code)
    return df


def generalize_occp_code(code):
    min_var = "min code"
    max_var = "max code"
    for index in codes_df.index:
        min_code = codes_df[min_var][index]
        max_code = codes_df[max_var][index]
        if code >= min_code and code <= max_code:
            return codes_df["Industry"][index]
        #print(f"{codes_df[min_var][index]},{codes_df[max_var][index]}")
    return "Other"


def combine_highschool_education(df):
    org_str1="Regular high school diploma"
    org_str2="GED or alternative credential"
    new_str="Highschool Education Equivalent"
    df["combined_education"]=df["SCHL"].copy()
    #df["combined_education"] = np.where(df["combined_education"] == secondary_ed_org, secondary_ed_new, df["EdLevel"])
    #df[df["combined_education"].isin(["Regular high school diploma","GED or alternative credential"])]="Highschool Education Equivalent"
    # df[df["combined_education"]=="Regular high school diploma" ]["combined_education"]="Highschool Education Equivalent"
    # df[df["combined_education"] == "GED or alternative credential"]["combined_education"] =
    df["combined_education"] = np.where(df["combined_education"] ==  org_str1,new_str, df["combined_education"])
    df["combined_education"] = np.where(df["combined_education"] == org_str2, new_str, df["combined_education"])
    return df
    #print(df["combined_education"])


def translate_and_take_prefix(attr, v, trans_dict):
    s = safe_translate((attr, v), trans_dict)
    if type(s) == str and '-' in s:
        return s.split('-')[0]


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    exclude_list1 = ["SERIALNO", "PWGTP"] + [f"PWGTP{x}" for x in range(1, 81)]
    exclude_for_PINCP = ["PINCP", "INTP", "OIP", "PAP", "RETP", "SEMP", "SSIP", "SSP", "WAGP", "PERNP", "POVPIP", "ADJINC",
                         "FPINCP", "FINTP", "FOIP", "FPAP", "FRETP", "FSEMP", "FSSIP", "FSSP", "FWAGP", "FPERNP"
                         ]
    both_exclude_lists = exclude_list1 + exclude_for_PINCP
    #is_numeric = ["AGEP", "WKHP", "PINCP"]
    is_numeric = ['PINCP', "AGEP", 'CITWP', 'JWMNP', 'MARHYP', 'WKHP', 'YOEP', 'JWAP', 'JWDP']
    is_bucket = []

    ######### DATA CREATION ################
    # df=get_Multiple_States_2018_All(["CA","TX","FL","NY","PA","IL","OH"])
    # df.to_csv("data/Folkstable/SevenStates/Seven_States.csv")

    # df = pd.read_csv("data/Folkstable/SevenStates/Seven_States.csv", index_col=0)
    # trans_dict = make_translation_for_ACS(df.columns)
    # df['OCCP_grouped'] = df['OCCP'].apply(lambda v: translate_and_take_prefix('OCCP', v, trans_dict))
    # df['NAICSP_grouped'] = df['NAICSP'].apply(lambda v: translate_and_take_prefix('NAICSP', v, trans_dict))
    # df.to_csv("data/Folkstable/SevenStates/Seven_States_grouped.csv")

    # df = pd.read_csv(DATAFRAME_PATH, index_col=0)
    # df_to_sql_DB(df)


    # create_filtered_csv(df, numeric_fields=is_numeric, bucket_fields=is_bucket, exclude_list=both_exclude_lists,
    #                     date_fields_dict={}, output_csv_name="data/Folkstable/SevenStates/Discretized_Seven_States.csv",
    #                     attribute_info_file_name="data/Folkstable/Discretized_Seven_States_attributes.txt",
    #                     create_db_flag=True)

    #######################################

    ################## CHERRYPICKING ###################

    if RUN_ACTUAL_CP:
        df = pd.read_csv(DATAFRAME_PATH, index_col=0)
        trans_dict = make_translation_for_ACS(df.columns)
        allocation_flags = [col for col in df.columns if "allocation flag" in trans_dict[col]]
        exclude_list = both_exclude_lists + allocation_flags

        # shuffle_and_save(df, exclude_list, is_numeric, trans_dict, start_time, iters=3)

        ######## main quality experiment ########
        methods = ['random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'original_order', 
                'ALL_TOP_K_MERGED', 'ALL_TOP_K_SERIAL',
                '0.01sample:1', '0.01sample:2', '0.01sample:3',
                '0.05sample:1', '0.05sample:2', '0.05sample:3',
                '0.10sample:1', '0.10sample:2', '0.10sample:3']
        run_multiple_methods_for_query(TARGET_ATTR, GRP_ATTR, COMPARE_LIST, AGG_TYPE,
                                       df, exclude_list, is_numeric, trans_dict, methods, stop_at_recall=False)
        
        ################# pred level ##########################
        #pred_level_cherrypicking(df, exclude_list, is_numeric, trans_dict)

        ########### sample size experiment #####################
        #run_sample_guided_experiment([0.01], 1, df, exclude_list, is_numeric, trans_dict)

        ############## num tuples experiment #########################
        methods = ['original_order', 'random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'ALL_TOP_K_MERGED',
                   '0.01sample:1', '0.01sample:2', '0.01sample:3',
                   'ALL_TOP_K_SERIAL']
        sample_sizes = range(100000, 1000001, 100000)
        # num_tuples_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, trans_dict, methods,
        #                                 stop_at_recall=True)

        ############## num columns experiment #########################
        col_sample_sizes = range(10, 101, 10)
        # num_columns_vs_time_for_full_run(col_sample_sizes, df, exclude_list, is_numeric, trans_dict, methods, column_subsets=None)

        ################## sensitivity to k ########################
        # run the serial top k method for each k value.
        sensitivity_to_k(range(100, 1001, 100),  # K values
                         df, exclude_list, is_numeric, trans_dict,
                         reference_path="data/Folkstable/SevenStates/results/ACS7_numeric_mean_2atoms_F_gt_M_original_order_guided_reference.csv")

    end_time = datetime.datetime.now()

    print(f"total runtime= {(end_time-start_time).seconds}")


