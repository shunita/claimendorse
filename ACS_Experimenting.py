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


def female_greater_than_male_COW(df,pvalue_filter=False):
    return cherrypick_proportions(df=df, subgroup_attr="SEX", subgroup_vals=["Male", "Female"], groupby_attr="COW",
                       cmpr_attr="Income>50k", cmpr_list=[less_than_cmp, True],pvalue_filter=pvalue_filter)


def female_greater_than_male_generalized_occp(df,pvalue_filter=False):
    return cherrypick_proportions(df=df, subgroup_attr="SEX", subgroup_vals=["Male", "Female"], groupby_attr="Generalized OCCP",
                           cmpr_attr="Income>50k", cmpr_list=[less_than_cmp, True],pvalue_filter=pvalue_filter)


def female_greater_than_male_generalized_occp_salary(df,pvalue_filter=False):
    # return multichoice_attribute_cherrypicking(df=df, subgroup_attr="SEX", subgroup_vals=["Male", "Female"], groupby_attr="Generalized OCCP",
    #                        cmpr_attr="PINCP", cmpr_list=[less_than_cmp, True],pvalue_filter=pvalue_filter)
    return multichoice_attribute_cherrypicking(df=df, split_attr="Generalized OCCP", grp_attr="SEX",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "Male", "Female"],
                                               pvalue_filter=pvalue_filter)


def female_greater_than_male_COW_salary(df,pvalue_filter=False):
    # return multichoice_attribute_cherrypicking(df=df, subgroup_attr="SEX", subgroup_vals=["Male", "Female"], groupby_attr="Generalized OCCP",
    #                        cmpr_attr="PINCP", cmpr_list=[less_than_cmp, True],pvalue_filter=pvalue_filter)
    return multichoice_attribute_cherrypicking(df=df, split_attr="COW", grp_attr="SEX",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "Male", "Female"],
                                               pvalue_filter=pvalue_filter)


def black_greater_than_white_generalized_OCCP_salary(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Generalized OCCP", grp_attr="RAC1P",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "White alone", "Black or African American alone"],
                                               pvalue_filter=pvalue_filter)


def black_greater_than_white_COW_salary(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="COW", grp_attr="RAC1P",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "White alone", "Black or African American alone"],
                                               pvalue_filter=pvalue_filter)


def highschool_greater_than_bachleors_generalized_OCCP_salary(df,pvalue_filter=False):
        return multichoice_attribute_cherrypicking(df=df, split_attr="Generalized OCCP", grp_attr="SCHL",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "Bachelor's degree","Regular high school diploma"],
                                               pvalue_filter=pvalue_filter)


def bachelors_greater_than_masters_generalized_OCCP_salary(df, pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Generalized OCCP", grp_attr="SCHL",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "Master's degree",
                                                             "Bachelor's degree"],
                                               pvalue_filter=pvalue_filter)


def full_cherrypicking_highschool_greater_than_bachelors_salary(df,exclude_list, is_numeric,pvalue_filter=False):
    full_multichoice_attribute_cherrypicking(df=df, grp_attr="combined_education", target_attr="PINCP",
                                             compare_list=[less_than_cmp, "Bachelor's degree", "Highschool Education Equivalent"],
                                             aggr_list=mean_list,
                                             exclude_list=exclude_list, is_numeric=is_numeric, pvalue_filter=pvalue_filter)


def female_greater_than_male_EdLevel_salary(df, pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="SCHL", grp_attr="SEX",
                                               cmp_attr="PINCP",
                                               compare_list=[less_than_cmp, "Male",
                                                             "Female"],
                                               pvalue_filter=pvalue_filter)


def female_greater_than_male_generalized_occp_Work_hours(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Generalized OCCP", grp_attr="SEX",
                                               cmp_attr="WKHP",
                                               compare_list=[less_than_cmp, "Male", "Female"],
                                               pvalue_filter=pvalue_filter)


def female_greater_than_male_generalized_occp_Work_Rate(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Generalized OCCP", grp_attr="SEX",
                                               cmp_attr="Work Rate",
                                               compare_list=[less_than_cmp, "Male", "Female"],
                                               pvalue_filter=pvalue_filter)


def female_greater_than_male_COW_Work_Rate(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="COW", grp_attr="SEX",
                                               cmp_attr="Work Rate",
                                               compare_list=[less_than_cmp, "Male", "Female"],
                                               pvalue_filter=pvalue_filter)


def full_cherrypicking_female_greater_than_male_salary(df,exclude_list, is_numeric,pvalue_filter=False):
    full_multichoice_attribute_cherrypicking(df=df, grp_attr="SEX", target_attr="PINCP",
                                             compare_list=[less_than_cmp, 1, 2],  #[less_than_cmp, "Male", "Female"],
                                             aggr_list=mean_list,
                                             exclude_list=exclude_list, is_numeric=is_numeric, pvalue_filter=pvalue_filter)


def full_cherrypicking_female_greater_than_male_salary_usingSql_filtering(df,exclude_list, is_numeric,pvalue_filter=False):
    full_multichoice_attribute_cherrypicking(df=df, grp_attr="SEX", target_attr="PINCP",
                                             compare_list=[less_than_cmp, 1, 2],
                                             aggr_list=mean_list,
                                             exclude_list=exclude_list, is_numeric=is_numeric, pvalue_filter=pvalue_filter, should_usesql=True)


def full_cherrypicking_black_greater_than_white_salary(df,exclude_list, is_numeric,pvalue_filter=False):
    full_multichoice_attribute_cherrypicking(df=df, grp_attr="RAC1P", target_attr="PINCP",
                                             compare_list=[less_than_cmp, "White alone", "Black or African American alone"],
                                             aggr_list=mean_list,
                                             exclude_list=exclude_list, is_numeric=is_numeric, pvalue_filter=pvalue_filter)


def full_cherrypicking_female_greater_than_male_salary_SQL(df,exclude_list,pvalue_filter=False):
    target_attr = "PINCP"
    attr_list = list(set(df.columns).difference(set(exclude_list + [target_attr])))
    MI_dict, Anova_dict = create_metrics_dictionary(df, attr_list, target_attr, is_numeric)
    #sorted_by_anova = sorted(Anova_dict.keys(), key=lambda col: Anova_dict[col][1])
    ordered=list(df.columns)
    random.shuffle(ordered)
    print(ordered)
    output_path = "data/Folkstable/ACS_mean_female_greater_than_male_baseline.csv"
    full_multichoice_attribute_cherrypicking_using_SQL(
        exclude_list=exclude_list, grp_attr="SEX", target_attr="PINCP", compare_list=[less_than_cmp, 1, 2],
        MI_dict=MI_dict, Anova_dict=Anova_dict, dataset_size=len(df), sorted_columns=ordered, agg_type='mean',
        pvalue_filter=pvalue_filter, output_path=output_path)
    #TODO: Might not use the output path twice as parameters, could be an issue for sampling
    #calculate_metrics_by_time_top_k(output_path, output_path, "Anova_Pvalue")


def full_cherrypicking_female_greater_than_male_median_salary_SQL(df, exclude_list, pvalue_filter=False):
    # This is the most updated wrapper function, take example from this.
    target_attr = "PINCP"
    attr_list = list(set(df.columns).difference(set(exclude_list + [target_attr])))
    MI_dict, Anova_dict = create_metrics_dictionary(df, attr_list, target_attr, is_numeric)
    sorted_by_anova = sorted(Anova_dict.keys(), key=lambda col: Anova_dict[col][1])
    output_path = "data/ACS_median_female_greater_than_male.csv"
    result_df = full_multichoice_attribute_cherrypicking_using_SQL(
        exclude_list=exclude_list, grp_attr="SEX", target_attr="PINCP", compare_list=[less_than_cmp, 1,2],
        MI_dict=MI_dict, Anova_dict=Anova_dict, dataset_size=len(df), sorted_columns=sorted_by_anova, agg_type='median',
        pvalue_filter=pvalue_filter, output_path=output_path)
    # TODO: Might not use the output path twice as parameters, could be an issue for sampling
    calculate_metrics_by_time_top_k_single_score(output_path, output_path, "Anova_Pvalue")


def full_cherrypicking_women_greater_than_men_median(df, exclude_list, is_numeric, pvalue_filter=False):
    full_multichoice_attribute_cherrypicking(df=df, grp_attr="SEX", target_attr="PINCP",
                                             compare_list=[less_than_cmp, 1,2],
                                             aggr_list=median_list,
                                             exclude_list=exclude_list, is_numeric=is_numeric,
                                             pvalue_filter=pvalue_filter)


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

    ############### RUN WITH WRAPPERS EXAMPLES ################
    # full_cherrypicking_female_greater_than_male_salary(df, exclude_list, is_numeric)
    # full_cherrypicking_women_greater_than_men_median(df, exclude_list, is_numeric)
    # full_cherrypicking_female_greater_than_male_median_salary_SQL(df,exclude_list)
    # full_cherrypicking_female_greater_than_male_salary_SQL(df, both_exclude_lists, is_numeric)
    # full_cherrypicking_female_greater_than_male_salary(df, both_exclude_lists, is_numeric)
    # full_cherrypicking_highschool_greater_than_bachelors_salary(df,exclude_list+["SCHL"],is_numeric)

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
        # if not USE_SQL or SORT_BY in ('REGRESSION', 'ALL_TOP_K_MERGED', 'ALL_TOP_K_SERIAL'):
        df = pd.read_csv(DATAFRAME_PATH, index_col=0)
        # else:
        #     df = pd.read_csv(DATAFRAME_PATH, index_col=0, nrows=1)
        trans_dict = make_translation_for_ACS(df.columns)
        allocation_flags = [col for col in df.columns if "allocation flag" in trans_dict[col]]
        exclude_list = both_exclude_lists + allocation_flags

        # shuffle_and_save(df, exclude_list, is_numeric, trans_dict, start_time, iters=3)

        ######## main quality experiment ########
        # result_df = run_cherrypicking_with_config(df, exclude_list, is_numeric, trans_dict)

        ################# pred level ##########################
        #pred_level_cherrypicking(df, exclude_list, is_numeric, trans_dict)

        ########### sample size experiment #####################
        #run_sample_guided_experiment([0.01], 1, df, exclude_list, is_numeric, trans_dict)

        ############## num tuples experiment #########################
        methods = ['original_order', 'random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'ALL_TOP_K_MERGED',
                   '0.01sample:1', '0.01sample:2', '0.01sample:3',
                   'ALL_TOP_K_SERIAL']
        # methods = ['0.01sample:2', '0.01sample:3']
        #sample_sizes = range(100000, 1000001, 100000)
        # sample_sizes = [100000, 200000, 300000, 400000, 500000]
        # num_tuples_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, trans_dict, methods,
        #                                 stop_at_recall=True)

        ############## num columns experiment #########################
        #col_sample_sizes = range(10, 101, 10)
        col_sample_sizes = [30,40,50,60,70,80,90,100]
        # # TODO repeat each sample size several times (to do an average later over the results)
        column_subsets = {
            10: ['POBP', 'SCIENGP', 'MLPH', 'HISP', 'SCIENGRLP', 'DDRS', 'HINS2', 'OCCP', 'INDP', 'MARHT'],
            20: ['SCHG', 'MARHD', 'POWPUMA', 'OC', 'ESR', 'WRK', 'LANP', 'NWAV', 'SOCP', 'DIVISION', 'ANC2P', 'SFN', 'HINS5', 'SCIENGRLP', 'FHINS5C', 'MIL', 'HINS3', 'DEYE', 'FER', 'NWAB'],
            30: ['DEAR', 'MSP', 'FHINS3C', 'MLPE', 'GCL', 'POWSP', 'POBP', 'GCR', 'RELP', 'FHINS4C', 'SCIENGP', 'MIGSP', 'NWLK', 'RAC3P', 'MLPA', 'ESP', 'OCCP_grouped', 'RAC1P', 'RACNUM', 'NATIVITY', 'JWMNP', 'ANC2P', 'RAC2P', 'NWAB', 'HINS2', 'SEX', 'AGEP', 'NAICSP', 'GCM', 'NWLA'],
            40: ['SEX', 'GCM', 'ANC', 'RAC1P', 'NATIVITY', 'MARHD', 'YOEP', 'PUBCOV', 'INDP', 'WAOB', 'HINS7', 'OCCP', 'AGEP', 'FHINS5C', 'ESP', 'LANP', 'MARHW', 'RAC3P', 'DPHY', 'DRAT', 'MIG', 'POWPUMA', 'ANC2P', 'DIVISION', 'MLPCD', 'PAOC', 'SCH', 'NAICSP_grouped', 'LANX', 'FOD1P', 'RACAIAN', 'DOUT', 'JWMNP', 'RACBLK', 'RACSOR', 'NWAV', 'GCL', 'MIGPUMA', 'SFN', 'FER'],
            50: ['INDP', 'JWTR', 'FHINS5C', 'RACPI', 'ESP', 'RACNUM', 'OCCP', 'FHINS4C', 'DOUT', 'DEAR', 'NWAB', 'DREM', 'SFN', 'MLPK', 'DPHY', 'RACBLK', 'RAC3P', 'HINS5', 'SFR', 'FOD1P', 'MLPE', 'RACASN', 'MIL', 'YOEP', 'MLPH', 'CIT', 'DECADE', 'PUMA', 'QTRBIR', 'MIG', 'RACSOR', 'RELP', 'MARHT', 'HINS7', 'MSP', 'LANP', 'RACWHT', 'NWAV', 'MLPA', 'SPORDER', 'MLPCD', 'MLPB', 'MIGPUMA', 'LANX', 'ANC1P', 'MARHD', 'NOP', 'NAICSP', 'HINS6', 'RACAIAN'],
            60: ['RAC3P', 'SCHG', 'RACASN', 'HINS6', 'MIGPUMA', 'MIG', 'MLPFG', 'NAICSP_grouped', 'LANP', 'AGEP', 'MLPCD', 'SCH', 'HINS4', 'FOD2P', 'WKL', 'RC', 'MARHM', 'DRIVESP', 'SOCP', 'HISP', 'DREM', 'CITWP', 'DOUT', 'RT', 'POWSP', 'GCL', 'OCCP', 'YOEP', 'GCR', 'INDP', 'RAC1P', 'RACSOR', 'DPHY', 'DECADE', 'JWRIP', 'SCIENGP', 'SFN', 'OC', 'WKW', 'JWDP', 'NWRE', 'RACNH', 'PRIVCOV', 'NWAB', 'NOP', 'FER', 'ESR', 'SPORDER', 'POWPUMA', 'VPS', 'MLPJ', 'RACWHT', 'MLPI', 'RACAIAN', 'MAR', 'MSP', 'MLPB', 'NWLK', 'ANC', 'DRATX'],
            70: ['MSP', 'GCM', 'RC', 'SEX', 'NATIVITY', 'RACSOR', 'RACASN', 'NWRE', 'NWAB', 'AGEP', 'ENG', 'NWLK', 'HICOV', 'ESP', 'MARHT', 'HINS1', 'GCL', 'MAR', 'MARHYP', 'SFN', 'WKL', 'LANP', 'WAOB', 'WRK', 'DECADE', 'RELP', 'PRIVCOV', 'ANC1P', 'RACBLK', 'OCCP', 'JWRIP', 'MLPI', 'HINS6', 'DDRS', 'DEYE', 'RT', 'ESR', 'MIGSP', 'OC', 'INDP', 'DRATX', 'RACWHT', 'DRAT', 'COW', 'RAC2P', 'MARHM', 'POWSP', 'HINS7', 'SCHG', 'YOEP', 'FHINS4C', 'PAOC', 'NAICSP_grouped', 'MLPA', 'HISP', 'MLPCD', 'MLPFG', 'ANC', 'SFR', 'GCR', 'DIVISION', 'SCIENGRLP', 'MIL', 'JWDP', 'LANX', 'NWLA', 'RACNUM', 'WKW', 'MLPJ', 'NOP'],
            80: ['MAR', 'MARHT', 'PUBCOV', 'JWMNP', 'VPS', 'RC', 'NWLA', 'ENG', 'DDRS', 'YOEP', 'SCH', 'GCL', 'NATIVITY', 'DIS', 'MIL', 'MIGSP', 'HINS1', 'DECADE', 'DEYE', 'NAICSP_grouped', 'AGEP', 'MLPFG', 'DRAT', 'MLPB', 'HINS5', 'WKHP', 'MLPE', 'ANC', 'POBP', 'OCCP_grouped', 'PAOC', 'PUMA', 'HINS2', 'NWRE', 'MSP', 'SCHL', 'OCCP', 'RAC3P', 'FHINS5C', 'COW', 'RELP', 'SFN', 'MARHD', 'FER', 'RACNH', 'FHINS3C', 'MLPA', 'SOCP', 'OC', 'NWAB', 'MARHM', 'HINS6', 'MARHYP', 'MLPCD', 'ANC2P', 'RACWHT', 'GCR', 'MIG', 'QTRBIR', 'NWLK', 'RACBLK', 'DREM', 'MARHW', 'SCHG', 'FHINS4C', 'HICOV', 'RACAIAN', 'LANP', 'WKW', 'SCIENGP', 'HINS3', 'SEX', 'RT', 'JWAP', 'GCM', 'JWRIP', 'NWAV', 'HINS7', 'MLPH', 'JWDP'],
            90: ['MIGSP', 'PUBCOV', 'OCCP', 'GCL', 'SOCP', 'SFR', 'MLPA', 'PUMA', 'DOUT', 'RACAIAN', 'JWDP', 'ST', 'MLPB', 'RACWHT', 'MLPH', 'WRK', 'SCHL', 'YOEP', 'MIL', 'SFN', 'HINS2', 'MAR', 'WAOB', 'REGION', 'MLPJ', 'GCR', 'CIT', 'NWLK', 'WKL', 'INDP', 'DPHY', 'HINS3', 'ENG', 'MIG', 'RACPI', 'SCHG', 'MLPK', 'OCCP_grouped', 'JWAP', 'SCH', 'RT', 'POBP', 'HINS6', 'COW', 'JWTR', 'DRIVESP', 'NWLA', 'DECADE', 'OC', 'JWMNP', 'DEAR', 'FOD2P', 'RACNH', 'MLPCD', 'DIVISION', 'RAC2P', 'POWPUMA', 'FHINS3C', 'MLPFG', 'MARHYP', 'MLPI', 'RACBLK', 'MARHD', 'SEX', 'WKHP', 'DRATX', 'NOP', 'RACSOR', 'RC', 'WKW', 'HISP', 'FHINS4C', 'RACNUM', 'MLPE', 'DREM', 'JWRIP', 'SCIENGRLP', 'FOD1P', 'PRIVCOV', 'HINS7', 'HINS4', 'MARHT', 'MARHW', 'ESP', 'ANC2P', 'RELP', 'NWRE', 'CITWP', 'AGEP', 'NAICSP'],
            100: ['HINS4', 'SOCP', 'GCL', 'FER', 'MLPB', 'FOD1P', 'MARHM', 'MARHYP', 'FHINS5C', 'HINS7', 'MLPFG', 'FHINS4C', 'RAC1P', 'YOEP', 'PAOC', 'PUMA', 'PRIVCOV', 'JWAP', 'FOD2P', 'HISP', 'HINS3', 'DREM', 'ANC1P', 'MLPE', 'RACNUM', 'WKL', 'MLPA', 'OCCP', 'DOUT', 'MARHW', 'SCH', 'SCHG', 'REGION', 'DIVISION', 'DRIVESP', 'SFN', 'RACAIAN', 'ANC2P', 'HINS2', 'ESR', 'MIG', 'MLPK', 'MARHD', 'RACSOR', 'RACASN', 'DRATX', 'HINS5', 'MIL', 'MLPCD', 'DDRS', 'SFR', 'MIGSP', 'POWSP', 'JWTR', 'MLPI', 'ST', 'DEYE', 'SCIENGRLP', 'NWAV', 'JWMNP', 'RACBLK', 'LANX', 'ENG', 'MIGPUMA', 'SCIENGP', 'SEX', 'QTRBIR', 'PUBCOV', 'CITWP', 'HINS6', 'POBP', 'NAICSP', 'RACWHT', 'NWLK', 'RAC2P', 'DEAR', 'NWAB', 'OCCP_grouped', 'DECADE', 'RELP', 'INDP', 'GCR', 'DRAT', 'RT', 'NWRE', 'RC', 'OC', 'AGEP', 'MLPH', 'HINS1', 'NOP', 'ESP', 'HICOV', 'CIT', 'JWDP', 'WKHP', 'MAR', 'DPHY', 'RAC3P', 'MARHT'],
        }
        num_columns_vs_time_for_full_run(col_sample_sizes, df, exclude_list, is_numeric, trans_dict, methods, column_subsets=None)

        ############# single metric experiment ######################
        # single_metric_exp(df, exclude_list, is_numeric, trans_dict)

        ################## sensitivity to k ########################
        # run the serial top k method for each k value.
        sensitivity_to_k(range(300, 1001, 100),  # K values
                         df, exclude_list, is_numeric, trans_dict,
                         reference_path="data/Folkstable/SevenStates/results/ACS7_numeric_mean_2atoms_F_gt_M_original_order_guided_reference.csv")

        ################## randomized queries #########################
        remaining = [c for c in df.columns if c not in exclude_list]
        target_attr = 'PINCP'
        #random_queries = randomize_queries(5, remaining, is_numeric, df, target_attr, agg_func='mean')
        # TODO check if the chosen values make sense.
        random_queries = [
            ('OCCP_grouped', [utils.less_than_cmp, 'ENG', 'BUS'], 'mean'),
            # ('DREM', [utils.less_than_cmp, 2.0, 1.0], 'mean'), # cognitive difficulty, 1-yes, 2-no
            # ('SCIENGP', [utils.less_than_cmp, 1.0, 2.0], 'mean'), # degree in science: 1- yes 2- no
            ('ESP', [utils.less_than_cmp, 6.0, 4.0], 'mean'),
            # 6 - living with non working father, 4- living with 2 non working parents.
            ('MAR', [utils.less_than_cmp, 1, 3], 'mean'),  # 1- married 3 - divorced
            ('SCHG', [utils.less_than_cmp, 10.0, 11.0], 'mean'),
            ('SCH', [utils.less_than_cmp, 1.0, 3.0], 'mean'),
            ('PRIVCOV', [utils.less_than_cmp, 1, 2], 'mean'),
            ('NWAB', [utils.less_than_cmp, 3.0, 1.0], 'mean'),
        ]


        # print(random_queries)
        # for grp_attr, compare_list, agg_type in random_queries:
        #     run_multiple_methods_for_query(target_attr, grp_attr, compare_list, agg_type,
        #                                    df, exclude_list, is_numeric, trans_dict, methods, stop_at_recall=True)

    end_time = datetime.datetime.now()

    print(f"total runtime= {(end_time-start_time).seconds}")


