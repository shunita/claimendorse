import os.path
import re
import sys

from ClaimEndorseFunctions import *


def value_cleaning_helper(x, old_vals_to_new_vals):
    if x in old_vals_to_new_vals:
        return old_vals_to_new_vals[x]
    return x


def value_cleaning_SO(df):
    """
    :param df: the dataframe
    :return: returns the dataframe but changes certain values to be more readable + remove nan columns
    """
    secondary_ed_org = "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)"
    secondary_ed_new = "Secondary school"
    bachelors_org = "Bachelor’s degree (B.A., B.S., B.Eng., etc.)"
    bachelors_new = "Bachelor’s degree"
    masters_org = "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)"
    masters_new = "Master’s degree"
    ed_level_d = {secondary_ed_org: secondary_ed_new, bachelors_org: bachelors_new, masters_org: masters_new}
    df["EdLevel"] = df["EdLevel"].apply(lambda v: value_cleaning_helper(v, ed_level_d))
    yc_d = {'More than 50 years': '50', 'Less than 1 year': '0'}
    df['YearsCode'] = df['YearsCode'].apply(lambda v: value_cleaning_helper(v, yc_d))
    df['YearsCodePro'] = df['YearsCodePro'].apply(lambda v: value_cleaning_helper(v, yc_d))

    attrs_before = set(df.columns)
    df = df.dropna(axis=1, how='all')
    attrs_after_drop = set(df.columns)
    print(f"Empty columns that were removed:{attrs_before.difference(attrs_after_drop)}")
    return df


def safe_split(s):
    if type(s) != str:
        return s
    return s.split(';')[0]


def read_dataset():
    processed_path = os.path.join(DATA_PATH, "temp_df_for_sql.csv")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, index_col=0)
        return df
    df = pd.read_csv(DATAFRAME_PATH, index_col=0)
    df = value_cleaning_SO(df)
    for col in df.columns:
        if is_multivalue_attr(df, col):
            print(f"Taking first value from multivalue attr: {col}")
            df[col] = df[col].apply(safe_split)
    df.to_csv(processed_path)
    df = pd.read_csv(processed_path, index_col=0)
    return df


def seperate_genders(df):
    # Makes any gender that is not "Man" or "Woman" into Other
    gender_column = df['Gender']
    gender_selection_list = ["Man", "Woman"]

    def simplify_gender(x):
        if x in gender_selection_list:
            return x
        return "Other"

    df['Gender_simplified'] = df['Gender'].apply(simplify_gender)
    # for i in range(gender_column.size):
    #     row = gender_column[i]
    #     if row not in gender_selection_list:
    #         df.at[i, 'Gender'] = "Other"
    # # print(df)
    return df



def create_manual_translation():
    translation_dict = {
        "EdLevel": "Education Level",
        "LearnCodeCoursesCert": "Learn Code Courses Certification",
        "YearsCodePro": "Years Code Professionally",
        "DevType": "Developer Type",
        "OrgSize": "Organization Size",
        "CompTotal": "Compensation Total",
        "CompFreq": "Compensation Frequency",
        "MiscTechHaveWorkedWith": "Miscellaneous Technology Have Worked With",
        "MiscTechWantToWorkWith": "Miscellaneous Technology Want To Work With",
        "OpSysProfessional use": "Operating System Professional use",
        "OpSysPersonal use": "Operating System Personal use",
        "VCInteraction": "Version Control Interaction",
        "VCHostingPersonal use": "Version Control Hosting Personal use",
        "VCHostingProfessional use": "Version Control Hosting Professional use",
        "SOVisitFreq": "Stack Overflow Visit Frequency",
        "WorkExp": "Work Experience",
        "ConvertedCompYearly": "Salary",
    }
    return translation_dict


def create_column_dictionary(df):
    updated_dict = create_manual_translation()
    for column in df.columns:
        if column not in updated_dict.keys():
            updated_dict[column] = ' '.join(re.findall('[A-Z][^A-Z]*', column))
    return updated_dict



if __name__ == '__main__':
    start_time = datetime.datetime.now()
    exclude_list = ["ResponseId", "CompTotal", "CompFreq",
                    "Currency", "SOAccount", "NEWSOSites", "SOVisitFreq", "SOPartFreq", "SOComm", "TBranch",
                    "TimeAnswering", "Onboarding", "ProfessionalTech", "SurveyLength", "SurveyEase",
                    "ConvertedCompYearly"]
    exclude_list += ["Knowledge_" + str(i) for i in range(1, 8)]
    exclude_list += ["Frequency_" + str(i) for i in range(1, 4)]
    exclude_list += ["TrueFalse_" + str(i) for i in range(1, 4)]

    is_numeric = ["YearsCode", "YearsCodePro", "ConvertedCompYearly", "WorkExp"]

    df = read_dataset()

    # This should only be done once!
    # print("Uploading to DB")
    # df_to_sql_DB(df)
    # sys.exit()

    remaining = [c for c in df.columns if c not in exclude_list and c not in is_numeric]
    print(f"{len(remaining)} remaining (string) cols after exclusion: {remaining}")
    translation_dict = create_column_dictionary(df)

    # save three random orders
    # shuffle_and_save(df, exclude_list, is_numeric, translation_dict, start_time, iters=3)
    # run_random_shuffle_over_full_table(df, exclude_list, is_numeric, translation_dict, 3, "Bsc_gt_Msc")

    ####################### main quality experiment #################################
    methods = ['random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'original_order', 
                'ALL_TOP_K_MERGED', 'ALL_TOP_K_SERIAL',
                '0.01sample:1', '0.01sample:2', '0.01sample:3',
                '0.05sample:1', '0.05sample:2', '0.05sample:3',
                '0.10sample:1', '0.10sample:2', '0.10sample:3']
    run_multiple_methods_for_query(TARGET_ATTR, GRP_ATTR, COMPARE_LIST, AGG_TYPE,
                                   df, exclude_list, is_numeric, translation_dict, methods, stop_at_recall=False)

    ############ sample size experiment ##################
    #run_sample_guided_experiment([0.05, 0.1], 3, df, exclude_list, is_numeric, translation_dict)

    ############## num tuples experiment #########################
    methods = ['original_order', 'random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'ALL_TOP_K_MERGED',
               '0.01sample:1', '0.01sample:2', '0.01sample:3', 'ALL_TOP_K_SERIAL']
    sample_sizes = range(10000, 50001, 5000)
    #num_tuples_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, translation_dict, methods)

    ############## num columns experiment #########################
    col_sample_sizes = range(5, 50, 5)
    #num_columns_vs_time_for_full_run(col_sample_sizes, df, exclude_list, is_numeric, translation_dict, methods)

    ################## Predicate Level ##########################
    #pred_level_cherrypicking(df, exclude_list, is_numeric, translation_dict)





