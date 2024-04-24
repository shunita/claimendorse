import os.path
import re
import sys

from ClaimEndorseFunctions import *

job_list_orig = ["Academic researcher", "Cloud infrastructure engineer", "Blockchain", "Data or business analyst",
                 "Data scientist or machine learning specialist",
                 "Database administrator", "Designer", "Developer, back-end",
                 "Developer, desktop or enterprise applications", "Developer, embedded applications or devices",
                 "Developer, front-end", "Developer, full-stack", "Developer, game or graphics",
                 "Developer, mobile",
                 "Developer, QA or test", "DevOps specialist",
                 "Educator", "Engineer, data", "Engineer, site reliability", "Engineering manager",
                 "Marketing or sales professional", "Product manager", "Project manager",
                 "Scientist", "Senior Executive (C-Suite, VP, etc.)", "Student", "System administrator",
                 "Security professional"]


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


def woman_greater_than_man_devtype_split(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="DevType", grp_attr="Gender_simplified",
                                        cmp_attr="ConvertedCompYearly", compare_list=[less_than_cmp, "Man", "Woman"],pvalue_filter=pvalue_filter)


def woman_greater_than_man_country_split(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Country", grp_attr="Gender_simplified",
                                        cmp_attr="ConvertedCompYearly", compare_list=[less_than_cmp, "Man", "Woman"],pvalue_filter=pvalue_filter)


def woman_greater_than_man_ethnicity_split(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Ethnicity", grp_attr="Gender_simplified",
                                        cmp_attr="ConvertedCompYearly", compare_list=[less_than_cmp, "Man", "Woman"],pvalue_filter=pvalue_filter)


def ethnicity_greater_than_rest(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Ethnicity", grp_attr="", cmp_attr="ConvertedCompYearly",
                                        compare_list=[less_than_cmp, False, True],pvalue_filter=pvalue_filter)


def highschool_greater_than_bachelors_devtype_split(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="DevType", grp_attr="EdLevel",
                                        cmp_attr="ConvertedCompYearly",
                                        compare_list=[less_than_cmp,
                                                      "Bachelor’s degree",
                                                      "Secondary school",
                                                      ],pvalue_filter=pvalue_filter)


def highschool_greater_than_bachelors_country_split(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="Country", grp_attr="EdLevel",
                                        cmp_attr="ConvertedCompYearly",
                                        compare_list=[less_than_cmp,
                                                      "Bachelor’s degree",
                                                      "Secondary school",
                                                      ],pvalue_filter=pvalue_filter)


def bachelors_greater_than_masters_devtype_split(df,pvalue_filter=False):
    return multichoice_attribute_cherrypicking(df=df, split_attr="DevType", grp_attr="EdLevel",
                                        cmp_attr="ConvertedCompYearly",
                                        compare_list=[less_than_cmp,
                                                      "Master’s degree",
                                                      "Bachelor’s degree",
                                                      ],pvalue_filter=pvalue_filter)


def are_black_greater_than_white_salary(df,pvalue_filter=False):
    ret = multichoice_subgroup_comparison_cherrypicking(df=df, split_attr="Ethnicity",
                                                        cmp_attr="ConvertedCompYearly",
                                                        compare_list=[less_than_cmp, "White", "Black"],pvalue_filter=pvalue_filter)
    print(ret)
    return ret


def any_greater_than_whites_salary(df,pvalue_filter=False):
    ret=multichoice_all_comparison_cherrypicking(df=sep_df, split_attr="Ethnicity",cmp_attr="ConvertedCompYearly",compare_list=[less_than_cmp,"White"],pvalue_filter=pvalue_filter)
    print(ret)
    return ret


def any_greater_than_blacks_salary(df,pvalue_filter=False):
    ret = multichoice_all_comparison_cherrypicking(df=sep_df, split_attr="Ethnicity", cmp_attr="ConvertedCompYearly", compare_list=[less_than_cmp, "Black"],pvalue_filter=pvalue_filter)
    print(ret)
    return ret





def bachelors_greater_than_masters_good_pvalue(df,pvalue_filter=True):
    check_if_true_for_good_pvalue(df=df, split_attr="DevType", grp_attr="EdLevel", cmp_attr="ConvertedCompYearly",
                                  compare_list=[less_than_cmp,
                                                "Master’s degree",
                                                "Bachelor’s degree",
                                                ], pvalue_filter=pvalue_filter)


def masters_greater_than_bachelors_good_pvalue(df, pvalue_filter=True):
    check_if_true_for_good_pvalue(df=df, split_attr="DevType", grp_attr="EdLevel", cmp_attr="ConvertedCompYearly",
                                  compare_list=[less_than_cmp,
                                                "Bachelor’s degree",
                                                "Master’s degree",
                                                ], pvalue_filter=pvalue_filter)


def secondary_greater_than_bachelors_good_pvalue(df,pvalue_filter=True):
    check_if_true_for_good_pvalue(df=df, split_attr="DevType", grp_attr="EdLevel", cmp_attr="ConvertedCompYearly",
                                  compare_list=[less_than_cmp,
                                                "Bachelor’s degree",
                                                "Secondary school",
                                                ],pvalue_filter=pvalue_filter)


def woman_greater_than_man__salary_devtype_split_good_pvalue(df,pvalue_filter=True):
    check_if_true_for_good_pvalue(df=df, split_attr="DevType", grp_attr="Gender_simplified",
                                  cmp_attr="ConvertedCompYearly", compare_list=[less_than_cmp, "Man", "Woman"],pvalue_filter=pvalue_filter
                                  )


def full_cherrypicking_women_greater_than_men_salary(df,exclude_list, is_numeric,translation_dict,pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="Gender_simplified", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp, "Man", "Woman"], aggr_list=[np.mean,"mean",sp.stats.ttest_ind],
                                                    exclude_list=exclude_list, is_numeric=is_numeric, pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_women_greater_than_men_count(df,exclude_list, is_numeric,pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="Gender_simplified", target_attr="Age",
                                                    compare_list=[less_than_cmp, "Man", "Woman"],
                                                    aggr_list=[np.ma.count, "count", None],
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter)


def full_cherrypicking_women_greater_than_men_salary_median(df, exclude_list, is_numeric,translation_dict,   pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="Gender_simplified", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp, "Man", "Woman"],
                                                    aggr_list=median_list,
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_highschool_greater_than_bach_salary_median(df, exclude_list, is_numeric, pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp,
                                                "Bachelor’s degree",
                                                "Secondary school",
                                                ],
                                                    aggr_list=median_list,
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter)


def full_cherrypicking_highschool_greater_than_bach_salary_median(df, exclude_list, is_numeric, translation_dict,
                                                                      pvalue_filter=False):
        return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                        compare_list=[less_than_cmp,
                                                               "Bachelor’s degree",
                                                               "Secondary school",
                                                               ],
                                                        aggr_list=median_list,
                                                        exclude_list=exclude_list, is_numeric=is_numeric,
                                                        pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_highschool_greater_than_bach_salary_mean(df, exclude_list, is_numeric,translation_dict,
                                                                  pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp,
                                                           "Bachelor’s degree",
                                                           "Secondary school",
                                                           ],
                                                    aggr_list=mean_list,
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_masters_greater_than_bach_salary_median(df, exclude_list, is_numeric, translation_dict,pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp,
                                                "Bachelor’s degree",
                                                "Master’s degree",
                                                ],
                                                    aggr_list=median_list,
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_masters_greater_than_bach_salary_mean(df, exclude_list, is_numeric,translation_dict, pvalue_filter=False):
        return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                        compare_list=[less_than_cmp,
                                                               "Bachelor’s degree",
                                                               "Master’s degree",
                                                               ],
                                                        aggr_list=mean_list,
                                                        exclude_list=exclude_list, is_numeric=is_numeric,
                                                        pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_bach_greater_than_masters_salary_median(df, exclude_list, is_numeric,translation_dict, pvalue_filter=False):
    attr_list = list(set(df.columns).difference(set(exclude_list + ["ConvertedCompYearly"])))
    time_before=time.time()
    MI_dict, Anova_dict = create_metrics_dictionary(df, attr_list, "ConvertedCompYearly", is_numeric)
    time_after=time.time()
    print(f"time to calculate anova and mi: {time_after-time_before}")
    sorted_by_anova = sorted(Anova_dict.keys(), key=lambda x: Anova_dict[x][0],reverse=True)
    output_path=os.path.join(DATA_PATH, "SO_bach_greater_master_salary_median_sorted_by_anova_pvalue.csv")
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp,
                                              "Master’s degree",
                                              "Bachelor’s degree"],
                                                    aggr_list=median_list,
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter, translation_dict=translation_dict, sorted_columns=sorted_by_anova,
                                                    MI_dict=MI_dict, Anova_dict=Anova_dict, output_path=output_path)


def full_cherrypicking_bach_greater_than_masters_salary_mean(df, exclude_list, is_numeric, translation_dict,
                                                             pvalue_filter=False):
    return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                    compare_list=[less_than_cmp,
                                                                  "Master’s degree",
                                                                  "Bachelor’s degree"
                                                                  ],
                                                    aggr_list=mean_list,
                                                    exclude_list=exclude_list, is_numeric=is_numeric,
                                                    pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_masters_greater_than_bach_salary_mean(df, exclude_list, is_numeric, translation_dict,
                                                                 pvalue_filter=False):
        return full_multichoice_attribute_cherrypicking(df=df, grp_attr="EdLevel", target_attr="ConvertedCompYearly",
                                                        compare_list=[less_than_cmp,
                                                               "Bachelor’s degree",
                                                               "Master’s degree"
                                                               ],
                                                        aggr_list=mean_list,
                                                        exclude_list=exclude_list, is_numeric=is_numeric,
                                                        pvalue_filter=pvalue_filter, translation_dict=translation_dict)


def full_cherrypicking_women_greater_men_mean_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_women_greater_than_men_salary, df, exclude_list, is_numeric,
                           translation_dict, create_sex_string)


def full_cherrypicking_highschool_greater_bach_mean_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_highschool_greater_than_bach_salary_mean, df, exclude_list, is_numeric,
                           translation_dict, create_education_string)


def full_cherrypicking_bach_greater_masters_mean_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_bach_greater_than_masters_salary_mean, df, exclude_list, is_numeric,
                           translation_dict, create_education_string)


def full_cherrypicking_masters_greater_bach_mean_salary_chatgpt(df, exclude_list, is_numeric, translation_dict):
        fullsplit_with_chatgpt(full_cherrypicking_masters_greater_than_bach_salary_mean, df, exclude_list, is_numeric,
                               translation_dict, create_education_string)


def full_cherrypicking_women_greater_men_median_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_women_greater_than_men_salary_median, df, exclude_list, is_numeric,
                           translation_dict, create_sex_string)


def full_cherrypicking_highschool_greater_bach_median_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_highschool_greater_than_bach_salary_median, df, exclude_list, is_numeric,
                           translation_dict, create_education_string)


def full_cherrypicking_bach_greater_masters_median_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_bach_greater_than_masters_salary_median, df, exclude_list, is_numeric,
                           translation_dict, create_education_string)


def full_cherrypicking_masters_greater_bach_median_salary_chatgpt(df,exclude_list,is_numeric,translation_dict):
    fullsplit_with_chatgpt(full_cherrypicking_masters_greater_than_bach_salary_median, df, exclude_list, is_numeric,
                           translation_dict, create_education_string)


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
    #df = remove_outliers(df, "ConvertedCompYearly")

    df = read_dataset()

    # This should only be done once!
    # print("Uploading to DB")
    # df_to_sql_DB(df)
    # sys.exit()

    remaining = [c for c in df.columns if c not in exclude_list and c not in is_numeric]
    print(f"{len(remaining)} remaining (string) cols after exclusion: {remaining}")
    translation_dict = create_column_dictionary(df)

    #df = remove_outliers(df, "ConvertedCompYearly")
    #df=df[["Age","ConvertedCompYearly","EdLevel"]]

    # save three random orders
    #shuffle_and_save(df, exclude_list, is_numeric, translation_dict, start_time, iters=3)
    #run_random_shuffle_over_full_table(df, exclude_list, is_numeric, translation_dict, 3, "Bsc_gt_Msc")

    ####################### main quality / best sample size experiment #################################
    result_df = run_cherrypicking_with_config(df, exclude_list, is_numeric, translation_dict)

    ############ sample size experiment ##################
    #run_sample_guided_experiment([0.05, 0.1], 3, df, exclude_list, is_numeric, translation_dict)

    ############## num tuples experiment #########################
    methods = ['original_order', 'random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'ALL_TOP_K_MERGED',
               '0.01sample:1', '0.01sample:2', '0.01sample:3', 'ALL_TOP_K_SERIAL']
    sample_sizes = range(10000, 50001, 5000)
    #num_tuples_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, translation_dict, methods)

    ############## num columns experiment #########################
    col_sample_sizes = range(5, 50, 5)
    # TODO repeat each sample size several times (to do an average later over the results)
    #num_columns_vs_time_for_full_run(col_sample_sizes, df, exclude_list, is_numeric, translation_dict, methods)

    ############# single metric experiment ######################
    # single_metric_exp(df, exclude_list, is_numeric, translation_dict)

    ################## Predicate Level ##########################
    #pred_level_cherrypicking(df, exclude_list, is_numeric, translation_dict)

    ################## Random Queries ###########################
    remaining = [c for c in df.columns if c not in exclude_list]
    # random_queries = randomize_queries(6, remaining, is_numeric, df, 'ConvertedCompYearly')
    random_queries = [('Ethnicity', [utils.less_than_cmp, "I don't know", 'Indian'], 'median'),
                      #('LearnCodeCoursesCert', [utils.less_than_cmp, 'Pluralsight', 'Codecademy'], 'mean'),
                      #('NEWCollabToolsHaveWorkedWith', [utils.less_than_cmp, 'Visual Studio', 'Nano'], 'median')
                      # ('MentalHealth', [utils.less_than_cmp, 'I have a concentration and/or memory disorder (e.g., ADHD, etc.)', 'I have an anxiety disorder'], 'mean'),
                      #  ('OfficeStackSyncHaveWorkedWith', [utils.less_than_cmp, 'Mattermost', 'Zoom'], 'mean'),
                      # ('Trans', [utils.less_than_cmp, 'Yes', 'Or, in your own words:'], 'mean'),
                      ]
    target_attr = "ConvertedCompYearly"
    for grp_attr, compare_list, agg_type in random_queries:
        run_multiple_methods_for_query(target_attr, grp_attr, compare_list, agg_type,
                                       df, exclude_list, is_numeric, translation_dict, methods, stop_at_recall=True)





