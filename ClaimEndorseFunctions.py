import itertools
import multiprocessing
import sys
import random
import pickle
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from statsmodels.stats.proportion import proportions_ztest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from feature_engine.encoding import OrdinalEncoder
from sentence_transformers import SentenceTransformer, util

from QueryRunner import *
from SemanticSim import SemanticSim
from utils import *
from config import *
from constants import *
from analyze_output import analyze_regression

PVALUE_CONST = 0.05
MIN_SAMPLE_SIZE_FOR_TEST = 4
MIN_SUBSET_SIZE = 8
COUNT_FILTERED_OUT_BY_SIZE = {"x": 0}
TIME_ANOVA_CALCULATION = {"x": datetime.timedelta(seconds=0)}

CHATGPT_MODEL_STRING = "gpt-3.5-turbo"


def isNaN(value):
    return value != value


def value_range_to_interval_size(value_range):
    if value_range > 50000:
        return 10 ** 4
    if value_range > 10000:
        return 1000
    if value_range > 1000:
        return 100
    return 10


def is_multivalue_attr(df, attr, split_str=";"):
    """
    :param df: the dataframe
    :param attr: the attribute we are checking
    :param split_str: the string we are delineating our values with
    :return: True if the attribute is a multivalue field, because it has values with the split_str between them
    """
    values = df[attr].unique()
    for v in values:
        if split_str in str(v):
            return True
    return False


def flatten_list_of_lists(list_of_lists):
    """
    :param list_of_lists: a list, where all the elements are lists(depth of one). Eg. [[1,2],[3,4,5]].
    However [[[3,4,6]]] is not valid.
    :return: returns all the elements of all the list that are in list_of_lists, in one list
    """
    res = []
    for l in list_of_lists:
        res += l
    return res


def get_unique_values(df, field, separator=';'):
    raw_values = df[~df[field].isna()][field].values

    values = set(flatten_list_of_lists([x.split(separator) for x in raw_values]))
    return list(values)


def get_dummies_multi_hot(df, field, separator=';'):
    """
    :param df: the dataframe we are using
    :param field: the field in which we are trying to separate. the elements of the field need to be of the form
    elm1-"Devops;Data Scientist"
    elm2- "Data Scientist; IT"
    Here each entry in the list can be part of many groups, and we want to split the participation in such a group into
    a binary field. The ';' is our separate
    :param separator: How we separate the values
    :return: returns a new dataframe where we added the participation in any of these groups as binary attributes.
    we also return a list of all the new attributes we added
    """
    values = get_unique_values(df, field, separator)

    def value_in_field(x, v):
        if pd.isna(x):
            return False
        return v in x.split(separator)

    new_columns = []
    for value in values:
        new_attr_name = field + ";" + value
        new_columns.append(new_attr_name)
        df[new_attr_name] = df[field].apply(lambda x: value_in_field(x, value))
    return df, new_columns


def df_to_sql_DB(df):
    # This should only be run once. (Per DB)
    engine = connect_sql_db()
    df.to_sql("my_table", engine)


def run_filtering_query(column, engine, grp_attr, target_attr, compare_list, aggr_list):
    _, agg_func_desc, _ = aggr_list
    # TODO: support MEDIAN
    func_desc_to_sql = {"mean": "AVG", "count": "COUNT"}
    sql_agg = func_desc_to_sql[agg_func_desc]
    cmpf, val1, val2 = compare_list
    table_name = "my_table"
    if SAMPLE_SIZE is not None:
        table_name = "sample_view"
    query = sqlalchemy.text(f"""SELECT "{column}"
                            FROM {table_name} GROUP BY "{column}"
                            HAVING {sql_agg}(CASE WHEN "{grp_attr}"='{val1}' THEN "{target_attr}" ELSE NULL END)<
                            {sql_agg}(CASE WHEN "{grp_attr}"='{val2}' THEN "{target_attr}" ELSE NULL END);""")
    query_result = engine.execute(query)
    return query_result


def run_counting_query(column, engine):
    table_name = "my_table"
    if SAMPLE_SIZE is not None:
        table_name = "sample_view"
    query = sqlalchemy.text(f"""SELECT "{column}", count("{column}")
                                FROM {table_name}
                                GROUP BY "{column}";""")
    query_result = engine.execute(query)
    return query_result


class Bucket(object):
    def __init__(self, low, high, count):
        self.low = low
        self.high = high + 1
        self.count = count

    def bucket_to_range_string(self, bucket_id):
        if bucket_id is None or math.isnan(bucket_id):
            return None
        bucket_width = (self.high - self.low) / self.count
        lower = self.low + int((bucket_id - 1) * bucket_width)
        upper = self.low + int(bucket_id * bucket_width)
        return f"{lower}-{upper}"

    def value_to_bucket_id(self, value):
        if value is None or math.isnan(value):
            return value
        bucket_width = (self.high - self.low) / self.count
        return math.ceil(value / bucket_width)

    def get_all_buckets(self):
        return range(1, self.count + 1)

    def __str__(self):
        s = ""
        for bucket_id in range(1, self.count + 1):
            s += f"{self.bucket_to_range_string(bucket_id)}\n"
        return s

    @classmethod
    def from_attr_name(cls, attr_name):
        engine = SQLEngineSingleton()
        query = sqlalchemy.text(f"""SELECT MIN("{attr_name}"), MAX("{attr_name}") 
                                    FROM my_table;""")
        min_val, max_val = list(engine.execute(query))[0]
        interval_size = value_range_to_interval_size(max_val - min_val)
        # round max (up) and min (down) to the closest <interval_size>
        rounded_max = int(math.ceil(max_val / interval_size) * interval_size)
        rounded_min = int((min_val // interval_size) * interval_size)
        count = int((rounded_max - rounded_min) // interval_size)
        # count = 10
        return cls(rounded_min, rounded_max, count)


def drop_sample_view(view_name='sample_view'):
    engine = SQLEngineSingleton()
    for vname in ['sample_view', view_name]:
        query = sqlalchemy.text(f"""DROP MATERIALIZED VIEW IF EXISTS {vname} ;""")
        query_result = engine.execute(query)


def make_sample_view(sample_size, from_table="my_table", to_table="sample_view", drop_if_existing=True):
    engine = SQLEngineSingleton()
    if drop_if_existing:
        drop_sample_view(to_table)
    query = sqlalchemy.text(f"""CREATE MATERIALIZED VIEW {to_table} AS
                                SELECT *
                                FROM {from_table}
                                WHERE "{TARGET_ATTR}" IS NOT NULL
                                ORDER BY RANDOM() LIMIT {sample_size};""")
    print(f"running query: {query}")
    try:
        query_result = engine.execute(query)
    except Exception as e:
        print(e)


def get_DB_size(main_table):
    engine = SQLEngineSingleton()
    query = sqlalchemy.text(f"""SELECT COUNT("{TARGET_ATTR}") FROM {main_table};""")
    return list(engine.execute(query))[0][0]


def single_attribute_cherrypicking(df, orig_df_size, grp_attr, cmp_attr, column_name, compare_list, aggr_list,
                                   pvalue_filter,
                                   sqlEngine, query_result_dict, count_query_result_dictionary,
                                   compare_value_numeric=False, complement_single_cherrypicking=False):
    """
    :param df: dataframe we will be cherrypicking
    :param orig_df_size: the number of rows in the original df, before removing any NaNs
    :param grp_attr: we are grouping by this attribute(E.g. Gender)
    :param cmp_attr: the multichoice attribute we want to compare (E.g. Yearly Salary)
    :param column_name: The attribute we are looking at. (E.g. The job title)
    :param compare_list: a list of the following form- [cmpf, value1,value2] where cmpf is a comparison function and
       value1, value2 are the values in field we will compare for cherrypicking
    :param aggr_list: a list of the form [aggr_function,aggr_name,pvalue_function]

    Example: [mean,"mean",sp.stats.ttest_ind]
    :param pvalue_filter: true if we only want to see groups with pvalue <= PVALUE_CONST=0.05
    :return: returns all the subgroups where when we group by grp_attr, and look at the field of
    cmp_attr we see that the condition specified by the compare list is satisfied.
    E.g.- If we are looking for job titles in which women are paid more than men on average,
    we will choose
    grp_attr="Gender"
    cmp_attr="YearlySalary"
    column_val="Devops"
    compare_list=[less_than_cmp,"Man","Woman"], where less_than_cmp(val1,val2) returns if val1<val2
    """

    if sqlEngine is None:
        values = df[column_name].unique()
    else:
        values = query_result_dict[column_name].keys()  # Transforming the dictionary into an iterable

    # print(values)
    successful = []
    cmpf, val1, val2 = compare_list
    aggr_function, aggr_name, pvalue_function = aggr_list
    extra_information_string = ": False" if complement_single_cherrypicking else ": True"
    for v in values:
        if count_query_result_dictionary is not None and count_query_result_dictionary[column_name][v][
            0] <= MIN_SUBSET_SIZE:
            COUNT_FILTERED_OUT_BY_SIZE["x"] += 1
            continue
        # print(f"{column_val} type = {df.dtypes[column_val]}")
        # if sqlEngine is not None and df.dtypes[column_val] is not str and not is_column_string:
        #     if v != '':
        #         v = float(v)
        df_cur_value = df[df[column_name] != v] if complement_single_cherrypicking else df[df[column_name] == v]
        a_aggr, b_aggr, n_a, n_b, pvalue = validate_claim_on_subset(df_cur_value, grp_attr, cmp_attr, compare_list,
                                                                    aggr_list, compare_value_numeric)

        if pvalue_filter and pvalue > PVALUE_CONST:
            continue

        if cmpf(a_aggr, b_aggr):
            subset_size = len(df_cur_value)
            percent = ((subset_size / orig_df_size) * 100)
            successful.append((column_name, v, extra_information_string, subset_size, round(percent, 2), val1, n_a,
                               a_aggr, val2, n_b, b_aggr, round(pvalue, 3), time.time()))
            print_significance_data(value=v, col_val=column_name, subset_size=subset_size, percent=percent, val1=val1,
                                    len_a=n_a, aggr_a=a_aggr, val2=val2, len_b=n_b, aggr_b=b_aggr,
                                    p_value=pvalue, aggr_func=aggr_name, additional_string=extra_information_string)
    return successful


def validate_claim_on_subset(df, attr, cmp_attr, compare_list, aggr_list, compare_value_numeric):
    """
       :param df: dataframe we will be cherrypicking
       :param grp_attr: we are grouping by this attribute(E.g. Gender)
       :param cmp_attr: the multichoice attribute we want to compare (E.g. Yearly Salary)
       :param column_val: The attribute we are looking at. (E.g. The job title)
       :param compare_list: a list of the following form- [cmpf, value1,value2] where cmpf is a comparison function and
           value1, value2 are the values in field we will compare for cherrypicking
       :param pvalue_filter: true if we only want to see groups with pvalue <= PVALUE_CONST=0.05
       :return: returns all the subgroups where when we group by grp_attr, and look at the field of
       cmp_attr we see that the condition specified by the compare list is satisfied.
       E.g.- If we are looking for job titles in which women are payed more than men on average,
       we will choose
       grp_attr="Gender"
       cmp_attr="YearlySalary"
       column_val="Devops"
       compare_list=[less_than_cmp,"Man","Woman"], where less_than_cmp(val1,val2) returns if val1<val2
       """
    df = df.dropna(subset=[attr, cmp_attr]).copy()
    cmpf, val1, val2 = compare_list
    a = df[df[attr] == val1][cmp_attr].values
    b = df[df[attr] == val2][cmp_attr].values
    if compare_value_numeric:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
    aggr_function, aggr_name, pvalue_function = aggr_list

    ttest_result = 0
    if len(a) < MIN_SAMPLE_SIZE_FOR_TEST or len(b) < MIN_SAMPLE_SIZE_FOR_TEST:
        return 0, 0, len(a), len(b), 1
    if pvalue_function is not None:
        ttest_result = pvalue_function(a, b)
        pvalue = ttest_result.pvalue
    else:
        pvalue = -1
    return aggr_function(a), aggr_function(b), len(a), len(b), pvalue


def print_significance_data(value, col_val, subset_size, percent, val1, len_a, aggr_a, val2, len_b, aggr_b, aggr_func,
                            p_value, additional_string="", ):
    percent_str = "%.1f" % percent
    p_value_str = "%.3f" % p_value
    print(
        f"\n{col_val}={value}{additional_string},N={subset_size} ({percent_str}%), {val1}: N1={len_a}, {aggr_func}1={aggr_a}, {val2}: N2={len_b}, {aggr_func}2={aggr_b}, p-value:{p_value_str}")


def multichoice_attribute_cherrypicking(df, split_attr, grp_attr, cmp_attr, compare_list, aggr_list, sqlEngine,
                                        query_result_dict, count_query_result_dictionary, pvalue_filter=False,
                                        compare_value_numeric=False):
    """
    :param df: dataframe we will be cherrypicking
    :param field: field will be the field we will cherrypick
    :param attr: the multichoice attribute we will split according to
    :param compare_list: a list of the following form- [cmpf, value1,value2] where cmpf is a comparison function and
        value1, value2 are the values in field we will compare for cherrypicking
    :return:
    """
    return_list = []
    orig_df_size = len(df)
    df1 = df.dropna(subset=[split_attr, grp_attr, cmp_attr]).copy()
    print(f"Cherypicking Split-Attr={split_attr}, Attribute={grp_attr} Comparison={cmp_attr}")
    if not is_multivalue_attr(df, split_attr):
        result = single_attribute_cherrypicking(df=df1, orig_df_size=orig_df_size, grp_attr=grp_attr, cmp_attr=cmp_attr,
                                                column_name=split_attr,
                                                compare_list=compare_list, aggr_list=aggr_list,
                                                pvalue_filter=pvalue_filter,
                                                compare_value_numeric=compare_value_numeric,
                                                complement_single_cherrypicking=False, sqlEngine=sqlEngine,
                                                query_result_dict=query_result_dict,
                                                count_query_result_dictionary=count_query_result_dictionary)
        if len(result) > 0:
            return_list.append(result)

        # To Enable predicates of the form attr != value, uncomment the following lines:
        # result = single_attribute_cherrypicking(df=df1,orig_df_size=orig_df_size, grp_attr=grp_attr, cmp_attr=cmp_attr, column_name=split_attr,
        #                                         compare_list=compare_list, aggr_list=aggr_list,
        #                                         pvalue_filter=pvalue_filter,
        #                                         compare_value_numeric=compare_value_numeric,
        #                                         complement_single_cherrypicking=True, sqlEngine=sqlEngine, query_result_dict=query_result_dict,
        #                                         count_query_result_dictionary=count_query_result_dictionary)
        # if len(result) > 0:
        #     return_list.append(result)

        return df, return_list

    # Here is_multivalue_attr is True(as in the current attribute is a multivalued one)
    new_df, new_columns = get_dummies_multi_hot(df1, split_attr)

    for c in new_columns:
        result = single_attribute_cherrypicking(df=new_df, orig_df_size=orig_df_size, grp_attr=grp_attr,
                                                cmp_attr=cmp_attr, column_name=c,
                                                compare_list=compare_list, aggr_list=aggr_list,
                                                pvalue_filter=pvalue_filter,
                                                compare_value_numeric=compare_value_numeric, sqlEngine=sqlEngine,
                                                query_result_dict=query_result_dict,
                                                count_query_result_dictionary=count_query_result_dictionary)
        if len(result) > 0:
            return_list.append(result)
    return new_df, return_list


def multichoice_subgroup_comparison_cherrypicking(df, split_attr, cmp_attr, compare_list):
    new_df, new_columns = get_dummies_multi_hot(df, split_attr)
    columns_dict = {}
    cmpf, val1, val2 = compare_list
    for c in new_columns:
        columns_dict[c] = df.groupby([c])[
            cmp_attr].mean()
    selected_val1 = columns_dict[val1][True]
    selected_val2 = columns_dict[val2][True]
    return cmpf(selected_val1, selected_val2)


def multichoice_all_comparison_cherrypicking(df, split_attr, cmp_attr, compare_list):
    """
    :param df:
    :param split_attr:
    :param cmp_attr:
    :param compare_list:
    :return:
    """
    df1 = df.dropna(subset=[split_attr, cmp_attr]).copy()
    new_df, new_columns = get_dummies_multi_hot(df1, split_attr)
    columns_dict = {}
    df = new_df
    cmpf, val1 = compare_list
    for c in new_columns:
        columns_dict[c] = df.groupby([c])[
            cmp_attr].mean()

    result_list = []
    selected_val1 = columns_dict[val1][True]
    for c in new_columns:
        selected_val2 = columns_dict[c][True]
        if cmpf(selected_val1, selected_val2):
            result_list.append(c)
            subset_size = len(df[df[c] == True])
            percent = ((subset_size / len(df)) * 100)
            a = df[(df[val1] == True)][cmp_attr].values
            b = df[(df[c] == True)][cmp_attr].values
            # print(a)
            # print(b)
            ttest_result = sp.stats.ttest_ind(a, b)
            # print(ttest_result)
            p_value = ttest_result.pvalue
            print_significance_data(value=True, col_val=c, subset_size=subset_size, percent=percent, val1=val1,
                                    len_a=len(a),
                                    mean_a=columns_dict[val1][True], val2=c, len_b=len(b), mean_b=columns_dict[c][True],
                                    p_value=p_value)
    print("-----------------------------")
    return result_list


def find_group_percentage(df, split_attr, subgroup_val, groupby_attr):
    new_df, new_columns = get_dummies_multi_hot(df, split_attr)
    groupby = new_df[new_df[subgroup_val] == True].groupby([groupby_attr])[subgroup_val].count()
    # print(gb)
    values = groupby.index.values.tolist()
    sum = 0
    size_dict = {}
    for v in values:
        # print(f"{v}={gb[v]}")
        sum = sum + groupby[v]
        size_dict[v] = groupby[v]
    subgroups_str = f"Group={subgroup_val} Split={groupby_attr}, N={sum}  "
    size_dict["SUM"] = sum
    i = 1
    for v in values:
        percent = (groupby[v] / sum) * 100
        percent_str = "%.1f" % percent
        subgroups_str = subgroups_str + f"{v}: N{i}={groupby[v]}, {percent_str}%  "
        i = i + 1
    print(subgroups_str)
    return size_dict


def print_proportions_significance_data(value, subgrp_attr, grp_attr, cmpr_attr, val1, len_a, percent1, val2, len_b,
                                        percent2,
                                        p_value):
    percent1_str = "%.1f" % percent1
    percent2_str = "%.1f" % percent2
    p_value_str = "%.3f" % p_value
    print(
        f"\nFor value={value}:\nCherrypicking subgroups of-{subgrp_attr}, split by-{grp_attr}, looking at percentages of-{cmpr_attr}:\n\
        {val1}: N1={len_a}, percent1={percent1_str}% {val2}: N2={len_b}, percent2={percent2_str}%, p-value:{p_value_str}")
    # print("----------------------------")


def cherrypick_proportions(df, subgroup_attr, subgroup_vals, groupby_attr, cmpr_attr, cmpr_list, pvalue_filter=False):
    """
   :param df: dataframe we will be cherrypicking
   :param subgroup_attr: we are comparing two subgroups from this attribute(E.g. Sex)
   :param subgroup_vals: a list of the form [val1,val2] where we will be looking and comparing tuples from
                        these two groups only. E.g ["Male","Female"]
   :param groupby_attr: the value we are grouping by (E.g COW)
   :param cmpr_attr: the attribute we are going to compare between the two groups(E.g. Income>50k)
   :param cmpr_list: a list of the following form- [cmpf, value] where cmpf is a comparison function and
       value is the values in field we will compare for cherrypicking within our two subgroups
   :param pvalue_filter: true if we only want to see groups with pvalue <= PVALUE_CONST=0.05
   :return: returns all the subgroups where when we group by group_attr, and look at the field of
   cmpr_attr we see that the percentages of the two groups satisfy some condition created by the compare list.
   E.g.- If we are looking for job types in which women have a higher proportion that earn more than 50k a year
     than men on average,
   we will choose
   subgroup_attr="SEX"
   subgroup_vals=["Male","Female"]
   groupby_attr="COW"
   cmpr_attr="Income>50k"
   cmpr_list=[less_than_cmp,True]
    """
    attr_list_to_groupby = [groupby_attr, cmpr_attr]
    values = df[groupby_attr].unique()
    subgroup_val1, subgroup_val2 = subgroup_vals
    col = df.columns.values[0]
    groupby1 = df[df[subgroup_attr] == subgroup_val1].groupby(attr_list_to_groupby)[col].count()
    groupby2 = df[df[subgroup_attr] == subgroup_val2].groupby(attr_list_to_groupby)[col].count()
    return_list = []
    compare_function, compare_value = cmpr_list

    for v in values:
        if v not in groupby1.index or v not in groupby2.index:
            continue
        if compare_value not in groupby1[v].index or compare_value not in groupby2[v].index:
            continue
        group1_positives_size = groupby1[v][compare_value]
        group1_size = sum(groupby1[v])
        percent1 = (group1_positives_size / group1_size) * 100
        group2_positives_size = groupby2[v][compare_value]
        group2_size = sum(groupby2[v])
        percent2 = (group2_positives_size / group2_size) * 100

        count = np.array([group1_positives_size, group2_positives_size])
        nobs = np.array([group1_size, group2_size])
        stat, pvalue = proportions_ztest(count, nobs)

        if pvalue_filter and pvalue > PVALUE_CONST:
            continue
        if compare_function(percent1, percent2):
            print_proportions_significance_data(value=v, subgrp_attr=subgroup_attr, grp_attr=groupby_attr,
                                                cmpr_attr=cmpr_attr,
                                                val1=subgroup_val1, len_a=group1_size, percent1=percent1,
                                                val2=subgroup_val2, len_b=group2_size,
                                                percent2=percent2, p_value=pvalue)
            return_list.append(v)
    return return_list


def check_if_true_for_good_pvalue(df, split_attr, grp_attr, cmp_attr, compare_list, aggr_list, pvalue_filter=True):
    list1 = multichoice_attribute_cherrypicking(df=df, split_attr=split_attr, grp_attr=grp_attr,
                                                cmp_attr=cmp_attr,
                                                compare_list=compare_list, aggr_list=aggr_list,
                                                pvalue_filter=pvalue_filter)
    # print(len(list1))
    if len(list1) == 0:
        print("None of the subgroups have a small enough pvalue")
        return
    df, cols = get_dummies_multi_hot(df, split_attr)

    def complicated_cond(row, attrs_and_values):
        for a, values in attrs_and_values:
            if row[a] in values:
                return True
        return False

    union_df = df[df.apply(lambda row: complicated_cond(row, list1), axis=1)]

    # print(union_df)
    cmprf, val1, val2 = compare_list
    aggr_function, aggr_name, pvalue_function = aggr_list
    a_aggr, b_aggr, n_a, n_b, pvalue = validate_claim_on_subset(union_df, grp_attr, cmp_attr,
                                                                compare_list, aggr_list)
    subset_size = len(union_df)
    percent = ((subset_size / len(df)) * 100)
    print_significance_data(value="Union devtypes", col_val=";".join([x[0] for x in list1]), subset_size=subset_size,
                            percent=percent, val1=val1,
                            len_a=n_a, aggr_a=a_aggr, val2=val2, len_b=n_b, aggr_b=b_aggr, aggr_func=aggr_name,
                            p_value=pvalue)
    print(f"The result is {cmprf(a_aggr, b_aggr)}")


def whole_data_set_comparison(df, attr, cmp_attr, compare_list, pvalue_filter=False):
    cmprf, val1, val2 = compare_list
    a = df[df[attr] == val1][cmp_attr]
    b = df[df[attr] == val2][cmp_attr]
    print(a.mean())
    print(b.mean())
    column_val = attr
    a_mean, b_mean, n_a, n_b, pvalue = validate_claim_on_subset(df, column_val, cmp_attr,
                                                                compare_list)
    subset_size = len(df[df[column_val].isin([val1, val2])])
    # successful.append((v, subset_size, len(df), n_a, n_b, a_mean, b_mean, pvalue))
    percent = ((subset_size / len(df)) * 100)

    print_significance_data(value="", col_val=column_val, subset_size=subset_size, percent=percent, val1=val1,
                            len_a=n_a, mean_a=a_mean, val2=val2, len_b=n_b, mean_b=b_mean,
                            p_value=pvalue)


def interval_string_generator(value, interval_size):
    if math.isnan(value):
        return value
    return str((value // interval_size) * interval_size) + "-" + str(
        (value // interval_size) * interval_size + interval_size)


def create_significance_df(aggr_list):
    aggr_func, name, pvalue_func = aggr_list
    name1 = f"{name}1"
    name2 = f"{name}2"
    return pd.DataFrame(columns=["Attr", "Original_Attr", "Split_Attr", "Value", "True/False", "N", "Percentage(%)",
                                 "Group1", "N1", name1, "Group2", "N2", name2, "pvalue", "Time", "MI", "Anova_F_Stat",
                                 "Anova_Pvalue", "Cosine Similarity"])


def run_query_over_all_attributes(df, exclude_list, query_function, *args):
    query_result_dictionary = {}
    for col in df.columns:
        if col in exclude_list:
            continue
        query_res = query_function(col, *args)
        values = {}
        for tup in query_res:
            if len(tup) == 1:
                values[tup[0]] = 0
            else:
                values[tup[0]] = tup[1:]
        query_result_dictionary[col] = values
    return query_result_dictionary


class CherryPicker(object):
    def __init__(self, exclude_list, numeric_list, grp_attr, target_attr, compare_list, MI_dict,
                 Anova_dict, dataset_size, bucket_dict=None,
                 agg_type='mean', std_dict=None,
                 translation_dict=None,
                 main_table="my_table",
                 output_path=None, start_time=None, stop_at_time=None, metric_subset=None, full_df=None,
                 reference=None, topk=100):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.stop_at_time = stop_at_time
        self.exclude_list = exclude_list + [target_attr, grp_attr]
        self.numeric_list = numeric_list
        self.grp_attr = grp_attr
        self.target_attr = target_attr
        self.compare_list = compare_list
        self.MI_dict = MI_dict
        self.Anova_dict = Anova_dict
        self.dataset_size = dataset_size
        self.agg_type = agg_type
        self.std_dict = std_dict
        self.translation_dict = translation_dict
        self.main_table = main_table
        self.output_path = output_path
        self.engine = SQLEngineSingleton()
        if bucket_dict is not None:
            self.bucket_objects = bucket_dict
        else:
            self.bucket_objects = {}
        self.metric_subset = metric_subset
        if self.metric_subset is None:
            self.metric_subset = [DF_MI_STRING, DF_ANOVA_F_STAT_STRING, DF_COSINE_SIMILARITY_STRING, DF_COVERAGE_STRING,
                                  DF_PVALUE_STRING, 'Count STD']
        self.topk = topk

        max_MI = max(MI_dict.values())
        self.max_MI = 1 if max_MI == 0 else max_MI
        self.max_anova_f_stat = max([v[0] for v in Anova_dict.values()])
        self.semantic_sim = SemanticSim(target_attr, translation_dict)
        self.full_df = full_df
        if reference is not None:
            print("Using reference result.")
        self.reference_result = reference


    def add_metrics_to_result_dataframe(self, result_df, column_tuple, calc_on_the_spot=False):
        average_over = []
        if DF_MI_STRING in self.metric_subset:
            if column_tuple in self.MI_dict:
                result_df[DF_MI_STRING] = self.MI_dict[column_tuple]
            else:
                print(f"Missing MI for column: {column_tuple}")
                # calc on the spot
                if calc_on_the_spot:
                    mi_res = calc_mi_columns(self.full_df[[*column_tuple, self.target_attr]], self.target_attr,
                                             self.bucket_objects)
                    result_df[DF_MI_STRING] = mi_res[0]
                else:
                    result_df[DF_MI_STRING] = 0
            result_df[DF_NORMALIZED_MI_STRING] = result_df[DF_MI_STRING] / self.max_MI
            average_over.append(DF_NORMALIZED_MI_STRING)

        if DF_ANOVA_F_STAT_STRING in self.metric_subset:
            if column_tuple in self.Anova_dict:
                anova_f_stat, anova_pvalue = self.Anova_dict[column_tuple]
                result_df[DF_ANOVA_F_STAT_STRING] = anova_f_stat
                result_df[DF_ANOVA_PVALUE_STRING] = anova_pvalue
            else:
                print(f"Missing ANOVA for column: {column_tuple}")
                if calc_on_the_spot:
                    f_stat, anova_pvalue = calc_anova_for_attrs(self.full_df, column_tuple, self.target_attr,
                                                                self.bucket_objects)
                    result_df[DF_ANOVA_F_STAT_STRING] = f_stat
                    result_df[DF_ANOVA_PVALUE_STRING] = anova_pvalue
                else:
                    result_df[DF_ANOVA_F_STAT_STRING] = 0
                    result_df[DF_ANOVA_PVALUE_STRING] = 1
            result_df[DF_NORMALIZED_ANOVA_F_STAT_STRING] = result_df[DF_ANOVA_F_STAT_STRING] / self.max_anova_f_stat
            average_over.append(DF_NORMALIZED_ANOVA_F_STAT_STRING)

        for i in range(len(column_tuple)):
            result_df[f'Attr{i + 1}_str'] = result_df[f'Attr{i + 1}'].apply(
                lambda x: safe_translate(x, self.translation_dict))
            result_df[f'Value{i + 1}_str'] = result_df.apply(
                lambda row: safe_translate((row[f'Attr{i + 1}'], row[f'Value{i + 1}']),
                                           self.translation_dict), axis=1)
        if DF_COSINE_SIMILARITY_STRING in self.metric_subset:
            result_df[DF_COSINE_SIMILARITY_STRING] = self.semantic_sim.calc_cosine_sim_batch(result_df,
                                                                                             len(column_tuple))
            average_over.append(DF_COSINE_SIMILARITY_STRING)

        if self.std_dict is not None and 'Count STD' in self.metric_subset:
            result_df['Count STD'] = self.std_dict.get(column_tuple)
            average_over.append('Count STD')
        if DF_PVALUE_STRING in result_df.columns and DF_PVALUE_STRING in self.metric_subset:
            result_df[DF_INVERTED_PVALUE_STRING] = 1 - result_df[DF_PVALUE_STRING]
            average_over.append(DF_INVERTED_PVALUE_STRING)
        if DF_COVERAGE_STRING in self.metric_subset:
            result_df[DF_COVERAGE_STRING] = (result_df[DF_N1_SIZE_STRING] + result_df[
                DF_N2_SIZE_STRING]) / self.dataset_size
            average_over.append(DF_COVERAGE_STRING)
        result_df[DF_METRICS_AVERAGE] = result_df[average_over].mean(axis=1)
        return result_df

    def cherrypick_by_attributes(self, sorted_column_tuples, sample_size=None, add_metrics=True, stop_at_k=None,
                                 attr_tuple_to_num_preds=None, existing_results_df=None):
        if attr_tuple_to_num_preds is None:
            attr_tuple_to_num_preds = {}
        table_name = self.main_table
        if sample_size is not None:
            if sample_size < 1:
                fraction = sample_size
                sample_size = int(self.dataset_size * sample_size)
            else:
                fraction = sample_size / self.dataset_size
            print(f"making sample view of size {sample_size} ({fraction})")
            make_sample_view(sample_size, self.main_table, "sample_view")
            table_name = "sample_view"
        else:
            print(f"Running on full database (table: {self.main_table}.")
        results = []
        if existing_results_df is not None:
            results = [existing_results_df]
        num_results_found = 0
        base_args = {"grp_attr": self.grp_attr, "value1": self.compare_list[1], "value2": self.compare_list[2],
                     "table_name": table_name, "where_string": WHERE}
        if self.agg_type == 'count':
            runner = QueryRunnerCountDiff(self.start_time)
        elif self.agg_type == 'mean':
            compute_stat_sig = DF_PVALUE_STRING in self.metric_subset
            runner = QueryRunnerMeanDiff(self.start_time, compute_stat_sig)
            base_args["target_attr"] = self.target_attr
            base_args["min_group_size"] = MIN_GROUP_SIZE_SQL
        elif self.agg_type == 'median':
            compute_stat_sig = DF_PVALUE_STRING in self.metric_subset
            runner = QueryRunnerMedianDiff(self.start_time, compute_stat_sig)
            base_args["target_attr"] = self.target_attr
            base_args["min_group_size"] = MIN_GROUP_SIZE_SQL
        else:
            raise (f"Unsupported agg type: {self.agg_type}")
        skipped = []
        last_recall_calc = time.time()
        for column_tuple in tqdm(sorted_column_tuples):
            if stop_at_k is not None and num_results_found > stop_at_k:
                break
            if self.stop_at_time is not None and time.time() - self.start_time > self.stop_at_time:
                print("stopping - reached time limit.")
                break
            if self.reference_result is not None and time.time() - last_recall_calc > 5*60:
                last_recall_calc = time.time()
                if len(results) > 0:
                    result_so_far = pd.concat(results, axis=0).reset_index(drop=True)
                    score_recall = measure_score_recall(self.reference_result, result_so_far, k=self.topk, score_name=DF_METRICS_AVERAGE)
                    if score_recall > 0.95:
                        print("stopping - reached required score recall.")
                        break
            if should_exclude(column_tuple, self.exclude_list):
                continue
            if column_tuple in attr_tuple_to_num_preds:
                skipped.append(column_tuple)
                # print(f"Skipping over {column_tuple}")
                num_results_found += attr_tuple_to_num_preds[column_tuple]
                continue
            # before_query = time.time()
            result_df = runner.run_query({"column_tuple": column_tuple, **base_args}, self.bucket_objects)
            # after_query = time.time()
            # with open(os.path.join(OUTPUT_DIR, 'time_per_attribute_tuple_median_join.csv'), "a") as out:
            #     out.write(f"{column_tuple},{after_query-before_query}")
            # even if no results were returned, we want to know that this attr_tuple was searched already.
            attr_tuple_to_num_preds[column_tuple] = len(result_df)
            if len(result_df) > 0:
                if add_metrics:
                    result_df = self.add_metrics_to_result_dataframe(result_df, column_tuple)
                results.append(result_df)
                num_results_found += len(result_df)
        final_df = pd.concat(results, axis=0).reset_index(drop=True) if len(results) > 0 else pd.DataFrame()
        if self.output_path is not None:
            final_df.to_csv(self.output_path)
        print(f"skipped: {skipped}")
        print(f"""SQL Runtime was:{TIME_SQL_QUERIES["x"]}""")
        return final_df

    def cherrypick_by_predicates(self, predicate_order):
        """
        :param predicate_list: [(att1, v1), (att2, v2),...]
        :return: dataframe with predicates where the claim holds, with calculated metrics.
        """
        # TODO: support count aggregation
        base_args = {"grp_attr": self.grp_attr, "value1": self.compare_list[1], "value2": self.compare_list[2],
                     "target_attr": self.target_attr, "min_group_size": MIN_GROUP_SIZE_SQL,
                     "table_name": self.main_table}

        if self.agg_type == 'mean':
            runner = QueryRunnerMeanDiffPredLevel(self.start_time)
            # sql_string = open("sql_queries/mean_query_single_pred_case_when.sql", "r").read()
            # res_df_column_names = ['Value', 'mean1', 'N1', 's1', 'mean2', 'N2', 's2']
        elif self.agg_type == 'median':
            runner = QueryRunnerMedianDiffPredLevel(self.start_time)
        else:
            raise (f"Unsupported agg type {self.agg_type} in cherrypick_by_predicates")
        results = []
        for pred_column, pred_val in tqdm(predicate_order):
            # run query
            result_df = runner.run_query({'pred_column': pred_column, 'pred_value': pred_val, **base_args},
                                         self.bucket_objects)
            if len(result_df) > 0:
                result_df = self.add_metrics_to_result_dataframe(result_df, (pred_column,))
                results.append(result_df)
        final_df = pd.concat(results, axis=0).reset_index(drop=True) if len(results) > 0 else pd.DataFrame()
        if self.output_path is not None:
            final_df.to_csv(self.output_path)
        print(f"""SQL Runtime was:{TIME_SQL_QUERIES["x"]}""")
        return final_df

    def sample_guided_cherrypicking_by_attribute(self, sample_size, columns_list):
        orig_out_path = self.output_path
        if orig_out_path is not None:
            self.output_path = orig_out_path[:-4] + f'_sample_{sample_size}.csv'
        # So that early stopping by recall is not activated for the sample!
        reference_backup = self.reference_result
        self.reference_result = None
        res_df = self.cherrypick_by_attributes(columns_list, sample_size, add_metrics=True)
        self.reference_result = reference_backup
        if len(res_df) > 0:
            res_df['Attr_tuple'] = res_df.apply(lambda row: unite_attr_names_to_tuple_field(row, MAX_ATOMS), axis=1)
            # sort the attribute combinations according to the number of successful predicates they yielded.
            # sorted_cols = res_df.Attr_tuple.value_counts().reset_index()['index'].values.tolist()
            # sort the attribute combinations according to the best predicate they yielded (best according to metrics average).
            sorted_cols = res_df.groupby('Attr_tuple')['Metrics Average'].max().sort_values(
                ascending=False).index.tolist()
            remaining_cols = list(set(columns_list).difference(set(sorted_cols)))
        else:
            sorted_cols = []
            remaining_cols = columns_list
        self.output_path = orig_out_path
        sampling_time = time.time() - self.start_time
        print(f"Finished sampling, time: {sampling_time}")
        res_df = self.cherrypick_by_attributes(sorted_cols + remaining_cols, sample_size=None)
        return res_df, sampling_time


# def calc_cosine_sim_multi_atom(vec1, value_dict, column_tuple, model, translation_dict):
#     descs = []
#     for i in range(len(column_tuple)):
#         attr = column_tuple[i]
#         val = value_dict[f'Value{i+1}']  # Value1, Value2..
#         desc = safe_translate(attr, translation_dict) + " " + safe_translate((attr, val), translation_dict)
#         descs.append(desc)
#     return util.cos_sim(vec1, model.encode([" ".join(descs)]))[0].item()


def full_multichoice_attribute_cherrypicking(df, grp_attr, target_attr, compare_list, aggr_type, exclude_list,
                                             is_numeric,
                                             sorted_columns,
                                             pvalue_filter=False, output_path="", should_usesql=False,
                                             translation_dict=None, MI_dict=None, Anova_dict=None):
    start_time = time.time()
    sqlEngine = SQLEngineSingleton() if should_usesql else None
    count_query_result_dictionary = None
    query_result_dictionary = None
    aggr_list = agg_type_to_agg_list[aggr_type]
    start_time_just_sql = datetime.datetime.now()
    if should_usesql:
        count_query_result_dictionary = run_query_over_all_attributes(
            df, exclude_list, run_counting_query, sqlEngine)

        query_result_dictionary = run_query_over_all_attributes(
            df, exclude_list, run_filtering_query, sqlEngine, grp_attr, target_attr, compare_list,
            aggr_list) if should_usesql else {}
    end_time_just_sql = datetime.datetime.now()
    _, val1, val2 = compare_list
    model = SentenceTransformer(MODEL_STRING)

    target_string = translation_dict[target_attr]
    # target_string=f"{val1} vs {val2} {translation_dict[cmp_attr]}"

    target_vector = model.encode([target_string])
    csv_df = create_significance_df(aggr_list)

    # MI_dict=create_MI_dictionary(df,attr_list,cmp_attr,is_numeric)
    # Anova_dict=create_Anova_dictionary(df,attr_list,cmp_attr,is_numeric)

    attribute_time_df = pd.DataFrame(columns=["Attr", "Unique Vals", "Time"])
    exclude_list += [target_attr, grp_attr]
    compare_value_is_numeric = False
    if target_attr in is_numeric:
        compare_value_is_numeric = True

    for c_and_v in sorted_columns:
        if ';' in c_and_v:
            c, v = c_and_v.split(';')
        else:
            c = c_and_v
        cur_attr = c
        # if c in exclude_list:
        num_vals_in_col = len(count_query_result_dictionary[c]) if should_usesql else df[c].nunique()
        if c in exclude_list or num_vals_in_col <= 1:
            continue
        time_before = datetime.datetime.now()
        new_df, res_list = multichoice_attribute_cherrypicking(df=df, split_attr=cur_attr, grp_attr=grp_attr,
                                                               cmp_attr=target_attr, compare_list=compare_list,
                                                               aggr_list=aggr_list, pvalue_filter=pvalue_filter,
                                                               compare_value_numeric=compare_value_is_numeric,
                                                               sqlEngine=sqlEngine,
                                                               query_result_dict=query_result_dictionary,
                                                               count_query_result_dictionary=count_query_result_dictionary)
        time_after = datetime.datetime.now()
        if is_multivalue_attr(df, cur_attr):
            unique_values_count = len(get_unique_values(df, cur_attr))
        else:
            unique_values_count = df[cur_attr].nunique()  # len(get_unique_values(df,cur_attr))
        time_difference = (time_after - time_before).microseconds
        attribute_time_tuple = (cur_attr, unique_values_count, time_difference)
        attribute_time_df.loc[len(attribute_time_df)] = attribute_time_tuple

        for element in res_list:
            row_list = list(element)

            for row in row_list:
                row_tuple_as_list = list(row)
                split_attr = row_tuple_as_list[0]
                orig_attr = c
                value = row_tuple_as_list[1]

                # MI
                cur_attr_MI = MI_dict[split_attr]
                row_tuple_as_list = [orig_attr] + row_tuple_as_list + [
                    cur_attr_MI]  # Adding the Original Attribute and MI values
                row_tuple_as_list = [split_attr] + row_tuple_as_list  # Adding the attr attribute
                # Anova
                row_tuple_as_list += list(Anova_dict[split_attr])
                # Calculating cosine similarity

                # Checking if the Original Attribute is the same as the Split Attribute(As in, did we split the original attribute)
                if orig_attr == split_attr:
                    description_string = translation_dict[orig_attr] + " " + str(value)
                else:
                    description_string = translation_dict[orig_attr] + " " + split_attr.split(";")[1]

                description_vector = model.encode([description_string])
                row_tuple_as_list.append(util.cos_sim(target_vector, description_vector)[0].item())
                row = tuple(row_tuple_as_list)
                csv_df.loc[len(csv_df)] = row
        print("------------------------------------------")

    csv_df["Time"] = csv_df["Time"].apply(lambda x: x - start_time)
    if output_path == "":
        print("Cannot write to output path.")
        return csv_df
    csv_df.to_csv(output_path)

    attribute_time_df.to_csv("data/attribute_time.csv")
    if sqlEngine is not None:
        sqlEngine.dispose()
    print(f"sql_time:{end_time_just_sql - start_time_just_sql}")
    print(f"Number of values whose subset was too small: {COUNT_FILTERED_OUT_BY_SIZE['x']}")
    print(f"Time for Anova Calcualtion: {TIME_ANOVA_CALCULATION['x']}")
    return csv_df


def count_groups_over_size(attr_tuple, df, size):
    return sum(df[[*attr_tuple]].value_counts() > size)


def read_metrics_from_path(df_path):
    results = pd.read_csv(df_path)
    results['anova'] = results[['anova_f_stat', 'anova_p_value']].apply(lambda row: tuple(row.values), axis=1)
    # from a string representing a tuple to an actual tuple
    results['attr_tuple'] = results['attr_tuple'].apply(eval)
    results = results.set_index('attr_tuple')
    anova_dict = results['anova'].to_dict()
    mi_dict = results['mi'].to_dict()
    cosine_sim_dict = results['cosine_sim'].to_dict()
    min_value_count = results['min_value_count'].to_dict()
    max_group_size = results['max_group_size'].to_dict()
    num_groups_over100 = results['num_groups_over100'].to_dict()
    num_groups_over500 = results['num_groups_over500'].to_dict()
    return anova_dict, mi_dict, cosine_sim_dict, min_value_count, max_group_size, num_groups_over100, num_groups_over500

def process_atom_combination(args):
    #attr_tuple = atom_combination
    attr_tuple, df, attr_to_num_values, already_processed, target_attr, bucket_objects = args
    print(f"Working on {attr_tuple}")
    if attr_tuple in already_processed:
        print("Already processed")
        return
    min_value_count = min([attr_to_num_values[attr] for attr in attr_tuple])
    vc = df[[*attr_tuple]].value_counts()
    max_group_size = vc.max()
    num_over_100 = sum(vc > 100)
    num_over_500 = sum(vc > 500)
    try:
        mi_res = calc_mi_columns(df[[*attr_tuple, target_attr]], target_attr, bucket_objects)
    except Exception as e:
        mi_res = "recalc!"
    try:
        f_stat, anova_pvalue = calc_anova_for_attrs(df, attr_tuple, target_attr, bucket_objects)
    except Exception as e:
        f_stat, anova_pvalue = "recalc!", "recalc!"
    return tuple(sorted(attr_tuple)), f_stat, anova_pvalue, mi_res, min_value_count, max_group_size, num_over_100, num_over_500


def progressive_metrics_calc(atom_combinations, target_attr, max_atoms=1, translation_dict={}, bucket_objects={}):
    output_path = os.path.join(DATA_PATH, f"metrics_{max_atoms}atoms_{target_attr}.csv")
    if os.path.exists(output_path):
        # FORMAT: attr_tuple, anova_f_stat, anova_p_value, mi, cosine sim
        results = pd.read_csv(output_path)
        print(f"calculated metrics: {len(results)}, atom combinations: {len(atom_combinations)}")
        if len(results) >= len(atom_combinations):
            return read_metrics_from_path(output_path)
        results['attr_tuple'] = results['attr_tuple'].apply(eval)
        already_processed = set(results.attr_tuple.values)
        # out = open(output_path, "a")
    else:
        already_processed = []
        with open(output_path, "w") as out:
            out.write(
                "attr_tuple,anova_f_stat,anova_p_value,mi,cosine_sim,min_value_count,max_group_size,num_groups_over100,num_groups_over500\n")
    df = pd.read_csv(DATAFRAME_PATH, index_col=0)
    # For num values:
    attr_to_num_values = {}
    for col in df.columns:
        attr_to_num_values[col] = df[col].nunique()
    args_list = [(atom_combination, df, attr_to_num_values, already_processed, target_attr, bucket_objects)
                 for atom_combination in atom_combinations]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(process_atom_combination, args_list)
    pool.close()
    pool.join()
    print("now calculating cosine sim and writing results.")
    # For cosine similarity:
    semantic_sim = SemanticSim(target_attr, translation_dict)
    with open(output_path, "a") as out:
        for result in results:
            if result:
                attr_tuple, f_stat, anova_pvalue, mi_res, min_value_count, max_group_size, num_over_100, num_over_500 = result
                emb_sim = semantic_sim.calc_cosine_sim_attr_level(attr_tuple)
                result_str = f'"{attr_tuple}",{f_stat},{anova_pvalue},{mi_res},{emb_sim},{min_value_count},{max_group_size},{num_over_100},{num_over_500}\n'
                out.write(result_str)

    # for i, attr_tuple in enumerate(atom_combinations):
    #     print(f"Working on {attr_tuple}, {i}/{len(atom_combinations)}")
    #     if attr_tuple in already_processed:
    #         print("Already proccessed")
    #         continue
    #     min_value_count = min([attr_to_num_values[attr] for attr in attr_tuple])
    #     vc = df[[*attr_tuple]].value_counts()
    #     max_group_size = vc.max()
    #     num_over_100 = sum(vc > 100)
    #     num_over_500 = sum(vc > 500)
    #     try:
    #         mi_res = calc_mi_columns(df[[*attr_tuple, target_attr]], target_attr, bucket_objects)
    #     except:
    #         mi_res = "recalc!"
    #     try:
    #         f_stat, anova_pvalue = calc_anova_for_attrs(df, attr_tuple, target_attr, bucket_objects)
    #     except:
    #         f_stat, anova_pvalue = "recalc!", "recalc!"
    #     # We ignore the field value at this point since this is done at preprocessing time.
    #     emb_sim = semantic_sim.calc_cosine_sim_attr_level(attr_tuple)
    #     with open(output_path, "a") as out:
    #         out.write(
    #             f'"{tuple(sorted(attr_tuple))}",{f_stat},{anova_pvalue},{mi_res},{emb_sim},{min_value_count},{max_group_size},{num_over_100},{num_over_500}\n')
    return read_metrics_from_path(output_path)


def create_metrics_dictionary(df, atom_combinations, target_attr, max_atoms=1,
                              calc_mi=True, calc_anova=True, calc_emb_sim=True, translation_dict={}, bucket_objects={}):
    mi_dictionary = {}
    anova_dictionary = {}
    emb_sim_dictionary = {}
    if not calc_mi and not calc_anova and not calc_emb_sim:
        return mi_dictionary, anova_dictionary, emb_sim_dictionary
    anova_path = os.path.join(DATA_PATH, f"Anova_{max_atoms}atoms_{target_attr}.pickle")
    mi_path = os.path.join(DATA_PATH, f"MI_{max_atoms}atoms_{target_attr}.pickle")
    emb_sim_path = os.path.join(DATA_PATH, f"emb_sim_{max_atoms}atoms_{target_attr}.pickle")

    recalc_mi, recalc_anova, recalc_emb_sim = True, True, True
    if os.path.exists(anova_path):
        recalc_anova = False
        anova_dictionary = pickle.load(open(anova_path, 'rb'))
    if os.path.exists(mi_path):
        recalc_mi = False
        mi_dictionary = pickle.load(open(mi_path, 'rb'))
    if os.path.exists(emb_sim_path):
        recalc_emb_sim = False
        emb_sim_dictionary = pickle.load(open(emb_sim_path, 'rb'))

    if not recalc_mi and not recalc_anova and not recalc_emb_sim:
        return mi_dictionary, anova_dictionary, emb_sim_dictionary

    if calc_emb_sim:
        model = SentenceTransformer(MODEL_STRING)
        target_string = utils.safe_translate(target_attr, translation_dict)
        target_vector = model.encode([target_string])

    print("Metrics calculation started.")
    # multi_atom_combinations = []
    # for num_atoms in range(1, max_atoms+1):
    #     multi_atom_combinations += list(itertools.combinations(attr_list, num_atoms))
    b = time.time()
    # for i, attr in enumerate(attr_list):
    for i, attr_tuple in enumerate(atom_combinations):
        print(f"Working on {attr_tuple}, {i}/{len(atom_combinations)}")
        # current_attributes = [attr]
        # curr_df = df
        # if is_multivalue_attr(df, attr):
        # print(f"Encountered multi value column in anova/mi calculation: {attr}")
        # curr_df, new_columns = get_dummies_multi_hot(df[[attr, target_attr]].copy(), attr)
        # current_attributes = new_columns
        # for curr_attr in current_attributes:
        if calc_mi and recalc_mi:
            mi_dictionary[attr_tuple] = calc_mi_columns(df[[*attr_tuple, target_attr]], target_attr, bucket_objects)
        if calc_anova and recalc_anova:
            f_stat, anova_pvalue = calc_anova_for_attrs(df, attr_tuple, target_attr, bucket_objects)
            anova_dictionary[attr_tuple] = (f_stat, anova_pvalue)
        if calc_emb_sim and recalc_emb_sim:
            descriptions = []
            for attr in attr_tuple:
                # We ignore the field value at this point since this is done at preprocessing time.
                descriptions.append(utils.safe_translate(attr, translation_dict))
            description_vector = model.encode([" ".join(descriptions)])
            emb_sim_dictionary[attr_tuple] = util.cos_sim(target_vector, description_vector)[0].item()
    a = time.time()
    print(f"Metrics calculation took {a - b} seconds.")
    if recalc_anova:
        pickle.dump(anova_dictionary, open(anova_path, 'wb'))
    if recalc_mi:
        pickle.dump(mi_dictionary, open(mi_path, 'wb'))
    if recalc_emb_sim:
        pickle.dump(emb_sim_dictionary, open(emb_sim_path, 'wb'))
    return mi_dictionary, anova_dictionary, emb_sim_dictionary


def create_count_std_dictionary(atom_combinations, bucket_dict, start_time):
    std_dict = {}
    runner = QueryRunnerCountSTD(start_time=start_time)
    for comb in atom_combinations:
        query_args = {"column_tuple": comb, "table_name":"my_table"}
        std_dict[comb] = runner.run_query(query_args, bucket_dict)
    return std_dict


def calc_anova_for_attrs(df, attr_iterable, target_attr, bucket_dict):
    # if is_multivalue_attr(df,attr):
    #     new_df, new_columns = get_dummies_multi_hot(df, attr)
    group_by = []
    for attr in attr_iterable:
        if attr in bucket_dict:
            bucket = bucket_dict[attr]
            df[f'{attr}_bucket'] = df[attr].apply(lambda v: bucket.value_to_bucket_id(v))
            group_by.append(f'{attr}_bucket')
        else:
            group_by.append(attr)
    smp_attr_grouped = df[~df[target_attr].isna()].groupby(group_by)[target_attr].apply(list).values.tolist()
    # Can't calculate anova with less than 2 groups.
    if len(smp_attr_grouped) <= 1:
        f_stat, anova_pvalue = 0, 1
    else:
        f_stat, anova_pvalue = sp.stats.f_oneway(*smp_attr_grouped)
    return f_stat, anova_pvalue


def remove_outliers(df, attr):
    before = len(df)
    df = df.dropna(subset=[attr]).copy()
    print(f"removed {before - len(df)} rows where {attr} is none")
    values = df[attr]
    q1, q3 = values.quantile([0.25, 0.75])
    IQR = q3 - q1
    LW = q1 - 1.5 * IQR
    UW = q3 + 1.5 * IQR
    len_orig = len(df)
    df = df[df[attr] >= LW]
    df = df[df[attr] <= UW]
    len_new = len(df)
    print(
        f"Removed Outliers: {len_orig - len_new} Rows Deleted, {round((len_orig - len_new) / len_orig * 100, 2)}% of the dataset")
    return df


def calc_mi(df, attr, is_numeric):
    # Can accept more than one attribute to calculate MI for
    df1 = df.dropna()
    X = df1.drop(attr, axis='columns')
    Y = df1[attr].values.astype(np.float32)
    discrete = [i for i, c in enumerate(X.columns) if c not in is_numeric]
    if attr in is_numeric:
        mi = mutual_info_regression(X.values, Y, discrete_features=discrete)
    else:
        mi = mutual_info_classif(X.values, Y, discrete_features=discrete)
    mi_dict = {}
    for col in zip(X.columns, mi):
        mi_dict[col[0]] = col[1]
    return mi_dict


def calc_mi_columns(df, target_attr, buckets_dict):
    # Gets a 2 column dataframe, calculates mi between them.
    # replace categorical values with numbers.
    df_local = df.copy()
    df_local = df_local.dropna()
    if len(df_local) == 0:  # all rows had NAs
        return 0
    cols = []
    # apply discretization
    for col in df_local.columns:
        if col == target_attr:
            continue
        if col in buckets_dict:
            bucket = buckets_dict[col]
            df_local[f'{col}_bucket'] = df_local[col].apply(lambda v: bucket.value_to_bucket_id(v))
            cols.append(f'{col}_bucket')
            print("Applied bucketing")
        else:
            cols.append(col)
    if len(cols) > 1:
        print(f"combining values of {cols}")
        df_local['combined'] = df_local[cols].apply(lambda row: tuple(row.values), axis=1)
    else:  # len(cols) == 1
        df_local.rename({cols[0]: 'combined'}, axis=1, inplace=True)
    # encode values into ordinal numbers, ordered by the mean of the target attribute.
    encoder = OrdinalEncoder(encoding_method='ordered', variables=['combined'], ignore_format=True)
    encoder.fit(df_local, df_local[target_attr])
    df_local = encoder.transform(df_local)

    if target_attr in buckets_dict:
        mi = mutual_info_regression(df_local['combined'].values.reshape(-1, 1), df_local[target_attr].values,
                                    discrete_features=[True])
    else:
        mi = mutual_info_classif(df_local['combined'].values.reshape(-1, 1), df_local[target_attr].values,
                                 discrete_features=[True])
    return mi


def create_filtered_csv(df, numeric_fields, bucket_fields, exclude_list, date_fields_dict, output_csv_name,
                        attribute_info_file_name, create_db_flag):
    """
    :param df:
    :param numeric_fields: fields that should be discretized
    :param bucket_fields: fields that should be treated as intervals (in post processing)
    :param exclude_list:
    :param date_fields_dict: from field name to datetime.strptime format
    :param output_csv_name:
    :param attribute_info_file_name:
    :return:
    """
    # print(f"orig len: {len(df)}")
    attributes = df.columns
    bucket_fields = set(bucket_fields)
    numeric_fields = set(numeric_fields)
    exclude_list = set(exclude_list)
    for attr in attributes:
        if attr in numeric_fields and attr not in exclude_list:
            # TODO: we are actually dropping all nulls in all numeric columns - not sure we want to do that
            # if attr == "PINCP":
            #     df = df.dropna(subset=[attr]).copy()
            #     #TODO: is it always int? Never float?
            # df[attr] = df[attr].astype(np.float64)
            df[attr] = pd.to_numeric(df[attr])
            #     continue
            # dropped_nan_df=df.dropna(subset=[attr]).copy()
            # print(dropped_nan_df[attr].max())
            # num1=float(dropped_nan_df[attr].max())
            # print(num1+1)
            interval = value_range_to_interval_size(df[attr].max())
            bucket_fields.add(attr)
            # cur_attr = "__" + c + "_split"
            df[attr] = df[attr].apply(lambda x: interval_string_generator(x, interval))
        elif attr in date_fields_dict:
            # This logic assumes that specific days are too small bins, and only keeps month and year.
            print(attr)
            dates = df[attr].apply(lambda x: datetime.datetime.strptime(x, date_fields_dict[attr]))
            df[f"{attr}_year"] = dates.apply(lambda x: x.year)
            df[f"{attr}_month"] = dates.apply(lambda x: x.month)
            df[f"{attr}_year_month"] = dates.apply(lambda x: f"{x.year}_{x.month}")
            exclude_list.add(attr)

    df.to_csv(path_or_buf=output_csv_name, index=False)
    # Record the new special attributes
    numeric_fields = numeric_fields.difference(bucket_fields)
    pickle.dump((numeric_fields, bucket_fields, exclude_list), open(attribute_info_file_name, 'wb'))
    # print(f" discretized len: {len(df)}")
    # connect_sql_db
    if create_db_flag:
        df_to_sql_DB(df)
    return df


def generate_all_preds(df, exclude_list, bucket_dict):
    # TODO: support multi atom preds
    s = time.time()
    preds = []
    for c in df.columns:
        if c in exclude_list:
            continue
        if c in bucket_dict:
            # generate all buckets?
            values = bucket_dict[c].get_all_buckets()
        else:
            values = df[c].unique()
        for v in values:
            preds.append((c, v))
    print(f"Found {len(preds)} predicates.")
    print(f"time to create all predicates: {time.time() - s} seconds.")
    return preds


def setup_cherrypicking(df, exclude_list, is_numeric, trans_dict, start_time, do_regression=False,
                        precompute_metrics=True, multi_atom_combinations=None):
    target_attr = TARGET_ATTR
    attr_list = list(set(df.columns).difference(set(exclude_list + [target_attr, GRP_ATTR])))
    # trans = utils.make_translation_for_fields_list(df.columns)
    bucket_objects = {}
    for attr_name in is_numeric:
        bucket = Bucket.from_attr_name(attr_name)
        bucket_objects[attr_name] = bucket
        print(f"{attr_name} bucket: {bucket.low}, {bucket.high}, {bucket.count}")
    if multi_atom_combinations is None:
        multi_atom_combinations = []
        for num_atoms in range(1, MAX_ATOMS + 1):
            multi_atom_combinations += list(itertools.combinations(attr_list, num_atoms))
        # keep the attribute tuples sorted lexicographically to avoid duplicates
        multi_atom_combinations = [tuple(sorted(x)) for x in multi_atom_combinations]
    Anova_dict, MI_dict, emb_sim_dict, min_value_count_dict, max_group_size_dict, num_groups_over100, num_groups_over500 = {}, {}, {}, {}, {}, {}, {}
    if precompute_metrics:
        Anova_dict, MI_dict, emb_sim_dict, min_value_count_dict, max_group_size_dict, num_groups_over100, num_groups_over500 = progressive_metrics_calc(
            multi_atom_combinations, target_attr, max_atoms=MAX_ATOMS, translation_dict=trans_dict,
            bucket_objects=bucket_objects)
    std_dict = None
    if AGG_TYPE == 'count' and precompute_metrics and 'Count STD' in METRICS_SUBSET:
        # TODO: test 2-atoms
        std_dict = create_count_std_dictionary(multi_atom_combinations, bucket_objects, start_time)
    reg_weight_dict = None
    if do_regression:
        reg_weight_dict = regression_for_feature_weight(df, TARGET_ATTR, GRP_ATTR, COMPARE_LIST[1:], exclude_list,
                                                        STRING_COLS, return_only_dict=True)
    # Remove single value columns and all their combinations. Also remove attr_tuples with groups that are too small.
    before = len(multi_atom_combinations)
    multi_atom_combinations = [attr_tuple for attr_tuple in multi_atom_combinations
                               if min_value_count_dict[attr_tuple] > 1
                               and max_group_size_dict[attr_tuple] > GROUP_SIZE_FILTER_THRESHOLD]
    print(f"Removed {before - len(multi_atom_combinations)} attr_tuples by min_value_count and max_group_size.")
    return Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500


def sort_attr_tuples(attr_tuples, sort_by, Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict,
                     num_large_groups, iter_index=1):
    if sort_by == DF_ANOVA_PVALUE_STRING:
        # Smaller p-value = more interesting
        sorted_by = sorted(attr_tuples, key=lambda col: Anova_dict[col][1])
    elif sort_by == DF_ANOVA_F_STAT_STRING:
        # Larger F stat = more interesting
        sorted_by = sorted(attr_tuples, key=lambda col: Anova_dict[col][0], reverse=True)
    elif sort_by == DF_COSINE_SIMILARITY_STRING:
        # Larger Cosine Similarity = more interesting
        sorted_by = sorted(attr_tuples, key=lambda col: emb_sim_dict[col], reverse=True)
    elif sort_by == DF_MI_STRING:
        # Larger MI = more interesting
        sorted_by = sorted(attr_tuples, key=lambda col: MI_dict[col], reverse=True)
    elif sort_by == 'REGRESSION':
        # sum the weights of the attributes in the tuple.
        sorted_by = sorted(attr_tuples, key=lambda cols: sum([reg_weight_dict[col] for col in cols]), reverse=True)
    elif sort_by == 'NUM_LARGE_GROUPS':
        sorted_by = sorted(attr_tuples, key=lambda col: num_large_groups[col], reverse=True)
    elif sort_by == 'COUNT_STD':
        sorted_by = sorted(attr_tuples, key=lambda col: std_dict[col], reverse=True)
    elif sort_by == 'random_shuffle':
        sorted_by = attr_tuples
        random.shuffle(sorted_by)
    elif sort_by == 'original_order':
        sorted_by = attr_tuples
    elif sort_by == 'read_sorted':
        print(f"reading sorted attr combs from: {os.path.join(OUTPUT_DIR, f'attr_comb_shuffle{iter_index}.pickle')}")
        sorted_by = pickle.load(open(os.path.join(OUTPUT_DIR, f"attr_comb_shuffle{iter_index}.pickle"), "rb"))
    else:
        raise (f"Unsupported sorting by {sort_by}")
    return sorted_by


def run_cherrypicking(df, exclude_list, is_numeric, trans_dict, metrics_subset,
                      output_path, main_table, sort_by, sample_size, start_time,
                      target_attr=TARGET_ATTR, grp_attr=GRP_ATTR, compare_list=COMPARE_LIST, agg_type=AGG_TYPE,
                      shuffle_iter_index=None,
                      reference=None):
    # incorporate all the possible running variations.
    start = time.time()
    preprocessing_time = 0
    Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500 = setup_cherrypicking(
        df, exclude_list, is_numeric, trans_dict, start_time,
        do_regression=(sort_by in ('REGRESSION', 'ALL_TOP_K_MERGED', 'ALL_TOP_K_SERIAL')))
    ##################################
    print("Cherrypicking is starting.")
    if USE_SQL:
        cherry_picker = CherryPicker(exclude_list=exclude_list, numeric_list=is_numeric,
                                     grp_attr=grp_attr, target_attr=target_attr,
                                     compare_list=compare_list, MI_dict=MI_dict, Anova_dict=Anova_dict,
                                     dataset_size=get_DB_size(main_table), bucket_dict=bucket_objects,
                                     agg_type=agg_type,
                                     translation_dict=trans_dict,
                                     std_dict=std_dict,
                                     output_path=output_path, start_time=start_time, stop_at_time=STOP_AT_TIME,
                                     metric_subset=metrics_subset, main_table=main_table, reference=reference)
        if sort_by == 'ALL_TOP_K_MERGED':
            print("method: merge rankings interleaving")
            merged_sorted = combine_top_k_from_pre_known_metrics_merged(
                Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations,
                num_groups_over100, num_groups_over500)
            preprocessing_time += (time.time() - start)
            print(f"preprocessing time: {preprocessing_time}")
            result_df = cherry_picker.cherrypick_by_attributes(merged_sorted)
        elif sort_by == 'ALL_TOP_K_SERIAL':
            print("method: fetch top k from each")
            result_df = combine_top_k_from_pre_known_metrics(
                cherry_picker, 100, Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict,
                multi_atom_combinations,
                num_groups_over100, num_groups_over500)
        elif sample_size is not None:
            print("method: sampling")
            sorted_by = sort_attr_tuples(multi_atom_combinations, sort_by, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                         reg_weight_dict, num_groups_over100)
            preprocessing_time += (time.time() - start)
            print(f"preprocessing time, not including sampling: {preprocessing_time}")
            result_df, sampling_time = cherry_picker.sample_guided_cherrypicking_by_attribute(sample_size=sample_size,
                                                                                              columns_list=sorted_by)
            preprocessing_time += sampling_time
            print(f"preprocessing time, including sampling: {preprocessing_time}")
        else:
            print(f"method: sort by {sort_by}")
            sorted_by = sort_attr_tuples(multi_atom_combinations, sort_by, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                         reg_weight_dict, num_groups_over100, shuffle_iter_index)
            preprocessing_time += (time.time() - start)
            result_df = cherry_picker.cherrypick_by_attributes(sorted_by)
    else:  # not using SQL
        sorted_by = sort_attr_tuples(multi_atom_combinations, sort_by, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                     reg_weight_dict, num_groups_over100)
        result_df = full_multichoice_attribute_cherrypicking(
            df=df, grp_attr=GRP_ATTR, target_attr=TARGET_ATTR, compare_list=COMPARE_LIST, aggr_type=AGG_TYPE,
            exclude_list=exclude_list, is_numeric=is_numeric, pvalue_filter=PVALUE_FILTER,
            translation_dict=trans_dict, sorted_columns=sorted_by, MI_dict=MI_dict, Anova_dict=Anova_dict,
            output_path=OUTPUT_PATH)
    print(f"Total runtime: {time.time() - start}")
    return result_df, preprocessing_time


def run_cherrypicking_with_config(df, exclude_list, is_numeric, trans_dict):
    run_cherrypicking(df, exclude_list, is_numeric, trans_dict, metrics_subset=METRICS_SUBSET,
                      output_path=OUTPUT_PATH, main_table="my_table", sort_by=SORT_BY, sample_size=SAMPLE_SIZE,
                      start_time=time.time())


def shuffle_and_save(df, exclude_list, is_numeric, trans_dict, start_time, iters=3):
    sort_by = 'random_shuffle'
    Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500 = setup_cherrypicking(
        df, exclude_list, is_numeric, trans_dict, start_time,
        do_regression=False)
    for i in range(iters):
        sorted_by = sort_attr_tuples(multi_atom_combinations, sort_by, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                     reg_weight_dict, num_groups_over100)
        pickle.dump(sorted_by, open(os.path.join(OUTPUT_DIR, f"attr_comb_shuffle{i + 1}.pickle"), "wb"))


def combine_top_k_from_pre_known_metrics(cherry_picker, top_k,
                                         Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict,
                                         multi_atom_combinations, num_groups_over100, num_groups_over500):
    metrics = [DF_ANOVA_F_STAT_STRING, DF_MI_STRING, DF_COSINE_SIMILARITY_STRING, 'REGRESSION', 'NUM_LARGE_GROUPS']
    combined_res_df = None
    tuple_to_found_preds = {}
    # We save the attr_tuples already covered and not run them again
    for metric in metrics:
        start_metric_time = time.time()
        # sort by metric
        # TODO: maybe use "over500" instead of "over100"
        sorted_by = sort_attr_tuples(multi_atom_combinations, metric, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                     reg_weight_dict, num_groups_over100)
        result_df = cherry_picker.cherrypick_by_attributes(sorted_by, sample_size=None, stop_at_k=top_k,
                                                           attr_tuple_to_num_preds=tuple_to_found_preds)
        if combined_res_df is None:
            combined_res_df = result_df
        else:
            combined_res_df = pd.concat([combined_res_df, result_df])
        print(f"Time for metric {metric}: {time.time() - start_metric_time}")
    remaining_atom_combinations = [attr_tuple for attr_tuple in multi_atom_combinations if
                                   attr_tuple not in tuple_to_found_preds]
    # print(f"remaining combinations: {len(remaining_atom_combinations)}")
    # if we want to sort the remaining combinations by anything:
    # sorted_by = sort_attr_tuples(remaining_atom_combinations, "original_order", Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, num_groups_over100)
    result_df = cherry_picker.cherrypick_by_attributes(remaining_atom_combinations, sample_size=None,
                                                       attr_tuple_to_num_preds=tuple_to_found_preds,
                                                       existing_results_df=combined_res_df)
    combined_res_df = pd.concat([combined_res_df, result_df])

    # the intermediate dataframes are also saved to this path, but now we will overwrite them.
    combined_res_df.to_csv(cherry_picker.output_path)
    return combined_res_df


def combine_top_k_from_pre_known_metrics_merged(
        Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict,
        multi_atom_combinations, num_groups_over100, num_groups_over500):
    start = time.time()
    metrics = [DF_ANOVA_F_STAT_STRING, DF_MI_STRING, DF_COSINE_SIMILARITY_STRING, 'NUM_LARGE_GROUPS', 'REGRESSION']
    ranks = []
    for metric in metrics:
        # TODO: maybe use "over500" instead of "over100"
        sorted_by = sort_attr_tuples(multi_atom_combinations, metric, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                     reg_weight_dict, num_groups_over100)
        ranks.append(sorted_by)
    # now merge them into one ranked list
    # merge_start = time.time()
    indexes = [0 for metric in metrics]
    curr_list = 0
    merged = []
    while min(indexes) < len(multi_atom_combinations):
        if indexes[curr_list] < len(ranks[curr_list]):
            cand = ranks[curr_list][indexes[curr_list]]
            if cand not in merged:
                merged.append(cand)
            indexes[curr_list] += 1
        curr_list = (curr_list + 1) % len(metrics)
    processing_time = time.time() - start
    print(f"merged {len(ranks)} ranked lists. Final list length: {len(merged)}. Time to merge: {time.time() - start}")
    return merged


def regression_for_feature_weight(df, target_attr, group_attr, group_value_pair, exclude, categoricals,
                                  should_test=False, analyze_results=False, return_only_dict=True):
    s = time.time()
    attrs = [c for c in df.columns if c not in exclude and c != target_attr and c != group_attr]
    df_local = df[attrs + [target_attr, group_attr]].copy()
    df_local = df_local.dropna(subset=target_attr)
    df_local = prepare_for_regression(df_local, DATA_PATH, attrs)  # mainly handle binary fields
    string_cols = [c for c in categoricals if c in df_local.columns]
    if len(string_cols) > 0:
        encoder = OrdinalEncoder(encoding_method='ordered', variables=string_cols, ignore_format=True,
                                 missing_values='ignore')
        encoder.fit(df_local, df_local[target_attr])
        df_g1 = encoder.transform(df_local[df_local[group_attr] == group_value_pair[0]])
        df_g2 = encoder.transform(df_local[df_local[group_attr] == group_value_pair[1]])
    else:
        df_g1 = df_local[df_local[group_attr] == group_value_pair[0]]
        df_g2 = df_local[df_local[group_attr] == group_value_pair[1]]
    df_g1 = df_g1.fillna(-1).drop(columns=[group_attr])
    df_g2 = df_g2.fillna(-1).drop(columns=[group_attr])
    tp = time.time()
    print(f"time to fit and transform categorical values: {tp - s}")
    if should_test:
        for i in range(2):
            group_df = [df_g1, df_g2][i]
            train, test = train_test_split(group_df, test_size=0.25)
            reg1 = LinearRegression().fit(train[attrs].values, train[target_attr].values)
            train_score = reg1.score(train[attrs].values, train[target_attr].values)
            test_score = reg1.score(test[attrs].values, test[target_attr].values)
            print(f"Model R2 score on group {group_value_pair[i]}: train: {train_score} test: {test_score}")
            c = train[target_attr].mean()
            ypred = np.ones(len(test)) * c
            baseline_rmse = mean_squared_error(test[target_attr], ypred, squared=False)
            pred_rmse = mean_squared_error(test[target_attr], reg1.predict(test[attrs].values), squared=False)
            print(f"Test RMSE: {pred_rmse}, vs. RMSE predicting the train mean ({c}): {baseline_rmse}")
    t_after_test = time.time()
    reg1 = LinearRegression().fit(df_g1[attrs].values, df_g1[target_attr].values)
    tp1 = time.time()
    print(f"time to fit regression for group 1: {tp1 - t_after_test}, {len(df_g1)} rows")
    reg2 = LinearRegression().fit(df_g2[attrs].values, df_g2[target_attr].values)
    print(f"time to fit regression for group 2: {time.time() - tp1}, {len(df_g2)} rows")
    attrs_and_scores = zip(attrs, reg2.coef_ - reg1.coef_)
    print(f"total time for regression: {time.time() - s}")
    if analyze_results:
        analyze_regression(reg1, reg2, attrs, encoder.encoder_dict_, df, group_attr, target_attr)
    if return_only_dict:
        return dict(attrs_and_scores)
    return dict(attrs_and_scores), reg1, reg2, encoder


def run_random_shuffle_over_full_table(df, exclude_list, is_numeric, trans_dict, iters):
    sort_by = 'read_sorted'
    for i in range(1, iters + 1):
        start = time.time()
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{DATABASE_NAME}_{AGG_TYPE}_{MAX_ATOMS}atoms_{QUERY_DESC}_random_shuffle{i}_guided.csv")
        run_cherrypicking(df, exclude_list, is_numeric, trans_dict, METRICS_SUBSET,
                          output_path, "my_table", sort_by, None, start, i)


def num_tuples_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, trans_dict, methods, stop_at_recall=False):
    for size in sample_sizes:
        sample_view_name = f"db_size_exp_{size}"
        make_sample_view(size, from_table="my_table", to_table=f"db_size_exp_{size}", drop_if_existing=True)
        reference = None
        for method in methods:
            print(f"running: method {method} with {size} tuples")
            start = time.time()
            TIME_SQL_QUERIES["x"] = datetime.timedelta(seconds=0)
            iter_index = None
            method_sample = None
            sort_by = method
            if 'sample' in method:
                sort_by = 'original_order'
                method_sample = float(method.split(':')[0][:-6])  # e.g., '0.01sample:3' -> 0.01
            elif method.startswith('random_shuffle'):  # e.g., "random_shuffle:1"
                sort_by = 'read_sorted'
                iter_index = int(method.split(":")[1])
            output_path = os.path.join(
                OUTPUT_DIR, "db_size_sensitivity",
                f"{DATABASE_NAME}_{AGG_TYPE}_{MAX_ATOMS}atoms_{QUERY_DESC}_{method.replace(':', '')}_guided_{size}_tuples.csv")
            res_df, preprocessing_time = run_cherrypicking(
                df, exclude_list, is_numeric, trans_dict, METRICS_SUBSET,
                output_path, sample_view_name, sort_by, method_sample, start,
                target_attr=TARGET_ATTR, grp_attr=GRP_ATTR, compare_list=COMPARE_LIST, agg_type=AGG_TYPE,
                shuffle_iter_index=iter_index,
                reference=reference)
            if method == 'original_order' and stop_at_recall:
                print("Setting reference df.")
                reference = res_df
            end = time.time()
            with open(os.path.join(OUTPUT_DIR, "num_tuples_exp.txt"), "a") as out_file:
                out_file.write(
                    f"Method: {method}, Num tuples: {size}, time for full cp: {end - start} seconds, {preprocessing_time} for setup,"
                    f" {TIME_SQL_QUERIES['x']} time for SQL\n")


def num_columns_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, trans_dict, methods, iterations=1,
                                     column_subsets=None, stop_at_recall=False):
    for size in sample_sizes:
        for i in range(iterations):
            # Sample <size> columns, put everything else in exclude.
            possible_cols = [c for c in df.columns if c not in exclude_list]
            if len(possible_cols) < size:
                return
            if column_subsets is not None:
                col_sample = column_subsets[size]
            else:
                col_sample = random.sample(possible_cols, size)
            print(f"sampled cols: {col_sample}")
            with open(os.path.join(OUTPUT_DIR, "num_columns_exp.txt"), "a") as out_file:
                out_file.write(f"{size} sampled cols: {col_sample}\n")
            additional_exclusion = [c for c in possible_cols if c not in col_sample]
            reference = None
            for method in methods:
                start = time.time()
                TIME_SQL_QUERIES["x"] = datetime.timedelta(seconds=0)
                iter_index = None
                sample_str = ''
                method_sample = None
                sort_by = method
                if 'sample' in method:
                    sort_by = 'original_order'
                    method_sample = float(method.split(":")[0][:-6])
                    sample_str = '_' + method
                elif method.startswith('random_shuffle'):  # e.g., "random_shuffle:1"
                    sort_by = 'read_sorted'
                    iter_index = int(method.split(":")[1])

                output_path = os.path.join(
                    OUTPUT_DIR, "db_width_sensitivity",
                    f"{DATABASE_NAME}_{AGG_TYPE}_{MAX_ATOMS}atoms_{QUERY_DESC}_{method.replace(':', '')}_guided_{size}cols_iter{i}.csv")
                result_df, preprocessing_time = run_cherrypicking(
                    df, exclude_list + additional_exclusion, is_numeric, trans_dict, METRICS_SUBSET,
                    output_path, 'my_table', sort_by, method_sample, start,
                    target_attr=TARGET_ATTR, grp_attr=GRP_ATTR, compare_list=COMPARE_LIST, agg_type=AGG_TYPE,
                    shuffle_iter_index=iter_index,
                    reference=reference)
                end = time.time()
                if method == 'original_order' and stop_at_recall:
                    print("Setting reference df.")
                    reference = result_df
                # with open(os.path.join(OUTPUT_DIR, "num_columns_exp.txt"), "a") as out_file:
                #     out_file.write(
                #         f"Num cols: {size}, method:{method} time for full cp: {end - start} seconds. {preprocessing_time} for setup."
                #         f" {TIME_SQL_QUERIES['x']} time for SQL.\n")


def single_metric_exp(df, exclude_list, is_numeric, trans_dict):
    metric_to_sort_by = {DF_MI_STRING: DF_MI_STRING, DF_ANOVA_F_STAT_STRING: DF_ANOVA_F_STAT_STRING,
                         DF_COSINE_SIMILARITY_STRING: DF_COSINE_SIMILARITY_STRING,
                         DF_COVERAGE_STRING: 'NUM_LARGE_GROUPS', DF_PVALUE_STRING: 'REGRESSION'}
    for metric in metric_to_sort_by:
        start = time.time()
        Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500 = setup_cherrypicking(
            df, exclude_list, is_numeric, trans_dict, start, do_regression=(metric == DF_PVALUE_STRING))

        ##################################
        print("Cherrypicking is starting.")
        cherry_picker = CherryPicker(
            exclude_list=exclude_list, numeric_list=is_numeric, grp_attr=GRP_ATTR, target_attr=TARGET_ATTR,
            compare_list=COMPARE_LIST, MI_dict=MI_dict, Anova_dict=Anova_dict,
            dataset_size=get_DB_size("my_table"), bucket_dict=bucket_objects, agg_type=AGG_TYPE,
            translation_dict=trans_dict, std_dict=std_dict,
            output_path=OUTPUT_PATH.replace("guided", "only_" + metric), start_time=start,
            stop_at_time=None, metric_subset=[metric])

        sorted_by = sort_attr_tuples(multi_atom_combinations, metric_to_sort_by[metric], Anova_dict, MI_dict,
                                     emb_sim_dict,
                                     std_dict,
                                     reg_weight_dict, num_groups_over100)
        mid = time.time()
        TIME_SQL_QUERIES["x"] = datetime.timedelta(seconds=0)
        res_df = cherry_picker.cherrypick_by_attributes(sorted_by, sample_size=None, add_metrics=True,
                                                        attr_tuple_to_num_preds={})
        end = time.time()
        with open(os.path.join(OUTPUT_DIR, "single_metric_exp.txt"), "a") as out_file:
            out_file.write(f"Metric: {metric}, time for full cp: {end - start} seconds. {mid - start} for setup."
                           f" {TIME_SQL_QUERIES['x']} time for SQL.\n")


def pred_level_cherrypicking(df, exclude_list, is_numeric, trans_dict):
    start = time.time()
    Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500 = setup_cherrypicking(
        df, exclude_list, is_numeric, trans_dict, start, do_regression=False)

    ##################################
    print("Cherrypicking is starting.")
    cherry_picker = CherryPicker(
        exclude_list=exclude_list, numeric_list=is_numeric, grp_attr=GRP_ATTR, target_attr=TARGET_ATTR,
        compare_list=COMPARE_LIST, MI_dict=MI_dict, Anova_dict=Anova_dict,
        dataset_size=get_DB_size("my_table"), bucket_dict=bucket_objects, agg_type=AGG_TYPE,
        translation_dict=trans_dict, std_dict=std_dict,
        output_path=OUTPUT_PATH.replace("guided", "pred_level"), start_time=start,
        stop_at_time=None)

    preds = generate_all_preds(df, exclude_list, bucket_objects)
    print(f"number of queries to run: {len(preds)}")
    cherry_picker.cherrypick_by_predicates(preds)
    print(f"Total time: {time.time() - start} seconds.")


def run_sample_guided_experiment(sample_sizes, iterations, df, exclude_list, is_numeric, trans_dict):
    for sample_size in sample_sizes:
        for i in range(iterations):
            start_time = time.time()
            print(f"Sample size: {sample_size} iteration: {i + 1}")
            sample_str = f'_{sample_size}sample'
            output_path = os.path.join(
                OUTPUT_DIR,
                "sampling",
                f"{DATABASE_NAME}_{AGG_TYPE}_{MAX_ATOMS}atoms_{QUERY_DESC}_{SORT_BY.replace(' ', '_')}{sample_str}_guided{i + 1}.csv")
            Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500 = setup_cherrypicking(
                df, exclude_list, is_numeric, trans_dict, start_time)
            print("Cherrypicking is starting.")
            if USE_SQL:
                cherry_picker = CherryPicker(exclude_list=exclude_list, numeric_list=is_numeric,
                                             grp_attr=GRP_ATTR, target_attr=TARGET_ATTR,
                                             compare_list=COMPARE_LIST, MI_dict=MI_dict, Anova_dict=Anova_dict,
                                             dataset_size=get_DB_size("my_table"), bucket_dict=bucket_objects,
                                             agg_type=AGG_TYPE,
                                             translation_dict=trans_dict,
                                             std_dict=std_dict,
                                             output_path=output_path, start_time=start_time, stop_at_time=STOP_AT_TIME,
                                             metric_subset=METRICS_SUBSET)
            sorted_by = sort_attr_tuples(multi_atom_combinations, SORT_BY, Anova_dict, MI_dict, emb_sim_dict, std_dict,
                                         reg_weight_dict, num_groups_over100)
            result_df = cherry_picker.sample_guided_cherrypicking_by_attribute(sample_size=sample_size,
                                                                               columns_list=sorted_by)


def sensitivity_to_k(k_values, df, exclude_list, is_numeric, trans_dict, reference_path=None):
    reference_df = None
    if reference_path is not None:
        reference_df = pd.read_csv(reference_path, index_col=0)
    for top_k in k_values:
        # for i in range(iterations):
        start_time = time.time()
        print(f"TOP_K: {top_k}")
        k_str = f'_K{top_k}'
        output_path = os.path.join(
            OUTPUT_DIR,
            "topk",
            f"{DATABASE_NAME}_{AGG_TYPE}_{MAX_ATOMS}atoms_{QUERY_DESC}_ALL_TOP_K_SERIAL{k_str}_guided.csv")
        Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict, multi_atom_combinations, bucket_objects, num_groups_over100, num_groups_over500 = setup_cherrypicking(
            df, exclude_list, is_numeric, trans_dict, start_time, do_regression=True)
        print("Cherrypicking is starting.")
        if USE_SQL:
            cherry_picker = CherryPicker(exclude_list=exclude_list, numeric_list=is_numeric,
                                         grp_attr=GRP_ATTR, target_attr=TARGET_ATTR,
                                         compare_list=COMPARE_LIST, MI_dict=MI_dict, Anova_dict=Anova_dict,
                                         dataset_size=get_DB_size("my_table"), bucket_dict=bucket_objects,
                                         agg_type=AGG_TYPE,
                                         translation_dict=trans_dict,
                                         std_dict=std_dict,
                                         output_path=output_path, start_time=start_time, stop_at_time=STOP_AT_TIME,
                                         metric_subset=METRICS_SUBSET, reference=reference_df, topk=top_k)
            result_df = combine_top_k_from_pre_known_metrics(
                cherry_picker, top_k, Anova_dict, MI_dict, emb_sim_dict, std_dict, reg_weight_dict,
                multi_atom_combinations,
                num_groups_over100, num_groups_over500)


def randomize_queries(number_of_queries, attributes, numeric_attrs, df, target_attr, agg_func=None):
    # choose a group attribute (numeric attrs are harder to support. so for now we don't choose them)
    attributes = [a for a in attributes if a not in numeric_attrs]
    df1 = df.dropna(subset=[target_attr], axis=0)
    selected_queries = []
    while len(selected_queries) < number_of_queries:
        grp_attr = random.choice(attributes)
        # choose two values
        possible_values = [x for x in df1[grp_attr].unique() if not utils.safe_is_nan(x)]
        if len(possible_values) < 2:
            continue
        grp1, grp2 = random.sample(possible_values, 2)
        # determine direction (or select it randomly?)
        if agg_func is None:
            agg_func = random.choice(['mean', 'median'])
        print(grp_attr, grp1, grp2, agg_func)
        gb = df1.groupby(grp_attr)[target_attr].agg(agg_func)
        if gb[grp1] > gb[grp2]:
            compare_list = [utils.less_than_cmp, grp1, grp2]
        else:
            compare_list = [utils.less_than_cmp, grp2, grp1]
        selected_queries.append((grp_attr, compare_list, agg_func))
    return selected_queries


def run_multiple_methods_for_query(target_attr, grp_attr, compare_list, agg_type,
                                   df, exclude_list, is_numeric, trans_dict, methods, stop_at_recall=False):
    query_desc = f"{grp_attr}_{compare_list[2]}_gt_{compare_list[1]}"
    reference = None
    for method in methods:
        print(f"running: method {method}")
        start = time.time()
        TIME_SQL_QUERIES["x"] = datetime.timedelta(seconds=0)
        method_sample = None
        sort_by = method
        if 'sample' in method:
            sort_by = 'original_order'
            method_sample = float(method.split(':')[0][:-6])  # e.g., '0.01sample:3' -> 0.01
        elif method.startswith('random_shuffle'):  # e.g., "random_shuffle:1"
            # Can't use pre-shuffled order - the attr combinations will be different for different queries.
            sort_by = 'random_shuffle'
        output_path = os.path.join(
            OUTPUT_DIR, "random_queries",
            f"{DATABASE_NAME}_{agg_type}_{MAX_ATOMS}atoms_{query_desc}_{method.replace(':', '')}_guided.csv")
        res_df, preprocessing_time = run_cherrypicking(df, exclude_list, is_numeric, trans_dict, METRICS_SUBSET,
                                                       output_path, "my_table", sort_by, method_sample, start,
                                                       target_attr, grp_attr, compare_list, agg_type,
                                                       shuffle_iter_index=None, reference=reference)
        if method == 'original_order' and stop_at_recall:
            print("Setting reference df.")
            reference = res_df
