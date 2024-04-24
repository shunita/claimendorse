import time

import dotenv
import sqlalchemy
import datetime
import pandas as pd
import numpy as np
import scipy as sp

from config import *
from utils import calc_t_stat, calc_mean_diff_degrees_freedom, calc_chi_squared_stat, get_attr_and_value_fields

TIME_SQL_QUERIES = {"x": datetime.timedelta(seconds=0)}


def connect_sql_db():
    dotenv.load_dotenv(dotenv_path="data/database_connection.env")
    USERNAME = os.getenv("CONNECTION_USERNAME")
    PASSWORD = os.getenv("CONNECTION_PASSWORD")
    SERVER = os.getenv("SERVER_IP")
    # DATABASE_NAME = os.getenv("DATABASE_NAME")
    url = sqlalchemy.engine.URL.create(
        drivername="postgresql",
        username=USERNAME,
        host="localhost",
        port=5432,
        password=PASSWORD,
        database=DATABASE_NAME
    )
    engine = sqlalchemy.create_engine(url)
    return engine


class SQLEngineSingleton(object):
    def __new__(cls):
        if not hasattr(cls, 'engine'):
            cls.engine = connect_sql_db()
        return cls.engine


class QueryRunnerBase(object):
    def __init__(self, start_time, query_path, result_columns):
        self.start_time = start_time
        self.engine = SQLEngineSingleton()
        self.query_path = query_path
        self.result_columns = result_columns

    def add_query_args(self, query_args_dict, bucket_dict):
        selecting_strings = []
        grouping_strings = []
        for col in query_args_dict["column_tuple"]:
            if col not in bucket_dict:
                selecting_strings.append(f'"{col}"')
                grouping_strings.append(f'"{col}"')
            else:
                bucket = bucket_dict[col]
                selecting_strings.append(
                    f'width_bucket("{col}", {bucket.low} ,{bucket.high}, {bucket.count}) as bucket_{col}')
                grouping_strings.append(f'bucket_{col}')
        query_args_dict["selecting_string"] = ", ".join(selecting_strings)
        query_args_dict["grouping_string"] = ", ".join(grouping_strings)
        return query_args_dict

    def prepare_query(self, query_args_dict, bucket_dict):
        sql_string = open(self.query_path, "r").read()
        query_args_dict = self.add_query_args(query_args_dict, bucket_dict)
        final_sql_string = sql_string.format(**query_args_dict)
        query = sqlalchemy.text(final_sql_string)
        return query

    def run_query(self, query_args_dict, bucket_dict):
        query = self.prepare_query(query_args_dict, bucket_dict)
        time_before = datetime.datetime.now()
        query_result = self.engine.execute(query)
        time_after = datetime.datetime.now()
        TIME_SQL_QUERIES["x"] += (time_after - time_before)
        #with open(os.path.join(OUTPUT_DIR, 'log_mean_case_when_query.csv'), "a") as log_file:
        #    log_file.write(f'{query_args_dict["column_tuple"]},{time_after - time_before}\n')
        return self.post_process(query_result, query_args_dict['column_tuple'], bucket_dict)

    def specific_post_process(self, query_res_df):
        return query_res_df

    def post_process(self, query_result, column_names, bucket_dict):
        val_columns = []
        for i in range(len(column_names)):
            val_columns.append(f'Value{i+1}')
        df = pd.DataFrame(query_result, columns=val_columns+self.result_columns)
        for i, cn in enumerate(column_names):
            df[f"Attr{i+1}"] = cn
            if cn in bucket_dict:
                bucket = bucket_dict[cn]
                try:
                    # replace bucket ids (e.g., 3) with range strings (e.g., "20-30")
                    df[f"Value{i+1}"] = df[f"Value{i+1}"].apply(bucket.bucket_to_range_string)
                except:
                    print(f"Could not map bucket ids to range string in {column_names}. Possible None or nan?")
        if REMOVE_NULL_PREDICATES:
            for i in range(len(column_names)):
                # keep rows where the attribute is null (where there are less than max atoms).
                # But if the attribute is full, then the value must not be null.
                df = df[(df[f'Attr{i+1}'].isna()) | (~df[f'Value{i+1}'].isna())]
        df = self.specific_post_process(df)
        end_time = time.time()
        df["Time"] = end_time - self.start_time
        output_column_order = get_attr_and_value_fields(len(column_names))
        output_column_order += [c for c in df.columns if not c.startswith('Attr') and not c.startswith('Value')]
        return df[output_column_order]


class QueryRunnerPredLevelBase(QueryRunnerBase):
    def __init__(self, start_time, query_path, result_columns):
        super(QueryRunnerPredLevelBase, self).__init__(
            start_time, query_path, result_columns)

    def add_query_args(self, query_args_dict, bucket_dict):
        # TODO: this works for single atom only
        col, val = query_args_dict['pred_column'], query_args_dict['pred_value']
        if col in bucket_dict:
            bucket = bucket_dict[col]
            if type(val) == str:
                val = val.replace("'", "''")
            pred_string = f'width_bucket("{col}", {bucket.low} ,{bucket.high}, {bucket.count})='+f"'{val}'"
        else:
            pred_string = f'"{col}"=' + f"'{val}'"
        query_args_dict["pred_string"] = pred_string
        return query_args_dict

    def run_query(self, query_args_dict, bucket_dict):
        query = self.prepare_query(query_args_dict, bucket_dict)
        time_before = datetime.datetime.now()
        query_result = self.engine.execute(query)
        time_after = datetime.datetime.now()
        TIME_SQL_QUERIES["x"] += (time_after - time_before)
        return self.post_process(query_result, query_args_dict['pred_column'], query_args_dict['pred_value'], bucket_dict)

    def post_process(self, query_result, pred_column, pred_value, bucket_dict):
        if MAX_ATOMS > 1:
            print("Warning: pred level query does not support multiple atoms!")
        # here the value needs to be added
        df = pd.DataFrame(query_result, columns=self.result_columns)
        df["Attr1"] = pred_column
        df["Value1"] = pred_value
        if pred_column in bucket_dict:
            bucket = bucket_dict[pred_column]
            try:
                # replace bucket ids (e.g., 3) with range strings (e.g., "20-30")
                df["Value1"] = df["Value1"].apply(bucket.bucket_to_range_string)
            except:
                print(f"Could not map bucket ids to range string in {cn}. Possible None or nan?")
        df = self.specific_post_process(df)
        end_time = time.time()
        df["Time"] = end_time - self.start_time
        output_column_order = get_attr_and_value_fields(1)
        output_column_order += [c for c in df.columns if not c.startswith('Attr') and not c.startswith('Value')]
        return df[output_column_order]


class QueryRunnerMeanDiffPredLevel(QueryRunnerPredLevelBase):
    def __init__(self, start_time, compute_stat_sig=True):
        if not compute_stat_sig:
            print("Warning: not computing statistical significance for predicate level mean diff "
                  "query is not supported. The stat sig will be computed anyway.")
        self.compute_stat_sig = compute_stat_sig
        qpath = "sql_queries/mean_query_single_pred_case_when.sql"
        result_cols = ['mean1', 'N1', 's1', 'mean2', 'N2', 's2']
        super(QueryRunnerMeanDiffPredLevel, self).__init__(
            start_time,
            query_path=qpath,
            result_columns=result_cols)

    def specific_post_process(self, query_res_df):
        if self.compute_stat_sig:
            query_res_df['t_stat'] = query_res_df.apply(
                lambda row: calc_t_stat(row['mean1'], row['N1'], row['s1'], row['mean2'], row['N2'], row['s2']), axis=1)
            query_res_df['deg_freedom'] = query_res_df.apply(
                lambda row: calc_mean_diff_degrees_freedom(row['N1'], row['s1'], row['N2'], row['s2']), axis=1)
            query_res_df['pvalue'] = query_res_df.apply(lambda row: sp.stats.t.sf(np.abs(row['t_stat']), row['deg_freedom']) * 2,
                                                        axis=1)  # We are doing a two sided test
            query_res_df = query_res_df.rename({'t_stat': 'statistical_significance_stat'}, axis=1)
            query_res_df = query_res_df.drop(['s1', 's2', 'deg_freedom'], axis='columns')
        return query_res_df


class QueryRunnerMedianDiffPredLevel(QueryRunnerPredLevelBase):  # maybe inherit from mean diff pred level
    def __init__(self, start_time, compute_stat_sig=True):
        if not compute_stat_sig:
            print("Warning: not computing statistical significance for predicate level "
                  "median diff query is not supported. The stat sig will be computed anyway.")
        self.compute_stat_sig = compute_stat_sig
        qpath = "sql_queries/median_diff_query_single_pred.sql"
        result_cols = ["v1_over", "v2_over", "v1_under", "v2_under", "total_median", "median1", "N1", "median2", "N2"]
        super(QueryRunnerMedianDiffPredLevel, self).__init__(
            start_time,
            query_path=qpath,
            result_columns=result_cols)

    def specific_post_process(self, df):
        if self.compute_stat_sig:
            df['statistical_significance_stat'] = df.apply(
                lambda row: calc_chi_squared_stat(row['v1_over'], row['v2_over'], row['v1_under'], row['v2_under']), axis=1)
            df['pvalue'] = df['statistical_significance_stat'].apply(lambda stat: 1 - sp.stats.chi2.cdf(stat, 1))
            df = df.rename({'chi_squared_stat': 'statistical_significance_stat'}, axis=1)
            df = df.drop(['v1_over', 'v1_under', 'v2_over', 'v2_under'], axis='columns')
        return df



class QueryRunnerMeanDiff(QueryRunnerBase):
    def __init__(self, start_time, compute_stat_sig=True):
        self.compute_stat_sig = compute_stat_sig
        if compute_stat_sig:
            qpath = "sql_queries/mean_query_case_when.sql"
            # qpath = "sql_queries/mean_diff_query_simplified.sql"
            result_cols = ['mean1', 'N1', 's1', 'mean2', 'N2', 's2']
        else:
            qpath = "sql_queries/mean_query_case_when_only_count.sql"
            result_cols = ['mean1', 'N1', 'mean2', 'N2']
        super(QueryRunnerMeanDiff, self).__init__(
            start_time,
            query_path=qpath,
            result_columns=result_cols)

    def specific_post_process(self, query_res_df):
        if self.compute_stat_sig:
            query_res_df['t_stat'] = query_res_df.apply(
                lambda row: calc_t_stat(row['mean1'], row['N1'], row['s1'], row['mean2'], row['N2'], row['s2']), axis=1)
            query_res_df['deg_freedom'] = query_res_df.apply(
                lambda row: calc_mean_diff_degrees_freedom(row['N1'], row['s1'], row['N2'], row['s2']), axis=1)
            query_res_df['pvalue'] = query_res_df.apply(lambda row: sp.stats.t.sf(np.abs(row['t_stat']), row['deg_freedom']) * 2,
                                                        axis=1)  # We are doing a two sided test
            query_res_df = query_res_df.rename({'t_stat': 'statistical_significance_stat'}, axis=1)
            query_res_df = query_res_df.drop(['s1', 's2', 'deg_freedom'], axis='columns')
        return query_res_df


class QueryRunnerMedianDiff(QueryRunnerBase):
    def __init__(self, start_time, compute_stat_sig=True):
        self.compute_stat_sig = compute_stat_sig
        if compute_stat_sig:
            qpath = "sql_queries/median_diff_query_unified.sql"
            # qpath = "sql_queries/median_diff_query_join.sql"
            result_cols = ["v1_over", "v2_over", "v1_under", "v2_under", "total_median", "median1", "N1", "median2", "N2"]
        else:
            qpath = "sql_queries/median_diff_query_no_stat_sig.sql"
            result_cols = ["median1", "N1", "median2", "N2"]
        super(QueryRunnerMedianDiff, self).__init__(
            start_time,
            query_path=qpath,
            result_columns=result_cols)

    def specific_post_process(self, df):
        if self.compute_stat_sig:
            df['statistical_significance_stat'] = df.apply(
                lambda row: calc_chi_squared_stat(row['v1_over'], row['v2_over'], row['v1_under'], row['v2_under']), axis=1)
            df['pvalue'] = df['statistical_significance_stat'].apply(lambda stat: 1 - sp.stats.chi2.cdf(stat, 1))
            df = df.rename({'chi_squared_stat': 'statistical_significance_stat'}, axis=1)
            df = df.drop(['v1_over', 'v1_under', 'v2_over', 'v2_under'], axis='columns')
        return df


class QueryRunnerCountDiff(QueryRunnerBase):
    def __init__(self, start_time):
        super(QueryRunnerCountDiff, self).__init__(
            start_time,
            query_path="sql_queries/count_diff_query.sql",
            result_columns=['N1', 'N2'])


class QueryRunnerCountSTD(QueryRunnerBase):
    def __init__(self, start_time):
        super(QueryRunnerCountSTD, self).__init__(
            start_time,
            query_path="sql_queries/count_std_query.sql",
            result_columns=[])

    def post_process(self, query_result, column_names, bucket_dict):
        return list(query_result)[0][0]
