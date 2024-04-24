import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
from constants import *
from utils import find_index_in_list, make_translation_for_ACS
import matplotlib.colors as mcolors

# COLORS = ["b", "r", "k", "g", "c", "m"]  # assume up to 6 series on one chart
COLORS = [mcolors.CSS4_COLORS[x] for x in ('blue', 'red', 'green','orange','magenta', 'blueviolet')]
MARKERS = ["o", "x", "s", "v", "^", "*"]
LINESTYLES = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]


def get_top_k_of_multiple_metrics(df, metrics, k):
    begin = time.time()
    df = df.reset_index(drop=True)
    index_set = set()
    for metric, sort_ascending in metrics:
        index_set.add(df.sort_values(by=metric, ascending=sort_ascending).head(k).index)
    res = df.loc[index_set]
    end = time.time()
    print(f"getting top k from result df for all metrics: {end-begin}")
    return res


def calculate_metrics_for_sample_result(sample_result_path, full_result_path):
    # We expect columns of "Attr" and "value" in both
    sample_df = pd.read_csv(sample_result_path, index_col=0)
    full_res_df = pd.read_csv(full_result_path, index_col=0)
    # filtering by pvalue
    sample_df = sample_df[sample_df["pvalue"] < 1][["Attr", "value"]]
    full_res_df = full_res_df[full_res_df["pvalue"] < 1][["Attr", "value"]]
    sample_set = set(sample_df.itertuples(index=False, name=None))
    full_res_set = set(full_res_df.itertuples(index=False, name=None))
    false_positives = len(sample_set.difference(full_res_set))
    precision = len(sample_set.intersection(full_res_set)) / len(sample_set)
    recall = len(sample_set.intersection(full_res_set)) / len(full_res_set)
    return precision, recall, false_positives



def calculate_metrics_by_time_top_k_single_score(result_path_list, labels_list, full_result_path, score_name, sorting_ascending=True,
                                                 k=10, time_bin_size_in_seconds=10, max_time_limit_in_seconds=180, output_prefix=""):
    colors = ["b", "r", "k", "g", "c", "m"]  # assume up to 6 results
    markers = ["o", "x", "s", "v", "^", "*"]

    #default sorting order is for anova_pvalue
    full_res_df = pd.read_csv(full_result_path, index_col=0)
    full_res_df = full_res_df.sort_values(by=score_name, ascending=sorting_ascending)
    #print(full_res_df.columns)
    if "pvalue" in full_res_df.columns:
        full_res_df = full_res_df[full_res_df["pvalue"] < 1]
    else:
        print("Could not filter full_res_df by pvalue - no such column.")
    top_k_total_score = full_res_df[score_name].head(k).sum()
    top_k_df = full_res_df[["Attr", "Value", score_name]].head(k)
    #print(top_k_df)
    top_k_set = set(top_k_df[["Attr", "Value"]].itertuples(index=False, name=None))

    for i,result_path in enumerate(result_path_list):
        result_df = pd.read_csv(result_path, index_col=0)
        if "pvalue" in result_df.columns:
            result_df = result_df[result_df["pvalue"] < 1]
        else:
            print("Could not filter result_df by pvalue - no such column.")

        recalls = []
        precisions = []
        score_recalls = []
        first_out_over_last_ins = []
        time_bins = []
        for time_limit in range(0, max_time_limit_in_seconds, time_bin_size_in_seconds):
            #print(f"time limit:{time_limit}")
            result_subset = result_df[result_df["Time"] <= time_limit].sort_values(by=score_name, ascending=sorting_ascending).head(k)
            result_top_k_total_score = result_subset[score_name].sum()
            if len(result_subset) == 0:
                precision = 0
                recall = 0
                score_recall = 0
                first_out_over_last_in = 0  # 1?
            else:
                result_subset_set = set(result_subset[["Attr", "Value"]].itertuples(index=False, name=None))
                last_in = result_subset[score_name].iloc[len(result_subset) - 1]
                precision = len(result_subset_set.intersection(top_k_set)) / len(result_subset_set)
                recall = len(result_subset_set.intersection(top_k_set)) / len(top_k_set)
                score_recall = result_top_k_total_score / top_k_total_score
                first_out = None
                for a, v, score in full_res_df[['Attr', 'Value', score_name]].itertuples(index=False, name=None):
                    if len(result_subset[(result_subset['Attr'] == a) & (result_subset['Value'] == v)]) == 0:
                        first_out = score
                        break
                first_out_over_last_in = 0
                if last_in > 0:
                    first_out_over_last_in = first_out / last_in

            precisions.append(precision)
            recalls.append(recall)
            score_recalls.append(score_recall)
            first_out_over_last_ins.append(first_out_over_last_in)
            # if recall > 0.9:
            #     print(f"recall: {recall}, time<{time_limit} seconds")
            time_bins.append(time_limit)
        plt.figure(1)   #figure for recalls
        plt.scatter(time_bins, recalls, c=colors[i], marker=markers[i], label=labels_list[i], alpha=0.5)
        plt.figure(2)   #figure for precision
        plt.scatter(time_bins, precisions, c=colors[i], marker=markers[i], label=labels_list[i], alpha=0.5)
        plt.figure(3)   # figure for recall as a fraction of the optimal recall score
        plt.scatter(time_bins, score_recalls, c=colors[i], marker=markers[i], label=labels_list[i], alpha=0.5)
        plt.figure(4)  # figure for first non retrieved over last retrieved
        plt.scatter(time_bins, first_out_over_last_ins, c=colors[i], marker=markers[i], label=labels_list[i], alpha=0.5)

    plt.figure(1)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_top{k}recall_over_time.png"))
    plt.figure(2)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_top{k}precision_over_time.png"))
    plt.figure(3)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_top{k}score_recall_over_time.png"))
    plt.figure(4)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_top{k}first_out_over_last_in_over_time.png"))


def calculate_metrics_by_time_top_k_all_scores(result_path, full_result_path, k=10, time_bin_size_in_seconds=100,
                                               max_time_limit_in_seconds=None, output_prefix="",
                                               filter_by_reference=False, output_format='figures',
                                               score_recall_threshold=None, score_name_subset=None):
    """output_format: 'figures' or 'dict'"""
    # colors = ["b", "r", "k", "g", "c", "m"]  # assume up to 6 series on one chart
    # markers = ["o", "x", "s", "v", "^", "*"]
    score_names = [DF_COSINE_SIMILARITY_STRING,
                   DF_INVERTED_PVALUE_STRING,
                   DF_NORMALIZED_ANOVA_F_STAT_STRING,
                   DF_NORMALIZED_MI_STRING,
                   DF_COVERAGE_STRING,
                   DF_METRICS_AVERAGE,
                   ]
    score_name_to_stats = {DF_COSINE_SIMILARITY_STRING: {},
                           DF_INVERTED_PVALUE_STRING: {},
                           DF_NORMALIZED_ANOVA_F_STAT_STRING: {},
                           DF_NORMALIZED_MI_STRING: {},
                           DF_COVERAGE_STRING: {},
                           DF_METRICS_AVERAGE: {}
                           }
    attr_value_columns = []
    for i in range(MAX_ATOMS):
        attr_value_columns += [f'Attr{i+1}', f'Value{i+1}']
    full_res_df = pd.read_csv(full_result_path, index_col=0)
    full_res_df = full_res_df.fillna({a: "NA" for a in attr_value_columns})
    result_df = pd.read_csv(result_path, index_col=0)
    result_df = result_df.fillna({a: "NA" for a in attr_value_columns})
    if score_name_subset is not None:
        scores = score_name_subset
    else:
        scores = score_names
    for i, score_name in enumerate(scores):
        if score_name not in full_res_df.columns:
            print(f"{score_name} is not in the columns of {full_result_path}")
            continue
        # the normalized scores all behave s.t. larger is better
        full_res_df = full_res_df.sort_values(by=score_name, ascending=False)
        top_k_total_score = full_res_df[score_name].head(k).sum()
        print(f"\n{score_name}, top k total score: {top_k_total_score}")
        top_k_df = full_res_df[[*attr_value_columns, score_name]].head(k)
        top_k_set = set(top_k_df[attr_value_columns].itertuples(index=False, name=None))

        if filter_by_reference:
            result_df = result_df[result_df[attr_value_columns].apply(
                lambda row: is_predicate_in_df(row.values, full_res_df), axis=1)]
            # Take the scores that were calculated on the full df
            if score_name in result_df:
                result_df = result_df.drop(score_name, axis=1)
            result_df = result_df.merge(full_res_df[[*attr_value_columns, score_name]], on=attr_value_columns)
        recalls = []
        # precisions = []
        score_recalls = []
        # first_out_over_last_ins = []
        time_bins = []
        max_time = max_time_limit_in_seconds
        if max_time_limit_in_seconds is None:
            max_time = int(np.round(result_df["Time"].max(), -1)+10)
        #else:
        #    max_time = min(max_time_limit_in_seconds, int(np.round(result_df["Time"].max(), -1)+10))
        print(f"max time in figure: {max_time}")
        time_to_score_recall_threshold = max_time
        for time_limit in range(0, max_time, time_bin_size_in_seconds):
            #print(f"time limit:{time_limit}")
            result_subset = result_df[result_df["Time"] <= time_limit].sort_values(by=score_name, ascending=False).head(k)
            result_top_k_total_score = result_subset[score_name].sum()
            if len(result_subset) == 0:
                # precision = 0
                recall = 0
                score_recall = 0
                # first_out_over_last_in = 0  # 1?
            else:
                result_subset_set = set(result_subset[attr_value_columns].itertuples(index=False, name=None))
                # last_in = result_subset[score_name].iloc[len(result_subset) - 1]
                # precision = len(result_subset_set.intersection(top_k_set)) / len(result_subset_set)
                recall = len(result_subset_set.intersection(top_k_set)) / len(top_k_set)
                if top_k_total_score > 0:
                    score_recall = result_top_k_total_score / top_k_total_score
                else:
                    score_recall = 0
                # first_out = None
                # for predicate_and_score in full_res_df[[*attr_value_columns, score_name]].itertuples(index=False, name=None):
                #     if not is_predicate_in_df(predicate_and_score[:-1], result_subset):
                        # first_out = predicate_and_score[-1]  # score of the first non-retrieved predicate
                        # break
                # first_out_over_last_in = 0
                # if last_in > 0:
                #     first_out_over_last_in = first_out / last_in
            # precisions.append(precision)
            recalls.append(recall)
            score_recalls.append(score_recall)
            # first_out_over_last_ins.append(first_out_over_last_in)
            time_bins.append(time_limit)
            if score_recall_threshold is not None and score_recall > score_recall_threshold and \
                    f'time_to_score_recall{score_recall_threshold}' not in score_name_to_stats[score_name]:
                #score_name_to_stats[score_name][f'time_to_score_recall{score_recall_threshold}'] = time_limit
                time_to_score_recall_threshold = time_limit
                break
        score_name_to_stats[score_name] = {'recalls': recalls, 'score_recalls': score_recalls,
                                           # 'foli': first_out_over_last_ins,
                                           'time_bins': time_bins}
        if score_recall_threshold is not None:
            score_name_to_stats[score_name][f'time_to_score_recall{score_recall_threshold}'] = time_to_score_recall_threshold
        if output_format == 'figures':
            for j, y_values in enumerate([recalls, score_recalls,
                                          # first_out_over_last_ins
                                          ]):
            #for j, y_values in enumerate([recalls, precisions score_recalls, first_out_over_last_ins]):
                plt.figure(j)   #figure for recalls
                # plt.scatter(time_bins, y_values, c=COLORS[i], marker=MARKERS[i], label=score_name, alpha=0.5)
                plt.plot(time_bins, y_values, c=COLORS[i], linestyle=LINESTYLES[i], label=score_name)
    # for j, fig_name in enumerate(['recall', 'precision', 'score_recall', 'first_out_over_last_in']):
    if output_format == 'figures':
        for j, fig_name in enumerate(['recall', 'score_recall',
                                      # 'first_out_over_last_in'
                                      ]):
            plt.figure(j)
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_top{k}{fig_name}_over_time.png"))
    elif output_format == 'dict':
        return score_name_to_stats
    print(f"Unsupported output format: {output_format}. Should be either 'figures' or 'dict'.")


def combine_result_files(res_file_dict, metric_name, reference_file, order,
                         k=100, time_bin_size_in_seconds=100,
                         max_time_limit_in_seconds=700, output_prefix="",
                         filter_by_reference=False, time_unit='seconds', prev_stat_dict=None, base_font_size=26
                         ):
    """
    :param res_file_dict: {desc1: [res_file1, res_file2], desc2: [res_file3]}
    :param metric_names: str, should be one of the metrics in the dictionary
    :param order: list of dict keys to define the order of series in the figure
    :return: create a figure
    """
    # {desc->[{metric->{'recalls': .., 'score_recalls': .., 'foli':.. 'time bins':..}}]}
    print(f"order: {order}")
    print(f"res_file_dict keys: {res_file_dict.keys()}")
    if prev_stat_dict is not None:
        stat_dict = prev_stat_dict
    else:
        stat_dict = {}
    for desc in res_file_dict:
        print(desc)
        if desc in stat_dict:
            print("skipping")
            continue
        stat_dict[desc] = []
        for res_file in res_file_dict[desc]:
            stat_dict[desc].append(calculate_metrics_by_time_top_k_all_scores(res_file, reference_file, k,
                                                                              time_bin_size_in_seconds,
                                                                              max_time_limit_in_seconds,
                                                                              filter_by_reference,
                                                                              output_format='dict'))
    print(stat_dict.keys())
    # stat_dict = {desc->[{metric->{'recalls': .., 'score_recalls': .., 'foli':.. 'time bins':..}}]}
    plt.figure(figsize=(6,6))
    for series_index, desc in enumerate(order):
        xs = stat_dict[desc][0][metric_name]['time_bins']
        num_files_to_include = len(stat_dict[desc])
        if num_files_to_include > 1:
            # average the results
            # ys = np.average([stat_dict[desc][i][metric_name]['score_recalls'] for i in range(num_files_to_include)], axis=0)
            all_ys = [stat_dict[desc][i][metric_name]['score_recalls'] for i in range(num_files_to_include)]
            print(f"{[len(ys) for ys in all_ys]}")
            ys = np.average(all_ys, axis=0)
        else:
            ys = stat_dict[desc][0][metric_name]['score_recalls']
        xs_to_plot, ys_to_plot = xs, ys
        # find the first non zero entry
        # first_non_zero_index = next((i for i, y in enumerate(ys) if y > 0), None)
        first_non_zero_index = None
        if first_non_zero_index is not None:
            xs_to_plot, ys_to_plot = xs[first_non_zero_index:], ys[first_non_zero_index:]
        if desc == 'Regression':
           desc = 'Measure-specific'
        plt.plot(xs_to_plot, ys_to_plot, c=COLORS[series_index], #marker=MARKERS[series_index],
                 linestyle=LINESTYLES[series_index], linewidth=6, label=desc, #alpha=0.5
                 )
    if metric_name == 'Inverted pvalue':
       plt.legend(fontsize=f"{base_font_size-2}", borderpad=0.2, loc='center right')
    if time_unit == 'minutes':
        plt.xticks(ticks=xs[::4], labels=[int(x/60) for x in xs[::4]], fontsize=base_font_size)
    else:
        plt.xticks(ticks=xs[::4], fontsize=base_font_size)
    plt.yticks(fontsize=base_font_size)
    plt.xlabel(f'Time ({time_unit})', fontsize=base_font_size)  # fontsize?
    ylabel = metric_name
    if metric_name == DF_NORMALIZED_ANOVA_F_STAT_STRING:
        ylabel = 'Normalized ANOVA'
    if metric_name == 'Metrics Average':
        plt.ylabel(f'{ylabel} score recall', fontsize=base_font_size)
        plt.ylabel("Score recall", fontsize=base_font_size)
    plt.rc('xtick', labelsize=base_font_size) 
    plt.rc('ytick', labelsize=base_font_size) 
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_top{k}{'score_recall'}_over_time_multi.pdf"), format="pdf")
    plt.clf()
    return stat_dict


def find_time_until_specific_score_recall(result_path, full_result_path, threshold=0.95, score_name_subset=None, k=100):
    score_name_to_stats = calculate_metrics_by_time_top_k_all_scores(
        result_path, full_result_path, k, time_bin_size_in_seconds=1,
        max_time_limit_in_seconds=None, output_prefix="",
        filter_by_reference=False, output_format='dict', score_recall_threshold=threshold,
        score_name_subset=score_name_subset)
    # metric_to_time_until_recall = {metric: score_name_to_stats[metric][f'time_to_score_recall{threshold}']
    #                                for metric in score_name_to_stats}
    metric_to_time_until_recall = {}
    if score_name_subset is not None:
        scores = score_name_subset
    else:
        scores = score_name_to_stats.keys()
    for metric in scores:
        if f'time_to_score_recall{threshold}' in score_name_to_stats[metric]:
            metric_to_time_until_recall[metric] = score_name_to_stats[metric][f'time_to_score_recall{threshold}']
        else:
            times = score_name_to_stats[metric]['time_bins']
            recalls = score_name_to_stats[metric]['score_recalls']
            for i in range(len(times)):
                if recalls[i] > threshold:
                    metric_to_time_until_recall[metric] = times[i]
                    print(f"{metric}, time until score recall>{threshold}: {times[i]}")
                    break
    return metric_to_time_until_recall




def is_predicate_in_df(attr_value_list, df):
    subset = df.copy()
    for i in range(0, len(attr_value_list), 2):
        a = attr_value_list[i]
        v = attr_value_list[i+1]
        # print(f'Attr{int((i/2)+1)}, Value{(int(i/2)+1)}')
        subset = subset[(subset[f'Attr{int((i/2)+1)}'] == a) & (subset[f'Value{int((i/2)+1)}'] == v)]
    if len(subset) > 0:
        return True
    return False


def analyze_regression(reg1, reg2, attrs, encoder_dict, df, grp_attr, target_attr):
    attr_to_score1 = dict(zip(attrs, reg1.coef_))
    attr_to_score2 = dict(zip(attrs, reg2.coef_))
    attrs_and_score_diff = sorted(list(zip(attrs, reg2.coef_ - reg1.coef_)), key=lambda x: x[1], reverse=True)
    for a, score_diff in attrs_and_score_diff[:10]:
        print(f"Attribute: {a}, score for reg2: {attr_to_score2[a]}, score for reg1: {attr_to_score1[a]}")
        if a in encoder_dict:
            print(f"Values in ascending order: {encoder_dict[a].keys()}")
        print(df.groupby(by=[a, grp_attr]).agg({target_attr: ['mean', 'count']}))


def sort_found_predicates_by_original_column_order(output_path, data_path):
    cols = list(pd.read_csv(data_path, index_col=0, nrows=1).columns)
    preds_df = pd.read_csv(output_path, index_col=0)
    preds_df['Attr1_loc'] = preds_df['Attr1'].apply(lambda x: find_index_in_list(x, cols))
    preds_df['Attr2_loc'] = preds_df['Attr2'].apply(lambda x: find_index_in_list(x, cols))
    preds_df.sort_values(by=['Attr2_loc', 'Value2'], inplace=True, kind='stable')
    preds_df.sort_values(by=['Attr1_loc', 'Value1'], inplace=True, kind='stable')
    return preds_df


def analyze_top_k_ACS(predicates_path, output_path):
    data_path = "data/Folkstable/SevenStates/Seven_States_grouped.csv"
    cols = list(pd.read_csv(data_path, index_col=0, nrows=1).columns)
    trans_dict = make_translation_for_ACS(cols)
    allocation_flags = [col for col in cols if "allocation flag" in trans_dict[col]]

    preds_df = sort_found_predicates_by_original_column_order(predicates_path, data_path)
    print(f"Read {len(preds_df)} predicates.")
    # Keep only large groups
    preds_df = preds_df[(preds_df['N1'] >= 30) & (preds_df['N2'] >= 30)]
    # blacklist specific predicate (RELP=16)
    preds_df = preds_df[~(((preds_df['Attr1'] == 'RELP') & (preds_df['Value1'] == '16')) | ((preds_df['Attr2'] == 'RELP') & (preds_df['Value2'] == '16')))]
    # blacklist allocation flags - they are not interesting (we should not run on them either)
    preds_df = preds_df[(~preds_df['Attr1'].isin(allocation_flags)) & (~preds_df['Attr2'].isin(allocation_flags))]
    # remove predicates where the value is nan - it's hard to phrase them into claims.
    # Attr1 is never NA, Attr2 may be NA. Remove only rows where Attr2 is not NA but Value2 is NA.
    preds_df = preds_df[~preds_df['Value1'].isna()]
    preds_df = preds_df[(preds_df['Attr2'].isna()) | (~preds_df['Value2'].isna())]
    print(f"After filtering, {len(preds_df)} predicates remain.")
    # preds_df = preds_df[preds_df['pvalue'] < 0.05]
    # print(f"When keeping only statistical significant groups: {len(preds_df)}")

    c = preds_df.head(20).copy()
    c['order'] = 'original'
    c.to_csv(output_path)
    metrics = ['Normalized_MI', 'Anova_F_Stat', 'Inverted pvalue', 'Cosine Similarity', 'Coverage', 'Metrics Average']
    for metric in metrics:
        c = preds_df.sort_values(metric, ascending=False, inplace=False).head(20).copy()
        c['order'] = metric
        c.to_csv(output_path, mode='a', header=False)
