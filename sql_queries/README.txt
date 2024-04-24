MEDIAN

median_diff_query_unified.sql - case when based query for median difference with vi_over/under counts.

median_diff_query_join.sql - join based query for median difference, returns the vi_over/under counts, and the medians in the two groups.

median_diff_query_no_stat_sig.sql - case when based, only returns the medians of the two groups. (used in single metric ablation)

MEAN

mean_query_case_when.sql - case when based, returns the information for calculating the statistic in python.

mean_diff_query_simplified.sql - same but join based.

mean_query_single_pred_case_when.sql - used for pred level ablation.