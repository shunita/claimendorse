WITH discretized AS (
    SELECT "{grp_attr}",
           "{target_attr}"
    FROM {table_name}
    WHERE {pred_string} AND "{grp_attr}" IN ('{value1}', '{value2}')
    ),
filtered AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{target_attr}") AS total_median,
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value1}' THEN "{target_attr}" ELSE NULL END) AS median1,
           COUNT(CASE WHEN "{grp_attr}"='{value1}' THEN "{target_attr}" ELSE NULL END) as N1,
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END) AS median2,
           COUNT(CASE WHEN "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END) as N2
    FROM discretized
    HAVING PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value1}' THEN "{target_attr}" ELSE NULL END) <
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END)
),
inter_table AS (SELECT "{target_attr}","{grp_attr}",total_median
                FROM discretized, filtered
),
observed AS (
    SELECT COUNT(CASE WHEN "{grp_attr}"='{value1}' AND "{target_attr}">total_median THEN "{target_attr}"
                      ELSE NULL END) AS v1_over,
           COUNT(CASE WHEN "{grp_attr}"='{value2}' AND "{target_attr}">total_median THEN "{target_attr}"
                      ELSE NULL END) AS v2_over,
           COUNT(CASE WHEN "{grp_attr}"='{value1}' AND "{target_attr}"<=total_median THEN "{target_attr}"
                      ELSE NULL END) AS v1_under,
           COUNT(CASE WHEN "{grp_attr}"='{value2}' AND "{target_attr}"<=total_median THEN "{target_attr}"
                      ELSE NULL END) AS v2_under
    FROM inter_table
)
SELECT v1_over, v2_over, v1_under, v2_under, total_median, median1, N1, median2, N2
FROM observed NATURAL INNER JOIN filtered
WHERE v1_over+v1_under>={min_group_size} AND v2_over+v2_under>={min_group_size};