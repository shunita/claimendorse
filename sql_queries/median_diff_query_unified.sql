WITH discretized AS (
    SELECT {selecting_string},
           "{grp_attr}",
           "{target_attr}"
    FROM {table_name}
    WHERE {where_string}
    ),
filtered AS (
    SELECT {grouping_string},
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value1}' OR "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END) AS total_median,
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value1}' THEN "{target_attr}" ELSE NULL END) AS median1,
           COUNT(CASE WHEN "{grp_attr}"='{value1}' THEN "{target_attr}" ELSE NULL END) as N1,
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END) AS median2,
           COUNT(CASE WHEN "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END) as N2
    FROM discretized
    GROUP BY {grouping_string}
    HAVING PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value1}' THEN "{target_attr}" ELSE NULL END) <
           PERCENTILE_CONT(0.5) WITHIN GROUP (
               ORDER BY CASE WHEN "{grp_attr}"='{value2}' THEN "{target_attr}" ELSE NULL END)
),
inter_table AS (SELECT {grouping_string},"{target_attr}","{grp_attr}",total_median
                FROM discretized INNER JOIN filtered USING ({grouping_string})
),
observed AS (
    SELECT {grouping_string},
    COUNT(CASE WHEN "{grp_attr}"='{value1}' AND "{target_attr}">total_median THEN "{target_attr}"
                      ELSE NULL END) AS v1_over,
    COUNT(CASE WHEN "{grp_attr}"='{value2}' AND "{target_attr}">total_median THEN "{target_attr}"
                      ELSE NULL END) AS v2_over,
    COUNT(CASE WHEN "{grp_attr}"='{value1}' AND "{target_attr}"<=total_median THEN "{target_attr}"
                      ELSE NULL END) AS v1_under,
    COUNT(CASE WHEN "{grp_attr}"='{value2}' AND "{target_attr}"<=total_median THEN "{target_attr}"
                      ELSE NULL END) AS v2_under
    FROM inter_table
    GROUP BY {grouping_string}
)
SELECT *
FROM observed NATURAL INNER JOIN filtered
WHERE v1_over+v1_under>={min_group_size} AND v2_over+v2_under>={min_group_size};
