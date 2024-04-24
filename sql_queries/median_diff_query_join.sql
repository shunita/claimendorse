WITH discretized AS (
    SELECT {selecting_string},
           "{grp_attr}",
           "{target_attr}"
    FROM {table_name}
    ),
median AS (SELECT {grouping_string},PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{target_attr}") AS total_median
           FROM discretized WHERE "{target_attr}" IS NOT NULL AND "{grp_attr}"='{value1}' OR "{grp_attr}"='{value2}'
           GROUP BY {grouping_string}
),
inter_table AS (SELECT {grouping_string},"{target_attr}","{grp_attr}",total_median
            FROM discretized INNER JOIN median USING ({grouping_string})
),
v1_over AS (SELECT {grouping_string},COUNT(*) AS v1_over
            FROM inter_table
            WHERE "{grp_attr}"='{value1}' AND "{target_attr}">total_median
            GROUP BY {grouping_string}
),
v2_over AS (SELECT {grouping_string},COUNT(*) AS v2_over
            FROM inter_table
            WHERE "{grp_attr}"='{value2}' AND "{target_attr}">total_median
            GROUP BY {grouping_string}
),
v1_under AS (SELECT {grouping_string},COUNT(*) AS v1_under
            FROM inter_table
            WHERE "{grp_attr}"='{value1}' AND "{target_attr}"<=total_median
            GROUP BY {grouping_string}
),
v2_under AS (SELECT {grouping_string},COUNT(*) AS v2_under
            FROM inter_table
            WHERE "{grp_attr}"='{value2}' AND "{target_attr}"<=total_median
            GROUP BY {grouping_string}
),
median1 AS (SELECT {grouping_string},PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{target_attr}") AS median1, COUNT(*) AS count1
                FROM discretized WHERE "{target_attr}" IS NOT NULL AND "{grp_attr}"='{value1}'
                GROUP BY {grouping_string}
),
median2 AS (SELECT {grouping_string},PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{target_attr}") AS median2, COUNT(*) AS count2
                FROM discretized WHERE "{target_attr}" IS NOT NULL AND "{grp_attr}"='{value2}'
                GROUP BY {grouping_string}
)
SELECT {grouping_string},v1_over,v2_over,v1_under,v2_under,total_median,median1,count1,median2,count2
        FROM v1_over NATURAL INNER JOIN v1_under NATURAL INNER JOIN v2_over NATURAL INNER JOIN
             v2_under NATURAL INNER JOIN median1 NATURAL INNER JOIN median2 NATURAL INNER JOIN median
        WHERE median1<median2 AND v1_over+v1_under>={min_group_size} AND v2_over+v2_under>={min_group_size};
