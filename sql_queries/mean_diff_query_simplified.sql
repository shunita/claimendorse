WITH table1 AS (SELECT {selecting_string},AVG("{target_attr}") AS mean1,COUNT("{target_attr}") AS count1,STDDEV("{target_attr}") AS s1
        FROM {table_name}
        WHERE "{grp_attr}"='{value1}'
        GROUP BY {grouping_string}
        HAVING COUNT("{target_attr}")>{min_group_size}
    ),
        table2 AS (SELECT {selecting_string}, AVG("{target_attr}") AS mean2,COUNT("{target_attr}") AS count2,STDDEV("{target_attr}") AS s2
        FROM {table_name}
        WHERE "{grp_attr}"='{value2}'
        GROUP BY {grouping_string}
        HAVING COUNT("{target_attr}")>{min_group_size}
    )
    SELECT {grouping_string}, mean1, count1, s1, mean2, count2, s2
    FROM table1 INNER JOIN table2 USING ({grouping_string})
    WHERE mean1<mean2;

