SELECT {selecting_string},
COUNT(CASE WHEN "{grp_attr}"='{value1}' THEN 1 ELSE NULL END) as count1,
COUNT(CASE WHEN "{grp_attr}"='{value2}' THEN 1 ELSE NULL END) as count2
FROM {table_name} 
WHERE {where_string}
GROUP BY {grouping_string}
HAVING COUNT(CASE WHEN "{grp_attr}"='{value1}' THEN 1 ELSE NULL END) <
       COUNT(CASE WHEN "{grp_attr}"='{value2}' THEN 1 ELSE NULL END);
