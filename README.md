# Claim Endorsement

This repository contains the implementation and experimental evaluation for the paper "Finding Convincing Views to Endorse a Claim", submitted to VLDB 2025.

##Setup:
1. Create a conda or pip environment on that server with the requirements.txt file.
1. Download the dataset files from https://shorturl.at/AKN67 and place them under the matching directories in data/\<Dataset name\>.
1. Install postgresql on a unix server.
1. Load the data files into separate postgres databases.
1. Create a data/database_connection.env file based on data/database_connection_dummy.env and edit it with the user name, password and IP address for the postgresql server.
   
## Running Experiments
1. In config.py:
   1. set a single RUN_\<DB_name\> flag to True and the others to False. 
   1. Adjust the aggregation function (AGG_TYPE), aggregation attribute (TARGET_ATTR) and 
   group-by attribute (GRP_ATTR), and population groups (in COMPARE_LIST) as needed.

Edit one of the following files (based on the dataset choice): ACS_Experimenting.py, SO_Experimenting.py or FlightLarge_Experimenting.py.
The next instructions are for SO_Experimenting.py but are very similar for the other datasets.

Select the experiment you would like to run and uncomment the matching lines:

* To run all prioritization methods for a given query - uncomment the lines under "main quality experiment".
* To run the sensitivity to number of tuples experiment - uncomment the lines under "num tuples experiment".
* To run the sensitivity to number of columns experiment - uncomment the lines under "num columns experiment".
* To run a predicate-level search (slower) - uncomment the lines under "Predicate level".

Finally, run:
```python SO_Experimenting.py.```

## Generating figures
In ```create_figures.py```, uncomment the lines for the specific figure you would like to create, based on the output of experiments you have run.
* For a comparison of prioritization methods recall over time in different measures, use one of the following lines (based on the choice of database): ```ACS_exp(metrics, "main")```, ```SO_exp(metrics, "main")```, or ```flights_exp(metrics, "main")```.
* For a comparison of sample sizes recall over time in different measures, use one of the following lines (based on the choice of database): ```ACS_exp(metrics, "sample_size")```, ```SO_exp(metrics, "sample_size")```, or ```flights_exp(metrics, "sample_size")```.
* To output the time until 0.95 recall was reached in each measure with various prioritization methods, use one of the following lines: ```ACS_exp([], "table_time_until_recall")```, ```SO_exp([], "table_time_until_recall")```, or ```flights_exp([], "table_time_until_recall")```.
* In the run examples under "main" you will also find corresponding lines for sensitivity to number of tuples or columns (```analyze_scale_sensitivity```) and to values of k (```sensitivity_to_k```). 