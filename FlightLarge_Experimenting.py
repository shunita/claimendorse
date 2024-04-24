import sys
import config
import pandas as pd
from ClaimEndorseFunctions import *
import time


def create_df_with_all_flights_data():
    df = pd.read_csv("data/flights/flights.csv")
    airports = pd.read_csv("data/flights/airports.csv")
    origin = airports.rename({x: 'ORIGIN_'+x for x in airports.columns}, axis=1)
    dest = airports.rename({x: 'DEST_' + x for x in airports.columns}, axis=1)
    merged1 = df.merge(origin, left_on='ORIGIN_AIRPORT', right_on='ORIGIN_IATA_CODE', how='left')
    merged2 = merged1.merge(dest, left_on='DESTINATION_AIRPORT', right_on='DEST_IATA_CODE', how='left')
    merged2 = merged2.drop(['ORIGIN_IATA_CODE', 'DEST_IATA_CODE'], axis=1)
    merged2.to_csv("data/flights/flights_with_airports.csv")



exclude_list = ["TAIL_NUMBER", "CANCELLED", "ORIGIN_AIRPORT_CODE"]  # this will be heavy for metrics calculation. maybe also flight number?

is_numeric = [#'YEAR',
              #'MONTH',
              'DAY', 
              #'DAY_OF_WEEK', 
              #'FLIGHT_NUMBER',
              'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
              'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
              'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY',
              'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'ORIGIN_LATITUDE',
              'ORIGIN_LONGITUDE', 'DEST_LATITUDE', 'DEST_LONGITUDE']

all_columns = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT_CODE',
               'DEST_AIRPORT_CODE', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
               'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN',
               'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
               'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY',
               'ORIGIN_AIRPORT', 'ORIGIN_CITY', 'ORIGIN_STATE', 'ORIGIN_COUNTRY', 'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE',
               'DEST_AIRPORT', 'DEST_CITY', 'DEST_STATE', 'DEST_COUNTRY', 'DEST_LATITUDE', 'DEST_LONGITUDE', 'DELAYED']
delays = [c for c in all_columns if "DELAY" in c]
exclude_list += delays
if TARGET_ATTR == 'DEPARTURE_DELAY':
    exclude_list += ['ARRIVAL_TIME', ]  # too dependant on the target attribute.
if 'AIRPORT' in GRP_ATTR:
    exclude_list += ['FLIGHT NUMBER']  # no point comparing the same flight number from different origins/arrivals, since it's usually not the same flight.
trans_dict = {c: c.replace("_", " ").lower() for c in all_columns}


if __name__ == '__main__':
    start_time = time.time()
    df = pd.read_csv(config.DATAFRAME_PATH, index_col=0)

    # This should only be done once!
    #print("Uploading to DB")
    #df_to_sql_DB(df)
    #sys.exit()

    #shuffle_and_save(df, exclude_list, is_numeric, trans_dict, start_time, iters=3)
    #run_random_shuffle_over_full_table(df, exclude_list, is_numeric, trans_dict, 3)

    ##################### main quality ################################
    #result_df = run_cherrypicking_with_config(df, exclude_list, is_numeric, trans_dict)
    methods =  ['random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'original_order', 
                'ALL_TOP_K_MERGED', 'ALL_TOP_K_SERIAL',
                '0.01sample:1', '0.01sample:2', '0.01sample:3',
                '0.05sample:1', '0.05sample:2', '0.05sample:3',
                '0.10sample:1', '0.10sample:2', '0.10sample:3']
    run_multiple_methods_for_query(TARGET_ATTR, GRP_ATTR, COMPARE_LIST, AGG_TYPE,
                                   df, exclude_list, is_numeric, trans_dict, methods, stop_at_recall=False)

    # TODO: revise (from here to the end)
    ########### sample size experiment #####################
    #run_sample_guided_experiment([0.01, 0.05, 0.1], 3, df, exclude_list, is_numeric, trans_dict)

    ############## num tuples experiment #########################
    methods = ['random_shuffle:1', 'random_shuffle:2', 'random_shuffle:3', 'original_order', 'ALL_TOP_K_MERGED',
               '0.01sample', 'ALL_TOP_K_SERIAL']
    sample_sizes = range(100000, 1000001, 100000)
    #num_tuples_vs_time_for_full_run(sample_sizes, df, exclude_list, is_numeric, trans_dict, methods)

    ############## num columns experiment #########################
    col_sample_sizes = range(5, 41, 5)
    #num_columns_vs_time_for_full_run(col_sample_sizes, df, exclude_list, is_numeric, trans_dict, methods)


