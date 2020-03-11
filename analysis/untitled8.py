# Imports
import pandas as pd
import numpy as np
import json
import pprint
import oandapyV20
import sys
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.forexlabs as labs

# Homegrown Libraries
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.database import dictionary_insert
from libraries.database import database_execute
from libraries.database import database_retrieve

# Oanda Account Parameters
from configs import oanda_api, oanda_account

import time




# Database Parameters
db = '/Users/user/Desktop/diff/streaming.db'

# Trade Parameters
stop_loss = 15

# Graph Parameters
graphs = {}
graph_lines = ['pair', 'indicator', 'mean_1', 'mean_2'] # 'mean_3']
graph_colors = [(0, 0, 255), (255, 0, 0), (200, 200, 200), 
                (50, 50, 50), (100, 100, 100)]
data_points = 1000
mean_windows = [25, 500, 900]
bids_or_asks = 'bids'




##############################################################################
# Data Structures
##############################################################################


# Trading and Currency Universe
pairs = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD', 
               'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD', 
               'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 
               'EUR_HKD', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
               'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD', 
               'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 
               'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD', 
               'USD_JPY']


def price_stream_to_db(pairs, db, bid_or_ask='bids', queue = False,
                       clear_databases=True, client=oanda_api):
    '''
    Stream Prices from Oanda for pairs list.  Insert into 'pairs' table
    Load data into a q if passed as argument
    '''


    # Streaming Parameters
    api = oandapyV20.API(access_token=oanda_api)
    params ={'instruments': ','.join(pairs),
             'max_records': 10} 
    r = pricing.PricingStream(accountID=oanda_account, params=params)   


        

    count = 0
    for ticks in api.request(r):
        # Start Data Stream
        if count ==  1000: 
            break
        if ticks['type'] == 'PRICE':
            if ticks['instrument'] in pairs:


                    
                count += 1
                    

    return count





start = time.time()
price_stream_to_db(pairs, db, bid_or_ask='bids', queue = False,
                       clear_databases=False, client=oanda_api)
print(time.time() - start)
print(1000 / (time.time() - start) )







