import numpy as np
import time
import sys
import copy
from datetime import datetime
from multiprocessing import Process, RawArray, Queue, Value, Lock
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore
# from pyqtgraph.ptime import time as pyqt_time
import yaml
from scipy import stats

# Notes
"""
Refactor code:
    Put shared arry in calculations
    let graph do rest of processing
    figure out how many data points I can hande in calculation and set
    have graph variable for how many to points to graph
    can i put this into a class ? 
        Yes but later - for christs sake its a days work.
    
"""


class Stream(Process):

    def __init__(self):
        Process.__init__(self)
        
        # Import Configs File
        self.configs_file = '/Users/user/Desktop/diff/configs.yaml'
        with open(self.configs_file) as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        
        # Stream Parameters
        self.bids_or_asks = 'bids'
        
        # Graph Window
        self.data_points = 10000
        self.debug_row = 100
        self.graph_points = 1000
        
        # Data Windows
        self.mean_windows = [10, 20, 30]
        self.cov_windows = [30, 75, 150]
        self.slope_windows = [30, 75]
        
        # Pairs
        self.pair_to_graph = 18
        self.pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD',
                       'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD',
                       'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP',
                       'EUR_HKD', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
                       'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD',
                       'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD',
                       'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD',
                       'USD_JPY']
        
        # Currencies
        self.currencies_index = list(set('_'.join(self.pairs_index).split('_')))
        self.currencies_index.sort()
        self.currency_1_to_graph = 3
        self.currency_2_to_graph = 8
        
        # Oanda Parameters - Need to update to configs
        self.oanda_api = self.configs['oanda_api']
        self.oanda_account = self.configs['oanda_account']

        # Shared Arrays, Q's and Values
        self.q = Queue()
        self.differences = RawArray('d', self.data_points * len(self.pairs_index))
        self.pairs = RawArray('d', self.data_points * len(self.pairs_index))
        self.calculated = RawArray('d', self.data_points * len(self.pairs_index))
        self.currencies = RawArray('d', self.data_points * len(self.currencies_index))
        self.row = Value('i', 0)


    def price_stream(self):
    
        """
        Stream Prices from Oanda for pairs list.
        Load data into a q if passed as argument
        No sql, Que only
        Load bid and ask for pair into configs file
        """
    
        # Streaming Parameters
        _pairs = np.ones(len(self.pairs_index))
        api = oandapyV20.API(access_token=self.oanda_api)
        params = {'instruments': ','.join(self.pairs_index)}
        r = pricing.PricingStream(accountID=self.oanda_account, params=params)
        
        # Start Data Stream
        row = 1
        while True:
            for ticks in api.request(r):
                if ticks['type'] == 'PRICE' and ticks['instrument'] in self.pairs_index:
                    try:
    
                        # Update Pairs with Latest price.  Set timestamp
                        pair = ticks['instrument']
                        _pairs[self.pairs_index.index(pair)] = float(ticks[self.bids_or_asks][0]['price'])
                        self.q.put([row, _pairs])
                        row += 1
    
                        # Debugging - timestamp - Pair info loaded into Q
                        if row % self.debug_row == 0:
                            print('Oanda Sent             {}:     {}'.format(row, ticks['time']))
                            print('Stream into q          {}:     {}'.format(row, datetime.now()))
    
                    except Exception as e:
                        print('Stream | Calculation exception: {}'.format(e))
    
                    # Load ask and bid data into configs to share with trading module
                    if pair == self.configs['pair']:
                        self.configs['bid'] = float(ticks['bids'][0]['price'])
                        self.configs['ask'] = float(ticks['asks'][0]['price'])
                        with open(self.configs_file, 'w') as f:
                            yaml.dump(self.configs, f)
            
            

    
    
        
    def calculations(self):
        
        # Do what needs to be done with shared arrays
        raw_pairs = np.frombuffer(self.pairs, dtype=np.float64).reshape(len(self.pairs_index), self.data_points)
        raw_currencies = np.frombuffer(self.currencies, dtype=np.float64).reshape(len(self.currencies_index), self.data_points)
        raw_calculated = np.frombuffer(self.calculated, dtype=np.float64).reshape(len(self.pairs_index), self.data_points)
        raw_differences = np.frombuffer(self.differences, dtype=np.float64).reshape(len(self.pairs_index), self.data_points)

        # Required Numpy Arrays
        pairs = np.ones((len(self.pairs_index), self.data_points))
        currencies = np.ones((len(self.currencies_index), self.data_points))
        calculated = np.ones((len(self.pairs_index), self.data_points))
        differences = np.ones((len(self.pairs_index), self.data_points))

        # Inverse and normal denominators for conversion calculation
        inverse = np.zeros((len(self.currencies_index), len(self.pairs_index))).astype(bool)
        given = np.zeros((len(self.currencies_index), len(self.pairs_index))).astype(bool)
        for r in range(inverse.shape[0]):
            for c in range(inverse.shape[1]):
                if self.currencies_index[r] == self.pairs_index[c].split('_')[0]:
                    inverse[r, c] = True
                if self.currencies_index[r] == self.pairs_index[c].split('_')[1]:
                    given[r, c] = True
                    
        # Start
        while True:
            if not self.q.empty():
    
                # Timestamp
                calculations_start = datetime.now()
    
                # Gather most recent data from qu stream
                _pairs = self.q.get()
                row = _pairs[0]
                _pairs = _pairs[1]

                # Roll arrays
                pairs[:, :-1] = pairs[:, 1:]
                currencies[:, :-1] = currencies[:, 1:]
                calculated[:, :-1] = calculated[:, 1:]
                differences[:, :-1] = differences[:, 1:]
                
                # Update Pairs
                pairs[:, -1] = _pairs
                # Calculate newest and update currencies
                a = np.tile(_pairs, (len(self.currencies_index), 1))
                _currencies = 1 / ((a * given).sum(1) + ((1 / a) * inverse).sum(1) + 1)
                currencies[:, -1] = _currencies.copy()
                # Calculate calculated and update calculated (yikes)
                a = np.tile(_currencies, (len(self.pairs_index), 1))
                _calculated = (a * inverse.T).sum(1) / (a * given.T).sum(1)
                calculated[:, -1] = _calculated.copy()
                # Calculate difference and update difference
                _differences = _calculated.copy() - _pairs.copy()
                differences[:, -1] = _differences.copy()
        
                
                # Write Values to Shared Arrays
                np.copyto(raw_pairs, pairs)
                np.copyto(raw_currencies, currencies)
                np.copyto(raw_calculated, calculated)
                np.copyto(raw_differences, differences)
                self.row.value = row
                
                # Debug
                calculations_end = datetime.now()
                if row % self.debug_row == 0:
                    print('Calculation Cycle Time {}:     {}'.format(row, calculations_end - calculations_start))
                    print('Recent Pairs rom Calc  {}:     {}'.format(row, pairs[0, -1]))


    def print_pairs(self):
        a = np.frombuffer(self.pairs).reshape(len(self.pairs_index), -1)
        printed = 0
        while True:
            row = self.row.value            
            if row % self.debug_row == 0 and row > printed:
                print('From Next Function     {}:     {}'.format(row, a[0, -1]))
                printed = row

    def start_processes(self):
        Process(target=self.calculations).start()
        Process(target=self.print_pairs).start()


if __name__ == '__main__':
    
    # instantiate class
    s = Stream()
    
    # Start Processes
    s.start_processes()
    
    # Start Price Steam ( in main Process )
    s.price_stream()
    










