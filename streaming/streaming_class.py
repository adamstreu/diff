import numpy as np
import time
import sys
import copy
import yaml
from datetime import datetime
from multiprocessing import Process, RawArray, Queue, Value
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from scipy import stats
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.oanda import get_tradable_instruments


"""
Refactor code:
    Implement New Currency Universe
    Speed Up New CUrrency Univers
    graph pair with difference
    figure out how many data points I can hande in calculation and set

"""


class Stream(Process):

    def __init__(self):

        # In order to be able to run processes in class properly
        Process.__init__(self)
        
        # Import Configs File
        self.configs_file = '/Users/user/Desktop/diff/configs.yaml'
        with open(self.configs_file) as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
                
        # Pairs
        self.pairs_index = get_tradable_instruments()
        self.pairs_index = [x['name'] for x in self.pairs_index['instruments']]
        
        # Currencies
        self.currencies_index = self.create_currencies_index()
        # self.currencies_index = list(set('_'.join(self.pairs_index).split('_')))
        # self.currencies_index.sort()
        
        # Streaming Parameters
        self.bids_or_asks = 'bids'
        
        # Data Windows
        self.mean_windows = [10, 20, 30]
        self.cov_windows = [30, 75, 150]
        self.slope_windows = [30, 75]
                
        # Oanda Parameters - Need to update to configs
        self.oanda_api = self.configs['oanda_api']
        self.oanda_account = self.configs['oanda_account']

        # Graphing
        self.graph_delay = 300
        self.data_points = 4000
        self.graph_points = 3000
        self.debug_row = 100
        self.pair_to_graph = 18
        self.currency_1_to_graph = 3
        self.currency_2_to_graph = 8

        # Shared Arrays, Q's and Values
        self.q = Queue()
        self.differences = RawArray('d', self.data_points * len(self.pairs_index))
        self.pairs = RawArray('d', self.data_points * len(self.pairs_index))
        self.calculated = RawArray('d', self.data_points * len(self.pairs_index))
        self.currencies = RawArray('d', self.data_points * len(self.currencies_index))
        self.row = Value('i', 0)


    def create_currencies_index(self):
        usd_provided, usd_inverse = self.create_usd_masks()
        p = [x.replace('_USD', '') for x in np.array(self.pairs_index)[usd_provided]]
        i = [x.replace('USD_', '') for x in np.array(self.pairs_index)[usd_inverse]]
        currencies_index = ['USD'] + p + i
        return currencies_index  


    def create_usd_masks(self):
        # Create Subset
        usd_inverse = []
        usd_provided = []
        for pair in list(self.pairs_index):
            if 'USD' == pair.split('_')[0]:
                usd_inverse.append(True)        
                usd_provided.append(False)
            elif 'USD' == pair.split('_')[1]:
                usd_inverse.append(False)        
                usd_provided.append(True)
            else:
                usd_inverse.append(False)        
                usd_provided.append(False)
        usd_provided = np.array(usd_provided)
        usd_inverse = np.array(usd_inverse)
        return usd_provided, usd_inverse
        

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
        raw_pairs = np.frombuffer(self.pairs, dtype=np.float64)
        raw_pairs = raw_pairs.reshape(len(self.pairs_index), self.data_points)
        raw_currencies = np.frombuffer(self.currencies, dtype=np.float64)
        raw_currencies = raw_currencies.reshape(len(self.currencies_index), self.data_points)
        raw_calculated = np.frombuffer(self.calculated, dtype=np.float64)
        raw_calculated = raw_calculated.reshape(len(self.pairs_index), self.data_points)
        raw_differences = np.frombuffer(self.differences, dtype=np.float64)
        raw_differences = raw_differences.reshape(len(self.pairs_index), self.data_points)

        # Required Numpy Arrays
        pairs = np.ones((len(self.pairs_index), self.data_points))
        currencies = np.ones((len(self.currencies_index), self.data_points))
        calculated = np.ones((len(self.pairs_index), self.data_points))
        differences = np.ones((len(self.pairs_index), self.data_points))

        # Fetch Masks for calculating usd price
        usd_provided, usd_inverse = self.create_usd_masks()
                
        # Make  Mask for calculating _calculated prices
        currency_nominator_mask = np.zeros((len(self.currencies_index), len(self.pairs_index))).astype(bool)
        currency_denominator_mask = np.zeros((len(self.currencies_index), len(self.pairs_index))).astype(bool)
        for i in range(len(self.pairs_index)):
            nom = self.pairs_index[i].split('_')[0]
            den = self.pairs_index[i].split('_')[1]
            currency_nominator_mask[self.currencies_index.index(nom), i] = True
            currency_denominator_mask[self.currencies_index.index(den), i] = True

        # Start
        while True:
            if not self.q.empty():
    
                # Timestamp
                calculations_start = datetime.now()

                ############## Get  @ 25000 points = .0001 ##############
                # Gather most recent data from qu stream
                _pairs = self.q.get()
                row = _pairs[0]
                _pairs = np.array(_pairs[1])
                ############## Get  @ 25000 points = .0001  ##############

                ############## Roll @ 25000 points = .016 ##############
                # Roll arrays
                pairs[:, :-1] = pairs[:, 1:]
                currencies[:, :-1] = currencies[:, 1:]
                calculated[:, :-1] = calculated[:, 1:]
                differences[:, :-1] = differences[:, 1:]
                ############## Roll@ 25000 points  = .016  ##############

                ############## Calculations cycle time @ 25000 points = .0002  ##############
                # Update Pairs
                pairs[:, -1] = _pairs
                # Calculate usd from _pairs
                usd = 1 / (1 + _pairs[usd_provided].sum() + (1 / _pairs[usd_inverse]).sum())
                # Calculate Rest of currencies from usd
                _currencies = np.array([usd] + list(usd * _pairs[usd_provided]) + list(usd / _pairs[usd_inverse]))
                currencies[:, -1] = _currencies.copy()
                # Calculate 'calculated' prices
                a = np.tile(_currencies, (len(self.pairs_index), 1)).T
                _calculated = (a * currency_nominator_mask).sum(0) / (a * currency_denominator_mask).sum(0)
                calculated[:, -1] = _calculated.copy()
                # Calculate Differences
                _differences = _pairs.copy() - _calculated.copy()
                differences[:, -1] = _differences.copy()
                ############## Calculations cycle time @ 25000 points = .0002  ##############

                ############## Write to Arrays @ 25000 points = .006 ##############
                # Write Values to Shared Arrays
                np.copyto(raw_pairs, pairs)
                np.copyto(raw_currencies, currencies)
                np.copyto(raw_calculated, calculated)
                np.copyto(raw_differences, differences)
                self.row.value = row
                ##############  Write to Arrays @ 25000 points = .006  ##############

                calculations_end = datetime.now()

                # Debug
                if row % self.debug_row == 0:
                    print('Calculations Sum       {}:     {}'.format(row, _currencies.sum()))
                    print('Calculation Cycle Time {}:     {}'.format(row, calculations_end - calculations_start))
                    print('Calculation at         {}:     {}'.format(row, datetime.now()))


    def print_currencies(self):
        a = np.frombuffer(self.currencies).reshape(len(self.currencies_index), -1)
        printed = 0
        while True:
            row = self.row.value            
            if row % self.debug_row == 0 and row > printed:
                print('From Next Function     {}:     {}'.format(row, a[:, -1]))
                printed = row


    def graph_differences(self):
        global curve, data, p, last_plotted

        def update():
            global curve, data, ptr, p, last_plotted
            start = datetime.now()
            # Only Update graph with new data
            row = self.row.value
            if row > last_plotted:
                # Draw Line
                for i in range(data.shape[0]):
                    curve[i].setData(data[i, -self.graph_points:])
                # app.processEvents()  ## force complete redraw for every plot
                # debug
                end = datetime.now()
                if row % self.debug_row == 0:
                    print('Plot missed on         {}:     {}'.format(row, row - last_plotted))
                    print('Time to graph          {}:     {}'.format(row, end - start))
                    print('Graphed at             {}:     {}'.format(row, end))
                # Update Latest Row
                last_plotted = row

        # Ready Plot
        app = QtGui.QApplication([])
        p = pg.plot()
        p.setWindowTitle('CURRENCIES')
        p.setLabel('bottom', 'Index', units='B')
        curve = p.plot()
        data = np.frombuffer(self.differences).reshape(len(self.pairs_index), -1)
        last_plotted = 0
        # Describe Curve Set
        curve = []
        for i in range(data.shape[0]):
            c = pg.PlotCurveItem(pen=(i, 9 * 1.3))
            # c.setPos(0, i + 1)
            p.addItem(c)
            curve.append(c)

        # Call Plot Update
        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start()
        QtGui.QApplication.instance().exec_()


    def graph_currencies(self):
        global curve, data, p, last_plotted

        def update():
            global curve, data, ptr, p, last_plotted
            start = datetime.now()
            # Only Update graph with new data
            row = self.row.value
            if row > last_plotted:
                # Draw Line
                for i in range(data.shape[0]):
                    line = data[i, -self.graph_points:].copy()
                    line -= line.mean()
                    line /= line.std()
                    curve[i].setData(line)
                # app.processEvents()  ## force complete redraw for every plot
                # debug
                end = datetime.now()
                if row % self.debug_row == 0:
                    print('Plot missed on         {}:     {}'.format(row, row - last_plotted))
                    print('Time to graph          {}:     {}'.format(row, end - start))
                    print('Graphed at             {}:     {}'.format(row, end))
                # Update Latest Row
                last_plotted = row

        # Ready Plot
        app = QtGui.QApplication([])
        p = pg.plot()
        p.setWindowTitle('CURRENCIES')
        p.setLabel('bottom', 'Index', units='B')
        curve = p.plot()
        data = np.frombuffer(self.currencies).reshape(len(self.currencies_index), -1)
        last_plotted = 0
        # Describe Curve Set
        curve = []
        for i in range(data.shape[0]):
            c = pg.PlotCurveItem(pen=(i, 9 * 1.3))
            # c.setPos(0, i + 1)
            p.addItem(c)
            curve.append(c)

        # Call Plot Update
        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start()
        QtGui.QApplication.instance().exec_()


    def graph_indicator_and_pair(self, pair_index):

        global curve1, curve2, data1, data2, viewbox, last_plotted, p

        # Instantiate plot
        win = pg.GraphicsWindow(self.pairs_index[pair_index])
        p = win.addPlot(title=self.pairs_index[pair_index])
        win.resize(600, 600)

        # Try to add a second graph - don't know what any of this does.
        viewbox = pg.ViewBox()
        viewbox.setXLink(p)
        p.scene().addItem(viewbox)
        p.getAxis('right').linkToView(viewbox)
        p.showAxis('right')
        p.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
        p.getAxis('right').setPen(pg.mkPen(color='#ffffff', width=1))
        p.setLabel('left', 'PAIR', color='#ED15DF', **{'font-size': '14pt'})
        p.setLabel('right', 'INDICATOR', color='#5AED15', **{'font-size': '14pt'})

        p.enableAutoRange('y', True)
        viewbox.enableAutoRange('y', True)

        # Plot Curves
        curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
        data1 = np.frombuffer(self.pairs).reshape(len(self.pairs_index), -1)
        curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=2))
        data2 = np.frombuffer(self.differences).reshape(len(self.pairs_index), -1)
        viewbox.addItem(curve2)

        # I have no idea - maybe try to delete
        def updateViews():
            global viewbox
            viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
            viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
        updateViews()
        p.getViewBox().sigResized.connect(updateViews)

        # Debugging
        last_plotted = 0

        def update():
            global curve1, curve2, data1, data2, viewbox, last_plotted, p
            # Only Update graph with new data
            start = datetime.now()
            row = self.row.value
            if row > last_plotted:
                # Draw Line
                curve1.setData(data1[pair_index, - self.graph_points:])
                curve2.setData(data2[pair_index, - self.graph_points:])
                # curve1.setData(data1[pair_index,  - (row - self.graph_points:]) # data_points - (latest - 300): -1]
                # curve2.setData(data2[pair_index,  - self.graph_points:])
                # app.processEvents()  ## force complete redraw for every plot
                # debug
                end = datetime.now()
                if row % self.debug_row == 0:
                    print('Plot missed on         {}:     {}'.format(row, row - last_plotted))
                    print('Time to graph          {}:     {}'.format(row, end - start))
                    print('Graphed at             {}:     {}'.format(row, end))
                # Update Latest Row
                last_plotted = row



        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start()
        QtGui.QApplication.instance().exec_()


    def start_processes(self):
        Process(target=self.calculations).start()
        Process(target=self.graph_indicator_and_pair, args=(23,)).start()
        # Process(target=self.graph_currencies).start()
        # Process(target=self.graph_differences).start()
        # Process(target=self.print_currencies).start()


if __name__ == '__main__':
    
    # instantiate class
    s = Stream()
    
    # Start Processes
    s.start_processes()
    
    # Start Price Steam ( in main Process )
    s.price_stream()
    










