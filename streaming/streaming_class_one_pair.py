import numpy as np
import pandas as pd
import time
import sys
import copy
import yaml
from datetime import datetime
from multiprocessing import Process, RawArray, Queue, Value
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
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


    def __init__(self, pair):

        # In order to be able to run processes in class properly
        Process.__init__(self)

        # Import Configs File
        self.configs_file = '/Users/user/Desktop/diff/configs.yaml'
        with open(self.configs_file) as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)

        # Pairs
        self.pair = pair
        self.pairs_index = get_tradable_instruments()
        self.pairs_index = [x['name'] for x in self.pairs_index['instruments']]

        # Currencies
        self.currencies_index = self.create_currencies_index()
        self.currency_nom = self.pair.split('_')[0]
        self.currency_den = self.pair.split('_')[1]
        self.currency_nom_subset = self.create_currencies_index()#self.currency_nom)
        self.currency_den_subset = self.create_currencies_index()#self.currency_den)

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
        self.data_points = 20000
        self.graph_points = 20000
        self.debug_row = 1000
        self.pair_to_graph = self.pairs_index.index(self.pair) # remove this when convenient

        # Shared Arrays, Q's and Values
        self.q = Queue()
        self.differences = RawArray('d', self.data_points)  # * len(self.pairs_index))
        self.pairs = RawArray('d', self.data_points)  # * len(self.pairs_index))
        self.calculated = RawArray('d', self.data_points * len(self.pairs_index))
        self.currencies = RawArray('d', self.data_points * len(self.currencies_index))
        self.row = Value('i', 0)


    def create_currencies_index(self):
        usd_provided, usd_inverse = self.create_currency_subset('USD')
        p = [x.replace('_USD', '') for x in np.array(self.pairs_index)[usd_provided]]
        i = [x.replace('USD_', '') for x in np.array(self.pairs_index)[usd_inverse]]
        currencies_index = ['USD'] + p + i
        return currencies_index


    def create_currency_subset(self, currency):
        # Create Subset
        inverse = []
        provided = []
        for pair in list(self.pairs_index):
            if currency.upper() == pair.split('_')[0]:
                inverse.append(True)
                provided.append(False)
            elif currency.upper() == pair.split('_')[1]:
                inverse.append(False)
                provided.append(True)
            else:
                inverse.append(False)
                provided.append(False)
        provided = np.array(provided)
        usd_inverse = np.array(inverse)
        return provided, inverse




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
        # raw_pairs = np.frombuffer(self.pairs, dtype=np.float64)
        # raw_pairs = raw_pairs.reshape(len(self.pairs_index), self.data_points)
        # raw_currencies = np.frombuffer(self.currencies, dtype=np.float64)
        # raw_currencies = raw_currencies.reshape(len(self.currencies_index), self.data_points)
        # raw_calculated = np.frombuffer(self.calculated, dtype=np.float64)
        # raw_calculated = raw_calculated.reshape(len(self.pairs_index), self.data_points)
        # raw_differences = np.frombuffer(self.differences, dtype=np.float64)
        # raw_differences = raw_differences.reshape(len(self.pairs_index), self.data_points)

        raw_pairs = np.frombuffer(self.pairs, dtype=np.float64)
        raw_pairs = raw_pairs.reshape(self.data_points)
        raw_differences = np.frombuffer(self.differences, dtype=np.float64)
        raw_differences = raw_differences.reshape(self.data_points)

        # Required Numpy Arrays
        pairs = np.ones((len(self.pairs_index), self.data_points))
        currencies = np.ones((len(self.currencies_index), self.data_points))
        calculated = np.ones((len(self.pairs_index), self.data_points))
        differences = np.ones((len(self.pairs_index), self.data_points))

        # Fetch Masks for calculating usd price
        usd_provided, usd_inverse = self.create_currency_subset('USD')

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

                ############## Get  @ 50000 points = .0002 ##############
                # Gather most recent data from qu stream
                _pairs = self.q.get()
                row = _pairs[0]
                _pairs = np.array(_pairs[1])

                ############## Roll @ 50000 points = .0001 ##############
                # Roll arrays
                pairs[61, :-1] = pairs[61, 1:]
                # currencies[61, :-1] = currencies[61, 1:]
                # calculated[:, :-1] = calculated[:, 1:]
                differences[61, :-1] = differences[61, 1:]

                # ############## Roll @ 25000 points = .016 ##############
                # # Roll arrays
                # pairs[:, :-1] = pairs[:, 1:]
                # currencies[:, :-1] = currencies[:, 1:]
                # calculated[:, :-1] = calculated[:, 1:]
                # differences[:, :-1] = differences[:, 1:]
                # ############## Roll@ 25000 points  = .016  ##############

                ############## Calculations cycle time @ 50000 points = .0002  ##############
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

                ############## Write to Arrays @ 50000 points = .016 ##############
                # Write Values to Shared Arrays
                # np.copyto(raw_pairs, pairs)
                # np.copyto(raw_currencies, currencies)
                # np.copyto(raw_calculated, calculated)
                # np.copyto(raw_differences, differences)
                # self.row.value = row

                np.copyto(raw_pairs, pairs[61])
                np.copyto(raw_differences, differences[61])
                self.row.value = row

                # Debug
                calculations_end = datetime.now()
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

        global curve1, curve2, data1, data2, viewbox, last_plotted, p, curve_zero_line

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
        curve_zero_line = pg.PlotCurveItem(np.zeros(self.graph_points), pen=pg.mkPen(color='#808080', width=1))

        viewbox.addItem(curve2)
        viewbox.addItem(curve_zero_line)


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
            global curve1, curve2, data1, data2, viewbox, last_plotted, p, curve_zero_line
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


    def graph_indicator_and_pair_rolling_diff(self, pair_index):

        global curve1, curve2, data1, data2, viewbox, last_plotted, p, curve_zero_line, cum_line, sum

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
        data1 = np.frombuffer(self.pairs)  # .reshape(len(self.pairs_index), -1)
        curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=2))
        data2 = np.frombuffer(self.differences)  # .reshape(len(self.pairs_index), -1)
        curve_zero_line = pg.PlotCurveItem(np.zeros(self.graph_points), pen=pg.mkPen(color='#808080', width=1))
        roll = np.zeros(self.graph_points)
        viewbox.addItem(curve2)
        viewbox.addItem(curve_zero_line)


        # I have no idea - maybe try to delete
        def updateViews():
            global viewbox
            viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
            viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)


        updateViews()
        p.getViewBox().sigResized.connect(updateViews)

        # Debugging
        last_plotted = 0
        sum = 0
        cum_line = np.zeros(data2[- self.graph_points:].shape)


        def update():
            global curve1, curve2, data1, data2, viewbox, last_plotted, p, curve_zero_line, cum_line, sum
            # Only Update graph with new data
            start = datetime.now()
            row = self.row.value
            if row > last_plotted:

                sum = cum_line[row - last_plotted]
                cum_line = np.cumsum(data2[- self.graph_points:])  # + sum

                # Draw Line
                curve1.setData(data1[- self.graph_points:])
                curve2.setData(cum_line)
                curve_zero_line.setData(np.ones(self.graph_points) * cum_line.mean())

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


    def graph_covariance_on_and_pair_rolling_diff(self, pair_index):

        global curve1, curve2, data1, data2, viewbox, last_plotted, p, curve_zero_line, cum_line, pair_line, ind_line

        # Instantiate plot
        win = pg.GraphicsWindow(self.pairs_index[pair_index])
        p = win.addPlot(title='CORRELATIONS')
        win.resize(600, 600)

        p.enableAutoRange('y', True)

        # Plot Curves
        pair_line = np.frombuffer(self.pairs)
        ind_line = np.frombuffer(self.differences)

        curve1 = p.plot(pen=pg.mkPen(color='#d8f0f3', width=2))
        curve2 = p.plot(pen=pg.mkPen(color='#78cad3', width=2))
        curve3 = p.plot(pen=pg.mkPen(color='#32909a', width=2))
        curve4 = p.plot(pen=pg.mkPen(color='#1f5a60', width=2))

        last_plotted = 0


        def update():
            global curve1, curve2, viewbox, last_plotted, p, curve_zero_line, cum_line, pair_line, ind_line
            # Only Update graph with new data
            start = datetime.now()
            row = self.row.value
            if row > last_plotted:

                cum_line = np.cumsum(ind_line[- self.graph_points:])
                pair = pair_line[- self.graph_points:]
                df = pd.DataFrame(np.c_[cum_line, pair])

                c1 = df.rolling(2000).corr().iloc[::2, 1].fillna(0).values
                c2 = df.rolling(5000).corr().iloc[::2, 1].fillna(0).values
                c3 = df.rolling(10000).corr().iloc[::2, 1].fillna(0).values
                c4 = df.rolling(15000).corr().iloc[::2, 1].fillna(0).values

                # Draw Line
                curve1.setData(c1)
                curve2.setData(c2)
                curve3.setData(c3)
                curve4.setData(c4)

                # debug
                end = datetime.now()
                if row % self.debug_row == 0:
                    print('Plot cov               {}:     {}'.format(row, c1[-10:]))
                    print('Time to graph          {}:     {}'.format(row, end - start))
                    print('Graphed at             {}:     {}'.format(row, end))
                # Update Latest Row
                last_plotted = row


        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start()
        QtGui.QApplication.instance().exec_()


    def graph_indicator_and_pair_with_delta_points(self, pair_index):

        global curve1, curve2, data1, data2, viewbox, last_plotted, p
        global curve_zero_line, curve_low_point, curve_high_point, data_low_points, data_high_points

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
        p.setLabel('right', 'INDICATOR', color='#00ffff', **{'font-size': '14pt'})

        p.enableAutoRange('y', True)
        viewbox.enableAutoRange('y', True)

        # Plot Curves
        curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
        data1 = np.frombuffer(self.pairs).reshape(len(self.pairs_index), -1)
        curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#00ffff', width=2))
        data2 = np.frombuffer(self.differences).reshape(len(self.pairs_index), -1)
        curve_zero_line = pg.PlotCurveItem(np.zeros(self.graph_points), pen=pg.mkPen(color='#808080', width=1))
        curve_low_point = pg.PlotCurveItem(np.zeros(self.graph_points), pen=pg.mkPen(color='#ff0000', width=1),
                                           symbolPen='w')
        curve_high_point = pg.PlotCurveItem(np.zeros(self.graph_points), pen=pg.mkPen(color='#40ff00', width=1),
                                            symbolPen='w')
        data_low_points = np.empty(self.graph_points)
        data_high_points = np.empty(self.graph_points)

        viewbox.addItem(curve2)
        viewbox.addItem(curve_zero_line)
        viewbox.addItem(curve_low_point)
        viewbox.addItem(curve_high_point)


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
            global curve_zero_line, curve_low_point, curve_high_point, data_low_points, data_high_points
            # Only Update graph with new data
            start = datetime.now()
            row = self.row.value
            if row > last_plotted:


                '''
                a = np.array([6, 6, 6, 3, 3, 3, 4, 5, 5, 5, 6])
                b = np.array([1, 2, 3, 3, 4, 4, 4, 2, 2, 5, 5])


                a_up = a[1:] > a[:-1]
                a_dw = a[1:] < a[:-1]
                b_sm = b[1:] == b[:-1]

                a_up = np.insert(a_up, 0, False)
                a_dw = np.insert(a_dw, 0, False)
                b_sm = np.insert(b_sm, 0, False)


                up_index = a_up & b_sm
                dw_index = a_dw & b_sm



                up_empty = np.zeros(a.shape)
                dw_empty = np.zeros(a.shape)

                up_empty[up_index] = a[up_index]
                dw_empty[dw_index] = a[dw_index]



                plt.plot(np.arange(11), dw_empty, 'o', color='red')
                plt.figure()
                plt.plot(np.arange(11), up_empty, 'o', color='green')
                '''

                # Calculate high points change
                data_low_points = np.zeros(self.graph_points)
                data_high_points = np.zeros(self.graph_points)

                pair = data1[pair_index, - self.graph_points:]
                diff = data2[pair_index, - self.graph_points:]

                ind_up = diff[1:] > diff[:-1] + .000001
                ind_down = diff[1:] < diff[:-1] - .000001
                pair_same = pair[1:] == pair[:-1]

                ind_up = np.insert(ind_up, 0, False)
                ind_down = np.insert(ind_down, 0, False)
                pair_same = np.insert(pair_same, 0, False)

                up_index = ind_up & pair_same
                dw_index = ind_down & pair_same

                data_high_points[up_index] = diff[up_index]
                data_low_points[dw_index] = diff[dw_index]

                # Draw Line
                curve1.setData(data1[pair_index, - self.graph_points:])
                curve2.setData(data2[pair_index, - self.graph_points:])
                curve_high_point.setData(data_high_points)
                curve_low_point.setData(data_low_points)

                # curve1.setData(data1[pair_index,  - (row - self.graph_points:]) # data_points - (latest - 300): -1]
                # curve2.setData(data2[pair_index,  - self.graph_points:])
                # app.processEvents()  ## force complete redraw for every plot
                # debug
                end = datetime.now()
                if row % self.debug_row == 0:
                    print('Plot missed on         {}:     {}'.format(row, pair[-15:] == pair[-16:-1]))
                    print('Plot missed on         {}:     {}'.format(row, row - last_plotted))
                    print('Time to graph          {}:     {}'.format(row, end - start))
                    print('Graphed at             {}:     {}'.format(row, end))
                # Update Latest Row
                last_plotted = row


        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start()
        QtGui.QApplication.instance().exec_()


    def graph_currency_with_pairs(self, currency_index):

        global pair_curves, currency_curve, pair_data, currency_data, viewbox, last_plotted, p

        # Instantiate plot
        win = pg.GraphicsWindow(self.currencies_index[currency_index])
        p = win.addPlot(title=self.currencies_index[currency_index])
        win.resize(600, 600)

        # Try to add a second graph - don't know what any of this does.
        viewbox = pg.ViewBox()
        viewbox.setXLink(p)
        p.scene().addItem(viewbox)
        p.getAxis('right').linkToView(viewbox)
        p.showAxis('right')
        p.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
        p.getAxis('right').setPen(pg.mkPen(color='#ffffff', width=1))
        p.setLabel('left', 'PAIRS', color='#ED15DF', **{'font-size': '14pt'})
        p.setLabel('right', 'CURRENCY', color='#5AED15', **{'font-size': '14pt'})

        p.enableAutoRange('y', True)
        viewbox.enableAutoRange('y', True)

        pair_curves = []

        # Plot Curves
        #
        # curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=2)
        #
        # currency_curve = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=2))
        # currency_data = np.frombuffer(self.currencies).reshape(len(self.currency_index), -1)
        # viewbox.addItem(currency_curve)

        '''
        curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
        data1 = np.frombuffer(self.pairs).reshape(len(self.pairs_index), -1)
        curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=2))
        data2 = np.frombuffer(self.differences).reshape(len(self.pairs_index), -1)
        viewbox.addItem(curve2)
        '''


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
            global pair_curves, currency_curve, pair_data, currency_data, viewbox, last_plotted, p
            # Only Update graph with new data
            start = datetime.now()
            row = self.row.value
            if row > last_plotted:
                # Draw Line
                curve1.setData(data1[self.pair_index, - self.graph_points:])
                curve2.setData(data2[self.pair_index, - self.graph_points:])
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
        # Process(target=self.graph_simple).start()
        # Process(target=self.graph_indicator_and_pair_with_delta_points, args=(self.pairs_index.index('EUR_CAD'),)).start()
        # Process(target=self.graph_indicator_and_pair_rolling_diff, args=(self.pair_to_graph,)).start()
        # Process(target=self.graph_covariance_on_and_pair_rolling_diff, args=(self.pair_to_graph,)).start()

        # Process(target=self.graph_indicator_and_pair, args=(self.pairs_index.index('EUR_USD'),)).start()
        # Process(target=self.graph_indicator_and_pair, args=(self.pairs_index.index('AUD_CAD'),)).start()
        # Process(target=self.graph_indicator_and_pair, args=(self.pairs_index.index('NZD_CHF'),)).start()
        # Process(target=self.graph_currencies).start()
        # Process(target=self.graph_differences).start()
        # Process(target=self.print_currencies).start()


if __name__ == '__main__':
    # instantiate class
    s = Stream('EUR_USD')

    # Start Processes
    s.start_processes()

    # Start Price Steam ( in main Process )
    s.price_stream()











