import numpy as np
import time
import sys
import copy
from datetime import datetime
from multiprocessing import Process, RawArray, Queue
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.ptime import time as pyqt_time
import yaml
from scipy import stats

# Notes
"""

    CAUTION - CALUCLATION IS ON SLEEP FOR WEEKEND WORK !
    
Before usable:

    Pips graph is still a mess.
    So be it
       
Today: 
        
    why is graph restarting
        
    just print out occsional compeltion timestamps/row so i know im not behind
        
    display and graph all sorts of trae paramete
        how many pips.
        quantity
        
    Another indicator - current position of indicator  - current mean position from window
    
    One more graph that just plot the current position of all our slopes and cov - currenct position only.
        Maybe the balls change colors and size when everything lights up to green.....
    Light button on good conditions


Sunday night:

    Test out trade parameters and placements.
    Test out new graphs, etc.  Make sure everything is working
    No alcohol today: fresh for tomorrow.  Get some excersise
    

Monday Morning:

    Up EARLY obviously.
    to watch.
    Dont fuck around too much.
    

Later:
    

    CLAULATION MATRIX ON ALL THAT ARENT
    
    Remove misc data from mean, cov

    buttons / lights that flash when trade conditions are right
    
    Slope ratios to between -1 and 1

    graphs tithgt layout.
    
    Can pay fees for gauranteed stop loss execution orders.
    
    
Much Later:

    what is historical position for instruemtn, position book, etc.

    how does position book affect price movements and turn arounds, etc.
    Does my indicator have anything to do with position book ? 
    
    does oanda currnecy heatmap reflect anything?
    How about my curreny charts?     
    
    graphs one process (?) and in nice layout window ? 


    log traddes

    set mean windows ( and data points ) in graph
    
    Timescale
        Right now I update plot only on data coming in - which alters the time scale .
    

PERFORMANCE NOTES AND IMPORVEMENTS:

    Current Summary:
        
        Pretty much there on the basics.
    
    Some Notes:
           
           Stream Peak records:
                Possible to recieve 4 prices per second per pair./
                that's 144 possible total for my currency universe
                That means each needs to be processed in .007 seconds.
                I can not meet this goal.
                
                Calculation Duration:
                    My simplest, np based calculations.  No means, no cov, etc.
                    
                    5000 data points:   ~ 
                    300000 data points: ~
                    100000 data Points: ~ 
                        giving me a max ( again - small data point size, of 100 updates per second
                    Conclusion: shortcuts will need to be consiously managed
            
                Mean line:
                    The litte for loop doesn't add too much
            
            Streaming:
                I do like that it is decoupled. 
                However, it is not processor heavy ( at all ), I might take up a processor that
                    might be better divoted to a calculation ( thinking mena,d cov, regressions, etc.
                    Actually - I can just run the stream in the main process......
                The plots can 'catch up' from missed data.  
                The calculation can't ( although we could backfill...........)
                So as soon as the calc are delayed the whole thing is bad( currently and ieally)
         major tiks on ly at zero
         change colors
         pip graph not right yet      MAYVE HAVE TO INDICATOR ON CONSISTENT SCALE  PAAN THEN NOT SCALE
        one mean line on indicator - timline mean   CALCS ARENT CORRECT YET.  WRITE IN  APAUE PLOT BUTTON.  CANWE ALTER ONE OF THE SCALES TO MAKE THEIR VARIANCE EQUAL AN CENTER THEM AT ZERO?
"""


# Parameters
if True:

    # Import Configs File
    configs_file = '/Users/user/Desktop/diff/configs.yaml'
    with open(configs_file) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Stream Parameters
    bids_or_asks = 'bids'

    # Window
    data_points = 10000
    mean_windows = [10, 20, 30]
    cov_windows = [30, 75, 150]
    slope_windows = [30, 75]


    # Shared Arrays and Q's
    q_streaming = Queue()
    q_covariance = Queue()
    raw_diff = RawArray('d', data_points + 1)
    raw_pair = RawArray('d', data_points + 1)
    raw_mean_1 = RawArray('d', data_points + 1)
    raw_mean_2 = RawArray('d', data_points + 1)
    raw_mean_3 = RawArray('d', data_points + 1)
    raw_cur_1 = RawArray('d', data_points + 1)
    raw_cur_2 = RawArray('d', data_points + 1)
    raw_cov_1 = RawArray('d', data_points + 1)
    raw_cov_2 = RawArray('d', data_points + 1)
    raw_cov_3 = RawArray('d', data_points + 1)
    raw_pair_slope_1 = RawArray('d', data_points + 1)
    raw_pair_slope_2 = RawArray('d', data_points + 1)
    raw_ind_slope_1 = RawArray('d', data_points + 1)
    raw_ind_slope_2 = RawArray('d', data_points + 1)
    raw_running_sum_1 = RawArray('d', data_points + 1)
    raw_running_sum_2 = RawArray('d', data_points + 1)
    raw_running_sum_3 = RawArray('d', data_points + 1)
    raw_diff_by_mean = RawArray('d', data_points + 1)

    # Pairs
    pair_to_graph = 18
    pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD',
                   'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD',
                   'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP',
                   'EUR_HKD', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
                   'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD',
                   'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD',
                   'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD',
                   'USD_JPY']

    # Currencies
    currencies_index = list(set('_'.join(pairs_index).split('_')))
    currencies_index.sort()
    currency_1_to_graph = 3
    currency_2_to_graph = 8

    # Oanda Parameters - Need to update to configs
    oanda_api = configs['oanda_api']
    oanda_account = configs['oanda_account']

    debug_row = 1000


def price_stream(pairs_index, bids_or_asks, q, oanda_api, oanda_account, configs, debug_row):

    """
    Stream Prices from Oanda for pairs list.
    Load data into a q if passed as argument
    No sql, Que only
    Load bid and ask for pair into configs file
    """

    # Streaming Parameters
    _pairs = np.ones(len(pairs_index))
    api = oandapyV20.API(access_token=oanda_api)
    params = {'instruments': ','.join(pairs_index)}
    r = pricing.PricingStream(accountID=oanda_account, params=params)

    # Start Data Stream
    row = 1
    while True:
        for ticks in api.request(r):
            if ticks['type'] == 'PRICE' and ticks['instrument'] in pairs_index:
                try:

                    # Update Pairs with Latest price.  Set timestamp
                    pair = ticks['instrument']
                    _pairs[pairs_index.index(pair)] = float(ticks[bids_or_asks][0]['price'])
                    q.put([row, _pairs])
                    row += 1

                    # Debugging - timestamp - Pair info loaded into Q
                    if row % debug_row == 0:
                        print('Oanda Sent            {}:     {}'.format(row, ticks['time']))
                        print('Stream into q         {}:     {}'.format(row, datetime.now()))

                except Exception as e:
                    print('Stream | Calculation exception: {}'.format(e))

                # Load ask and bid data into configs to share with trading module
                if pair == configs['pair']:
                    configs['bid'] = float(ticks['bids'][0]['price'])
                    configs['ask'] = float(ticks['asks'][0]['price'])
                    with open(configs_file, 'w') as f:
                        yaml.dump(configs, f)


def calculations(pairs_index, currencies_index, q, data_points, pair_to_graph,
                 raw_diff, raw_pair,
                 raw_mean_1, raw_mean_2, raw_mean_3, mean_windows,
                 raw_cov_1, raw_cov_2, raw_cov_3, cov_windows,
                 raw_cur_1, raw_cur_2, currency_1_to_graph, currency_2_to_graph,
                 raw_pair_slope_1, raw_pair_slope_2, raw_ind_slope_1, raw_ind_slope_2, slope_windows,
                 debug_row, raw_running_sum_1, raw_running_sum_2, raw_running_sum_3,
                 raw_diff_by_mean):

    # Read Raw Array
    R_diff = np.frombuffer(raw_diff, dtype=np.float64).reshape(data_points + 1)
    R_pair = np.frombuffer(raw_pair, dtype=np.float64).reshape(data_points + 1)
    R_mean_1 = np.frombuffer(raw_mean_1, dtype=np.float64).reshape(data_points + 1)
    R_mean_2 = np.frombuffer(raw_mean_2, dtype=np.float64).reshape(data_points + 1)
    R_mean_3 = np.frombuffer(raw_mean_3, dtype=np.float64).reshape(data_points + 1)
    R_cov_1 = np.frombuffer(raw_cov_1, dtype=np.float64).reshape(data_points + 1)
    R_cov_2 = np.frombuffer(raw_cov_2, dtype=np.float64).reshape(data_points + 1)
    R_cov_3 = np.frombuffer(raw_cov_3, dtype=np.float64).reshape(data_points + 1)
    R_cur_1 = np.frombuffer(raw_cur_1, dtype=np.float64).reshape(data_points + 1)
    R_cur_2 = np.frombuffer(raw_cur_2, dtype=np.float64).reshape(data_points + 1)
    R_pair_slope_1 = np.frombuffer(raw_pair_slope_1, dtype=np.float64).reshape(data_points + 1)
    R_pair_slope_2 = np.frombuffer(raw_pair_slope_2, dtype=np.float64).reshape(data_points + 1)
    R_ind_slope_1 = np.frombuffer(raw_ind_slope_1, dtype=np.float64).reshape(data_points + 1)
    R_ind_slope_2 = np.frombuffer(raw_ind_slope_2, dtype=np.float64).reshape(data_points + 1)
    R_running_sum_1 = np.frombuffer(raw_running_sum_1, dtype=np.float64).reshape(data_points + 1)
    R_running_sum_2 = np.frombuffer(raw_running_sum_2, dtype=np.float64).reshape(data_points + 1)
    R_running_sum_3 = np.frombuffer(raw_running_sum_3, dtype=np.float64).reshape(data_points + 1)
    R_diff_by_mean = np.frombuffer(raw_diff_by_mean, dtype=np.float64).reshape(data_points + 1)



    # Required Numpy Arrays
    currencies = np.ones((len(currencies_index), data_points))
    pairs = np.ones((len(pairs_index), data_points))
    differences = np.ones((len(pairs_index), data_points))
    calculated = np.ones((len(pairs_index), data_points))
    diff_by_mean = np.ones(data_points)
    # mean_lines = np.zeros((len(mean_windows), len(pairs_index), data_points))
    covariances = np.ones((len(cov_windows), len(pairs_index), data_points))
    pair_slopes = np.ones((len(slope_windows), data_points))
    ind_slopes = np.ones((len(slope_windows), data_points))
    # running_sum_lines = np.ones((len(mean_windows), len(pairs_index), data_points))
    mean_lines = np.ones((len(mean_windows), data_points))
    running_sum_lines = np.ones((len(mean_windows), data_points))


    # Inverse and normal denominators for conversion calculation
    inverse = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
    given = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
    for r in range(inverse.shape[0]):
        for c in range(inverse.shape[1]):
            if currencies_index[r] == pairs_index[c].split('_')[0]:
                inverse[r, c] = True
            if currencies_index[r] == pairs_index[c].split('_')[1]:
                given[r, c] = True
    # Start
    while True:
        if not q.empty():

            # Timestamp
            calculations_start = datetime.now()

            # Gather most recent data from qu stream
            _pairs = q.get()
            row = _pairs[0]
            _pairs = _pairs[1]

            # Roll arrays
            pairs[:, :-1] = pairs[:, 1:]
            diff_by_mean[:-1] = diff_by_mean[1:]
            differences[:, :-1] = differences[:, 1:]
            calculated[:, :-1] = calculated[:, 1:]
            currencies[:, :-1] = currencies[:, 1:]
            pair_slopes[:, :-1] = pair_slopes[:, 1:]
            ind_slopes[:, :-1] = ind_slopes[:, 1:]
            # mean_lines[:, :, :-1] = mean_lines[:, :, 1:]
            mean_lines[:, :-1] = mean_lines[:, 1:]
            running_sum_lines[:, :-1] = running_sum_lines[:, 1:]
            covariances[:, :, :-1] = covariances[:, :, 1:]



            # Update Pairs
            pairs[:, -1] = _pairs
            # Calculate newest and update currencies
            a = np.tile(_pairs, (len(currencies_index), 1))
            _currencies = 1 / ((a * given).sum(1) + ((1 / a) * inverse).sum(1) + 1)
            currencies[:, -1] = _currencies.copy()
            # Calculate calculated and update calculated (yikes)
            a = np.tile(_currencies, (len(pairs_index), 1))
            _calculated = (a * inverse.T).sum(1) / (a * given.T).sum(1)
            calculated[:, -1] = _calculated.copy()
            # Calculate difference and update difference
            _differences = _pairs.copy() - _calculated.copy()  # Should go in same directions now ( I suppose )
            differences[:, -1] = _differences.copy()
            # Mean lines - just make difference while data is populating

            # for i in range(len(mean_windows)):  # Could do with array calculation
            #     if row < mean_windows[i] + 200:
            #         # mean_lines[i, :, -1] = differences[:, -row:].mean(1)
            #         mean_lines[i, :, -1] = _differences
            #     else:
            #         mean_lines[i, :, -1] = differences[:, -mean_windows[i]:].mean(1)

            # Mean Lines - should do this with arrays
            for j in range(len(mean_windows)):
                mean_lines[j, -1] = differences[pair_to_graph, - mean_windows[j]:].mean()



            # Running Sum Lines
            for j in range(len(mean_windows)):
                running_sum_lines[j,  -1] = (differences[pair_to_graph, -1] - mean_lines[j, -1]) + running_sum_lines[j, -2]

            # Diff by mean
            diff_by_mean[-1] = ((differences[pair_to_graph, -1] / mean_lines[1, -1]) - 1 ) + diff_by_mean[-2]

            # Try Again
            diff_by_mean[-1] = ((pairs[pair_to_graph, -1] * (1 - mean_lines[1, -1])) + differences[pair_to_graph, -1])
            diff_by_mean[-1] /= mean_lines[1, -1]

            # Try Again
            diff_by_mean[-1] = (differences[pair_to_graph, -1] / mean_lines[0, -1]) - 1
            #diff_by_mean[-1] += diff_by_mean[-2]

            diff_by_mean[-1] = (differences[pair_to_graph, -1] - mean_lines[0, -1])
            diff_by_mean[-1] += diff_by_mean[-2]

            # ROlling diff
            diff_by_mean[-1] = differences[pair_to_graph, -1] - differences[pair_to_graph, -2]
            diff_by_mean[-1] += diff_by_mean[-2]


            # Covariance
            for j in range(len(cov_windows)):
                _covariances = []
                a = differences[pair_to_graph, data_points - cov_windows[j]:]
                b = pairs[pair_to_graph, data_points - cov_windows[j]:]
                if len(set(list(a))) == 1 or len(set(list(b))) == 1:
                    _covariances.append(0)
                    # print(row, j, len(set(list(a))), len(set(list(b))))
                else:
                    comb = np.c_[a, b].T
                    _covariances.append(np.corrcoef(comb)[0][1])
                covariances[j, :, -1] = np.array(_covariances)

            # Slopes
            for j in range(len(slope_windows)):
                pair_slopes[j, -1] = stats.linregress(np.arange(slope_windows[j]),
                                                      pairs[pair_to_graph, - slope_windows[j]:])[0] # data_points - slope_windows[j]:])
                ind_slopes[j, -1] = stats.linregress(np.arange(slope_windows[j]),
                                                     differences[pair_to_graph, - slope_windows[j]:])[0]







            # Timestamp - Calculations_complete
            array_load_start = datetime.now()

            # Copy data to shared arrays.  Takes: ~ .00005 seconds
            np.copyto(R_diff[: -1], differences[pair_to_graph])
            np.copyto(R_pair[: -1], pairs[pair_to_graph])
            np.copyto(R_cur_1[: -1], currencies[currency_1_to_graph])
            np.copyto(R_cur_2[: -1], currencies[currency_2_to_graph])
            # np.copyto(R_mean_1[: -1], mean_lines[0, pair_to_graph, :])
            # np.copyto(R_mean_2[: -1], mean_lines[1, pair_to_graph, :])
            # np.copyto(R_mean_3[: -1], mean_lines[2, pair_to_graph, :])
            np.copyto(R_mean_1[: -1], mean_lines[0, :])
            np.copyto(R_mean_2[: -1], mean_lines[1, :])
            np.copyto(R_mean_3[: -1], mean_lines[2, :])
            np.copyto(R_cov_1[: -1], covariances[0, pair_to_graph, :])
            np.copyto(R_cov_2[: -1], covariances[1, pair_to_graph, :])
            np.copyto(R_cov_3[: -1], covariances[2, pair_to_graph, :])
            np.copyto(R_pair_slope_1[: -1], pair_slopes[0, :])
            np.copyto(R_pair_slope_2[: -1], pair_slopes[1, :])
            np.copyto(R_ind_slope_1[: -1], ind_slopes[0, :])
            np.copyto(R_ind_slope_2[: -1], ind_slopes[1, :])
            np.copyto(R_running_sum_1[: -1], running_sum_lines[0, :])
            np.copyto(R_running_sum_2[: -1], running_sum_lines[1, :])
            np.copyto(R_running_sum_3[: -1], running_sum_lines[2, :])
            np.copyto(R_diff_by_mean[: -1], diff_by_mean[:])


            # Copy Row
            np.copyto(R_diff[-1:], row)
            np.copyto(R_pair[-1:], row)
            np.copyto(R_cur_1[-1:], row)
            np.copyto(R_cur_2[-1:], row)
            np.copyto(R_mean_1[-1:], row)
            np.copyto(R_mean_2[-1:], row)
            np.copyto(R_mean_3[-1:], row)
            np.copyto(R_cov_1[-1:], row)
            np.copyto(R_cov_2[-1:], row)
            np.copyto(R_cov_3[-1:], row)
            np.copyto(R_pair_slope_1[-1:], row)
            np.copyto(R_pair_slope_2[-1:], row)
            np.copyto(R_ind_slope_1[-1:], row)
            np.copyto(R_ind_slope_2[-1:], row)
            np.copyto(R_running_sum_1[-1:], row)
            np.copyto(R_running_sum_2[-1:], row)
            np.copyto(R_running_sum_3[-1:], row)
            np.copyto(R_diff_by_mean[-1:], row)

            # Timestamp - Calculations complete
            end_calculations = datetime.now()

            # Debug
            if row % debug_row == 0:
                calculation_time = array_load_start - calculations_start
                load_time = end_calculations - array_load_start
                print('End Calculation cycle {}:     {}'.format(row, end_calculations))
                print('Calculation time      {}:     {}'.format(row, calculation_time))
                print('Load array  time      {}:     {}'.format(row, load_time))
                # calc_per_sec = calculation_time.seconds + calculation_time.microseconds * 0.000001
                # calc_per_sec += (load_time.seconds + load_time.microseconds * 0.000001)
                # calc_per_sec = int(100 / calc_per_sec)
                # print('Calculations Per Sec. {}:     {}'.format(row, calc_per_sec))





def graph_pips(data_points, raw_line, debug_row):

    global curve1, data1, curve2, curve3, data2, data3, ptr, p, latest_pip, latest, last_plotted

    # Set Plot
    win = pg.GraphicsWindow()
    win.resize(1000, 600)
    win.setWindowTitle('Targets')
    p = win.addPlot(title="Updating plot")
    p.enableAutoRange('y', True)

    # Set curves
    curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=1))
    data1 = np.frombuffer(raw_line).reshape(data_points + 1)
    curve2 = p.plot(pen=pg.mkPen(color='#ffffff', width=1))
    data2 = np.ones(data_points + 1) * data1[-2]
    curve3 = p.plot(pen=pg.mkPen(color='#ffffff', width=1))
    data3 = np.ones(data_points + 1) * -data1[-2]

    # TRacking
    last_plotted = 0
    latest = int(data1[-1])
    latest_pip = 0

    def update():
        global curve1, data1, curve2, curve3, data2, data3, ptr, p, latest_pip, latest, last_plotted

        latest = int(data1[-1])
        if latest > last_plotted:
            last_plotted = latest

            try:
                with open(configs_file) as f:
                    configs = yaml.load(f, Loader=yaml.FullLoader)
                pips = configs['pips']
                data2[:] = data1[-2] + (pips * .0001)
                data3[:] = data1[-2] - (pips * .0001)
                curve1.setData(data1[data_points - (latest - 300): -1])
                curve2.setData(data2[data_points - (latest - 300): -1])
                curve3.setData(data3[data_points - (latest - 300): -1])
            except:
                pass



    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10000)
    QtGui.QApplication.instance().exec_()


def graph_double(data_points, line1, line2, title, title1, title2, debug_row, grid=False, set_y=False):

    global curve1, curve2, data1, data2, viewbox
    global missed, latest, number_missed, last_plotted, p

    # Instantiate plot
    win = pg.GraphicsWindow(title)
    win.resize(600, 600)
    p = win.addPlot(title=title)

    # Try to add a second graph - don't know what any of this does.
    viewbox = pg.ViewBox()
    viewbox.setXLink(p)
    if set_y:
        viewbox.setYLink(p)
    p.scene().addItem(viewbox)
    p.getAxis('right').linkToView(viewbox)
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#ffffff', width=1))
    p.setLabel('left', title1, color='#ED15DF', **{'font-size': '14pt'})
    p.setLabel('right', title2, color='#5AED15', **{'font-size': '14pt'})
    if grid: p.showGrid(y=True)
    p.enableAutoRange(y=True)
    viewbox.enableAutoRange(y=True)
    # p.enableAutoRange('y', True)

    # Plot Curves
    data1 = np.frombuffer(line1).reshape(data_points + 1)
    data2 = np.frombuffer(line2).reshape(data_points + 1)
    curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=1))
    curve2 = p.plot(pen=pg.mkPen(color='#5AED15', width=1))
    # curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=1))
    # viewbox.addItem(curve2)

    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging
    missed = 0
    last_plotted = 0
    latest = int(data1[-1])

    def update():
        # Don't update plot without new data
        global curve1, curve2, data1, data2, title, viewbox
        global missed, latest, number_missed, last_plotted, p

        latest = int(data2[-1])
        if latest > last_plotted:
            last_plotted = latest

            # Update Plot
            curve1.setData(data1[data_points - (latest - 300): -1])
            curve2.setData(data2[data_points - (latest - 300): -1] + data1[data_points - (latest - 300): -1])

            # Debugging
            if latest - last_plotted > 1:
                number_missed = (latest - last_plotted) - 1
                missed += number_missed
                # print(latest, last_plotted, number_missed, missed)

            # Debugging
            # if latest % debug_row == 0:
            #     # print('single updates: {}'.format(latest))
            #     print('{}: {}, {}'.format('?', data1[-2], data2[-2]))
            #     # print('{}            {}:     {}'.format(title1 + '  ' + title2, latest, datetime.now()))
            #     # print('{} Missed   {}:     {}'.format(title, latest, missed / latest))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    QtGui.QApplication.instance().exec_()



def one_on_top(data_points, line1, line2, mean_line, diff_by_mean, title, title1, title2, debug_row, grid=False, set_y=False):

    global curve1, curve2, curve3, curve_diff_mean, data1, data2, data_diff_mean, viewbox
    global missed, latest, number_missed, last_plotted, p

    # Instantiate plot
    win = pg.GraphicsWindow('One on Top')
    win.resize(600, 600)
    p = win.addPlot(title='One on Top')

    # Try to add a second graph - don't know what any of this does.
    viewbox = pg.ViewBox()
    viewbox.setXLink(p)
    if set_y:
        viewbox.setYLink(p)
    p.scene().addItem(viewbox)
    p.getAxis('right').linkToView(viewbox)
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#ffffff', width=1))
    p.setLabel('left', title1, color='#ED15DF', **{'font-size': '14pt'})
    p.setLabel('right', title2, color='#5AED15', **{'font-size': '14pt'})
    if grid: p.showGrid(y=True)
    p.enableAutoRange(y=True)
    viewbox.enableAutoRange(y=True)
    # p.enableAutoRange('y', True)

    # Plot Curves
    data1 = np.frombuffer(line1).reshape(data_points + 1)
    data2 = np.frombuffer(line2).reshape(data_points + 1)
    data3 = np.frombuffer(mean_line).reshape(data_points + 1)
    data_diff_mean = np.frombuffer(diff_by_mean).reshape(data_points + 1)
    curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=1))
    curve2 = p.plot(pen=pg.mkPen(color='#5AED15', width=1))
    curve3 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=1))
    curve_diff_mean = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=1))
    # curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=1))
    viewbox.addItem(curve3)
    viewbox.addItem(curve_diff_mean)
    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging
    missed = 0
    last_plotted = 0
    latest = int(data1[-1])

    def update():
        # Don't update plot without new data
        global curve1, curve2, curve3, curve_diff_mean, data1, data2, data_diff_mean, title, viewbox
        global missed, latest, number_missed, last_plotted, p

        latest = int(data2[-1])
        if latest > last_plotted:
            last_plotted = latest

            #Update Plot
            # l = data2[data_points - (latest - 300): -1] + data1[data_points - (latest - 300): -1]
            # l -= data3[data_points - (latest - 300): -1]
            # curve1.setData(data1[data_points - (latest - 300): -1])
            # curve2.setData(l)
            # a = data2[data_points - (latest - 300): -1] / data3[data_points - (latest - 300): -1]     # l - data1[data_points - (latest - 300): -1]
            # curve3.setData(1 - a)
            #
            curve1.setData(data1[data_points - (latest - 300): -1])
            curve_diff_mean.setData(data_diff_mean[data_points - (latest - 300): -1])






            # Debugging
            if latest - last_plotted > 1:
                number_missed = (latest - last_plotted) - 1
                missed += number_missed
                # print(latest, last_plotted, number_missed, missed)

            # Debugging
            # if latest % debug_row == 0:
            #     # print('single updates: {}'.format(latest))
            #     print('{}: {}, {}'.format('?', data1[-2], data2[-2]))
            #     # print('{}            {}:     {}'.format(title1 + '  ' + title2, latest, datetime.now()))
            #     # print('{} Missed   {}:     {}'.format(title, latest, missed / latest))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    QtGui.QApplication.instance().exec_()

def graph_running_sum(data_points, line1, line2, line3, pair_line, debug_row, grid=False, set_y=False):

    global curve1, curve2, curve3, curve4, data1, data2, data3, data4, viewbox
    global missed, latest, number_missed, last_plotted, p

    # Instantiate plot
    win = pg.GraphicsWindow()
    win.resize(600, 600)
    p = win.addPlot(title='Running Means')

    # Try to add a second graph - don't know what any of this does.
    viewbox = pg.ViewBox()
    viewbox.setXLink(p)
    if set_y:
        viewbox.setYLink(p)
    p.scene().addItem(viewbox)
    p.getAxis('right').linkToView(viewbox)
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#ffffff', width=1))
    p.setLabel('left', color='#ED15DF', **{'font-size': '14pt'})
    p.setLabel('right', color='#5AED15', **{'font-size': '14pt'})
    if grid: p.showGrid(y=True)

    # p.enableAutoRange('y', True)

    # Plot Curves
    data1 = np.frombuffer(line1).reshape(data_points + 1)
    data2 = np.frombuffer(line2).reshape(data_points + 1)
    data3 = np.frombuffer(line3).reshape(data_points + 1)
    data4 = np.frombuffer(pair_line).reshape(data_points + 1)
    curve1 = p.plot(pen=pg.mkPen(color='#0A66C2', width=1)) #b
    curve2 = p.plot(pen=pg.mkPen(color='#F00E19', width=1)) # r
    curve3 = p.plot(pen=pg.mkPen(color='#0EF012', width=1)) # g
    curve4 = pg.PlotCurveItem(pen=pg.mkPen(color='#ffffff', width=1)) # w
    # curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#F00E19', width=1)) # r
    # curve4 = p.plot(pen=pg.mkPen(color='#ffffff', width=1)) # w
    viewbox.addItem(curve4)



    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging
    missed = 0
    last_plotted = 0
    latest = int(data1[-1])

    def update():
        # Don't update plot without new data
        global curve1, curve2, curve3, curve4, data1, data2, data3, data4, viewbox
        global missed, latest, number_missed, last_plotted, p

        latest = int(data1[-1])
        if latest > last_plotted:
            last_plotted = latest

            # Update Plot
            curve1.setData(data1[data_points - (latest - 300): -1])
            curve2.setData(data2[data_points - (latest - 300): -1])
            curve3.setData(data3[data_points - (latest - 300): -1])
            curve4.setData(data4[data_points - (latest - 300): -1])

            # # Debugging
            # if latest - last_plotted > 1:
            #     number_missed = (latest - last_plotted) - 1
            #     missed += number_missed
                # print(latest, last_plotted, number_missed, missed)

            # Debugging
            # if latest % debug_row == 0:
            #     # print('single updates: {}'.format(latest))
            #     print('{}: {}, {}'.format('?', data1[-2], data2[-2]))
            #     # print('{}            {}:     {}'.format(title1 + '  ' + title2, latest, datetime.now()))
            #     # print('{} Missed   {}:     {}'.format(title, latest, missed / latest))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    QtGui.QApplication.instance().exec_()




def graph_double_zeroed(data_points, line1, line2, mean_line, title, title1, title2, debug_row, grid=False, set_y=False):

    global curve1, curve2, curve3, data1, data2, data3, viewbox
    global missed, latest, number_missed, last_plotted, p

    # Instantiate plot
    win = pg.GraphicsWindow(title)
    win.resize(600, 600)
    p = win.addPlot(title=title)

    # Try to add a second graph - don't know what any of this does.
    viewbox = pg.ViewBox()
    viewbox.setXLink(p)
    if set_y:
        viewbox.setYLink(p)
    p.scene().addItem(viewbox)
    p.getAxis('right').linkToView(viewbox)
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#ffffff', width=1))
    p.setLabel('left', title1, color='#ED15DF', **{'font-size': '14pt'})
    p.setLabel('right', title2, color='#5AED15', **{'font-size': '14pt'})
    if grid: p.showGrid(y=True)

    # p.enableAutoRange('y', True)
    p.setAutoPan(x=None, y=True)

    # Plot Curves
    curve1 = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
    data1 = np.frombuffer(line1).reshape(data_points + 1)
    curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=2))
    data2 = np.frombuffer(line2).reshape(data_points + 1)
    curve3 = pg.PlotCurveItem(pen=pg.mkPen(color='#ffffff', width=1))
    data3 = np.frombuffer(mean_line).reshape(data_points + 1)
    viewbox.addItem(curve2)
    viewbox.addItem(curve3)

    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging
    missed = 0
    last_plotted = 0
    latest = int(data1[-1])

    def update():
        # Don't update plot without new data
        global curve1, curve2, curve3, data1, data2, data3, title, viewbox
        global missed, latest, number_missed, last_plotted, p

        latest = int(data2[-1])
        if latest > last_plotted:
            last_plotted = latest

            # Update Plot
            curve1.setData((data1[data_points - (latest - 300): -1]) - data1[:-1].mean())
            curve2.setData((data2[data_points - (latest - 300): -1]) - data2[:-1].mean())
            curve3.setData((data3[data_points - (latest - 300): -1]) - data3[:-1].mean())
            # Debugging
            if latest - last_plotted > 1:
                number_missed = (latest - last_plotted) - 1
                missed += number_missed
                # print(latest, last_plotted, number_missed, missed)

            # Debugging
            # if latest % debug_row == 0:
            #     # print('single updates: {}'.format(latest))
            #     print('{}: {}, {}'.format('?', data1[-2], data2[-2]))
            #     # print('{}            {}:     {}'.format(title1 + '  ' + title2, latest, datetime.now()))
            #     # print('{} Missed   {}:     {}'.format(title, latest, missed / latest))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()






if __name__ == '__main__':

    # Calculation Process
    calc = Process(target=calculations,
                   args=(pairs_index, currencies_index, q_streaming, data_points, 18,
                         raw_diff, raw_pair,
                         raw_mean_1, raw_mean_2, raw_mean_3, mean_windows,
                         raw_cov_1, raw_cov_2, raw_cov_3, cov_windows,
                         raw_cur_1, raw_cur_2, currency_1_to_graph, currency_2_to_graph,
                         raw_pair_slope_1, raw_pair_slope_2, raw_ind_slope_1, raw_ind_slope_2, slope_windows,
                         debug_row, raw_running_sum_1, raw_running_sum_2, raw_running_sum_3,
                         raw_diff_by_mean))

    # Graph Processes
    graph_indicator_and_pair = Process(target=graph_double,
                                       args=(data_points, raw_pair, raw_diff,
                                             'Indicator and Pair', 'Pair', 'Indicator', debug_row, False, False))
    graph_indicator_and_pair_a = Process(target=one_on_top,
                                       args=(data_points, raw_pair, raw_diff, raw_mean_1, raw_diff_by_mean,
                                             'Indicator and Pair', 'Pair', 'Indicator', debug_row, False, False))
    # graph_indicator_and_pair = Process(target=graph_double_zeroed,
    #                                    args=(data_points, raw_pair, raw_diff, raw_mean_2,
    #                                          'Indicator and Pair', 'Pair', 'Indicator', debug_row, False, False))
    # graph_running_mean = Process(target = graph_running_sum,
    #                              args=(data_points, raw_running_sum_1, raw_running_sum_2,
    #                              raw_running_sum_3, raw_pair, debug_row, False, False))
    graph_running_mean = Process(target=graph_running_sum,
                                 args=(data_points, raw_running_sum_1, raw_running_sum_2, raw_running_sum_3, raw_pair,
                                       debug_row, False, False))
    graph_slopes_1 = Process(target=graph_double,
                             args=(data_points, raw_pair_slope_1, raw_ind_slope_1,
                                   'Slopes @ {}'.format(str(slope_windows[0])),
                                   'Pair', 'Indicator', debug_row, False, True))
    graph_slopes_2 = Process(target=graph_double,
                             args=(data_points, raw_pair_slope_2, raw_ind_slope_2,
                                   'Slopes @ {}'.format(str(slope_windows[1])),
                                   'Pair', 'Indicator', debug_row, False, True))
    graph_covariance_process = Process(target=graph_double,
                                       args=(data_points, raw_cov_1, raw_cov_2,
                                            'Covariance', str(cov_windows[0]), str(cov_windows[1]),
                                             debug_row, False, True))
    pips_graph = Process(target=graph_pips, args=(data_points, raw_pair, debug_row))

    # # Start Processes
    calc.start()
    # graph_running_mean.start()
    graph_indicator_and_pair.start()
    graph_indicator_and_pair_a.start()
    # graph_slopes_1.start()
    # graph_slopes_2.start()
    # graph_covariance_process.start()
    # pips_graph.start()

    # Stream in main process
    price_stream(pairs_index, bids_or_asks, q_streaming, oanda_api, oanda_account, configs, debug_row)

    # Join Processes
    calc.join()
    # graph_running_mean.join()
    graph_indicator_and_pair.join()
    # graph_slopes_1.join()
    # graph_slopes_2.join()
    # graph_covariance_process.join()
    # pips_graph.join()





















    # graph_pair_slopes = Process(target=graph_slopes, args=(data_points, raw_pair_slope_1, raw_pair_slope_2,
    #                                                     raw_pair_slope_2, slope_windows))
    # graph_diff_slopes = Process(target=graph_slopes, args=(data_points, raw_ind_slope_1, raw_ind_slope_2,
    #                                                     raw_ind_slope_2, slope_windows))
    # graph_cov_process = Process(target=graph_cov,
    #                             args=(data_points, raw_cov_1, raw_cov_2, raw_cov_3, cov_windows))
    # graph_ind_w_means = Process(target=graph_indicator_with_means,
    #                             args=(data_points, raw_diff, 'Difference Indicator with Means',
    #                                   raw_mean_1, raw_mean_2, raw_mean_3, mean_windows))
    # graph_double_zero_process = Process(target=zero_scaled_graph_double,
    #                                     args=(data_points, raw_diff, raw_pair, raw_mean_1, raw_mean_2,
    #                                           raw_mean_3, mean_windows))
    # graph_currencies_process = Process(target=graph_currencies,
    #                                args=(data_points, raw_cur_1, raw_cur_2))
    # graph_pair_process = Process(target=graph_single, args=(data_points, raw_pair, 'PAIR', mean_windows))
    # graph_diff_process = Process(target=graph_single, args=(data_points, raw_pair, 'DIFF', mean_windows))
    # graph_cov_process.start()
    # graph_pair_slopes.start()
    # graph_diff_slopes.start()
    # graph_double_zero_process.start()
    # graph_currencies_process.start()
    # graph_pair_process.start()
    # graph_ind_w_means.start()













'''


# For weekend work:
from libraries.database import database_retrieve
a = database_retrieve('/Users/user/Desktop/diff/streaming.db', 'select * from pairs')
for i in range(15000):
    q_streaming.put([a[i][0], list(a[i][2:])])

def graph_currencies(data_points, raw_cur_1, raw_cur_2):

    """
    DOUBLE AXIS GRAPH.
    """
    global fps, lastTime, ptr, p, viewbox, last_plotted, missed, latest
    global cur_1_curve, cur_2_curve, cur_1_data, cur_2_data, fps, lastTime, number_missed
    # Instantiate plot
    win = pg.GraphicsWindow(title='CURRENCIES')
    win.resize(1000, 600)
    win.setWindowTitle('CURRENCIES')
    p = win.addPlot(title="CURRENCIES")

    # Plot Curves
    cur_1_curve = p.plot(pen=pg.mkPen(color='#60F073', width=1))
    cur_1_data = np.frombuffer(raw_cur_1).reshape(data_points + 1)
    cur_2_curve = pg.PlotCurveItem(pen=pg.mkPen(color='#F06084', width=1))
    cur_2_data = np.frombuffer(raw_cur_2).reshape(data_points + 1)

    # Add Second Axis.
    viewbox = pg.ViewBox()
    p.scene().addItem(viewbox)
    p.getAxis('right').linkToView(viewbox)
    viewbox.setXLink(p)
    viewbox.addItem(cur_2_curve)

    # Axis Controls
    p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=1))
    p.getAxis('left').setPen(pg.mkPen(color='#60F073', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#F06084', width=1))
    p.setLabel('left', 'EUR', color='#60F073', **{'font-size': '12pt'})
    p.setLabel('right', 'USD', color='#F06084', **{'font-size': '12pt'})
    p.showAxis('right')

    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging and frames per second
    missed = 0
    last_plotted = 0
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        # Don't update plot without new data
        global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows
        global cur_1_curve, cur_2_curve, cur_1_data, cur_2_data
        latest = int(cur_1_data[-1])
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)
            # Update Plot
            cur_1_curve.setData(cur_1_data[data_points - (latest - 300): -1])
            cur_2_curve.setData(cur_2_data[data_points - (latest - 300): -1])

            # Frames per Second
            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format('both', latest, datetime.now()))
                print('{} Missed   {}:     {}'.format('both', latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how fast data moves when this is in latest > last loop


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()



def covariance(q_covariance):
    pass

def graph_single(data_points, raw_line, title, mean_windows):

    global curve, data, ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted, plot_wait
    # Instantiate plot
    win = pg.GraphicsWindow(title=title)
    win.resize(1000, 600)
    win.setWindowTitle(title)
    p = win.addPlot(title="Updating plot")
    # Plot Curves

    curve = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
    data = np.frombuffer(raw_line).reshape(data_points + 1)
    # Debugging
    missed = 0
    last_plotted = 0
    latest = data[-1]
    ptr = 0
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        # Don't update plot without new data
        global curve, data, ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows, plot_wait
        latest = int(data[-1])
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)
            # Update Plot
            curve.setData(data[data_points - (latest-300): -1])
            # Frames per Second
            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format(title, latest, datetime.now()))
                print('{} Missed   {}:     {}'.format(title, latest, missed / latest))
            # if ptr % 1000 == 0:
            #     print('single updates: {}'.format(ptr))  # can tell how fast data moves when this is in latest > last loop

    # Start graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def graph_double(data_points, raw_diff, raw_pair, raw_mean_1, raw_mean_2, raw_mean_3, mean_windows):

    """
    DOUBLE AXIS GRAPH.
    """
    global curve_mean_1, curve_mean_2, curve_mean_3, mean_curve_data_1, mean_curve_data_2, mean_curve_data_3
    global diff_curve, pair_curve, fps, lastTime, ptr, p, viewbox, last_plotted, missed, latest, point_curve
    global curve, data, ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted, plot_wait
    # Instantiate plot
    win = pg.GraphicsWindow(title='Both')
    win.resize(1000, 600)
    win.setWindowTitle('both')
    p = win.addPlot(title="Updating plot")

    # Plot Curves
    diff_curve = p.plot(pen=pg.mkPen(color='#ED15DF', width=1))
    diff_data = np.frombuffer(raw_diff).reshape(data_points + 1)
    pair_curve = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=1))
    pair_data = np.frombuffer(raw_pair).reshape(data_points + 1)
    point_curve = p.plot(symbolPen='w')
    # curve_mean_1 = p.plot(pen=pg.mkPen(color='#007BF5', width=3))
    # mean_curve_data_1 = np.frombuffer(raw_mean_1).reshape(data_points + 1)
    # curve_mean_2 = p.plot(pen=pg.mkPen(color='#7EB3E8', width=1))
    # mean_curve_data_2 = np.frombuffer(raw_mean_2).reshape(data_points + 1)
    # curve_mean_3 = p.plot(pen=pg.mkPen(color='#C7DAED', width=1))
    # mean_curve_data_3 = np.frombuffer(raw_mean_3).reshape(data_points + 1)

    # Try to add a second graph - don't know what any of this does.
    viewbox = pg.ViewBox()
    viewbox.setXLink(p)
    p.scene().addItem(viewbox)
    p.getAxis('right').linkToView(viewbox)
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#ED15DF', width=2))
    p.getAxis('right').setPen(pg.mkPen(color='#5AED15', width=2))
    # viewbox.setYRange(-10, 10)
    # pair_curve = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=3))

    viewbox.addItem(pair_curve)

    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging
    missed = 0
    last_plotted = 0
    ptr = 0
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        # Don't update plot without new data
        global point_curve, curve, data, ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows, plot_wait
        latest = int(diff_data[-1])
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)
            # Update Plot
            pair_curve.setData(pair_data[data_points - (latest - 300): -1])
            diff_curve.setData(diff_data[data_points - (latest - 300): -1])
            # point_curve.setData(x=[(diff_data[data_points - (latest - 300): -1]).shape[0]],
            #                    y=[diff_data[-2]])
            # curve_mean_1.setData(mean_curve_data_1[data_points - (latest-300): -1])
            # curve_mean_2.setData(mean_curve_data_2[data_points - (latest-300): -1])
            # curve_mean_3.setData(mean_curve_data_3[data_points - (latest-300): -1])
            # Frames per Second

            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format('both', latest, datetime.now()))
                print('{} Missed   {}:     {}'.format('both', latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how fast data moves when this is in latest > last loop


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def zero_scaled_graph_double(data_points, raw_diff, raw_pair, raw_mean_1, raw_mean_2, raw_mean_3, mean_windows):

    """
    DOUBLE AXIS GRAPH.
    """
    global curve_mean_1, curve_mean_2, curve_mean_3, mean_curve_data_1, mean_curve_data_2, mean_curve_data_3
    global diff_curve, pair_curve, fps, lastTime, ptr, p, viewbox, last_plotted, missed, latest
    global curve, data, ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted, plot_wait
    # Instantiate plot
    win = pg.GraphicsWindow(title='Both')
    win.resize(1000, 600)
    # win.setWindowTitle('both')
    p = win.addPlot(title="Updating plot")


    p.setYRange(-6, 6)

    # Plot Curves
    diff_curve = p.plot(pen=pg.mkPen(color='#ED15DF', width=1))
    diff_data = np.frombuffer(raw_diff).reshape(data_points + 1)
    pair_curve = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=1))
    pair_data = np.frombuffer(raw_pair).reshape(data_points + 1)
    # curve_mean_1 = p.plot(pen=pg.mkPen(color='#007BF5', width=3))
    # mean_curve_data_1 = np.frombuffer(raw_mean_1).reshape(data_points + 1)
    # curve_mean_2 = p.plot(pen=pg.mkPen(color='#7EB3E8', width=1))
    # mean_curve_data_2 = np.frombuffer(raw_mean_2).reshape(data_points + 1)
    # curve_mean_3 = p.plot(pen=pg.mkPen(color='#C7DAED', width=1))
    # mean_curve_data_3 = np.frombuffer(raw_mean_3).reshape(data_points + 1)

    # Try to add a second graph - don't know what any of this does.
    viewbox = pg.ViewBox()
    p.scene().addItem(viewbox)
    viewbox.addItem(pair_curve)

    # Set Axes
    p.showAxis('right')
    p.getAxis('right').linkToView(viewbox)
    p.getAxis('left').setPen(pg.mkPen(color='#ED15DF', width=2))
    p.getAxis('right').setPen(pg.mkPen(color='#5AED15', width=2))
    # viewbox.setYLink(p)
    # viewbox.scaleBy(center=0)
    # p.enableAutoRange('y', True)
    # viewsetAutoPan(x=None, y=None)
    # p.enableAutoRange('y', True)
    p.setYRange(-7, 7)
    viewbox.setXLink(p)
    viewbox.setYLink(p)
    viewbox.enableAutoRange('y', True)



    # I have no idea - maybe try to delete
    def updateViews():
        global viewbox
        viewbox.setGeometry(p.getViewBox().sceneBoundingRect())
        viewbox.linkedViewChanged(p.getViewBox(), viewbox.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    # Debugging
    missed = 0
    last_plotted = 0
    ptr = 0
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        # Don't update plot without new data
        global curve, data, ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows, plot_wait
        latest = int(diff_data[-1])
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)
            # Update Plot
            # pair_curve.setData(pair_data[data_points - (latest - 300): -1])
            # diff_curve.setData(diff_data[data_points - (latest - 300): -1])
            # curve_mean_1.setData(mean_curve_data_1[data_points - (latest-300): -1])
            # curve_mean_2.setData(mean_curve_data_2[data_points - (latest-300): -1])
            # curve_mean_3.setData(mean_curve_data_3[data_points - (latest-300): -1])

            # Just to mess around with scaling for a bit
            pair_norm = pair_data[data_points - (latest - 300): -1] - pair_data[-2]
            pair_norm /= pair_data[data_points - (latest - 300): -1].std()
            diff_norm = diff_data[data_points - (latest - 300): -1] - diff_data[-2]
            diff_norm /= diff_data[data_points - (latest - 300): -1].std()
            pair_curve.setData(pair_norm)
            diff_curve.setData(diff_norm)


            # Frames per Second
            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format('both', latest, datetime.now()))
                print('{} Missed   {}:     {}'.format('both', latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how fast data moves when this is in latest > last loop


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def graph_indicator_with_means(data_points, raw_line, title, raw_mean_1, raw_mean_2, raw_mean_3, mean_windows):

    # Instantiate plot
    global curve, data, plot_wait
    global curve_mean_1, curve_mean_2, curve_mean_3, mean_curve_data_1, mean_curve_data_2, mean_curve_data_3
    global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted
    win = pg.GraphicsWindow(title=title)
    win.resize(1000, 600)
    win.setWindowTitle(title)
    p = win.addPlot(title="Updating plot")
    # Plot Curves
    curve = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
    data = np.frombuffer(raw_line).reshape(data_points + 1)
    curve_mean_1 = p.plot(pen=pg.mkPen(color='#007BF5', width=2))
    mean_curve_data_1 = np.frombuffer(raw_mean_1).reshape(data_points + 1)
    curve_mean_2 = p.plot(pen=pg.mkPen(color='#7EB3E8', width=2))
    mean_curve_data_2 = np.frombuffer(raw_mean_2).reshape(data_points + 1)
    curve_mean_3 = p.plot(pen=pg.mkPen(color='#C7DAED', width=2))
    mean_curve_data_3 = np.frombuffer(raw_mean_3).reshape(data_points + 1)
    # Debugging
    missed = 0
    last_plotted = 0
    latest = data[-1]
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    # Update Plot
    def update():
        # Global variables
        global curve, data, mean_windows
        global curve_mean_1, curve_mean_2, curve_mean_3, mean_curve_data_1, mean_curve_data_2, mean_curve_data_3
        global lastTime, last_plotted, missed, latest, number_missed, ptr, p, fps
        # Don't update plot without new data
        latest = int(data[-1])



        if latest > last_plotted:
            if latest % 10 == 0:
                last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)
            # Update Plot
            curve.setData(data[data_points - (latest-300): -1])
            curve_mean_1.setData(mean_curve_data_1[data_points - (latest-300): -1])
            curve_mean_2.setData(mean_curve_data_2[data_points - (latest-300): -1])
            curve_mean_3.setData(mean_curve_data_3[data_points - (latest-300): -1])
            # Frames per Second
            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format(title, latest, datetime.now()))
                print('{} Missed   {}:     {}'.format(title, latest, missed / latest))


    # Start graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def graph_simple():

    global curve, data, ptr, p

    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('SIMPLE')

    p = win.addPlot(title="Updating plot")
    curve = p.plot(pen='y')
    data = np.random.normal(size=(10, 1000))
    ptr = 0

    with open(configs_file) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    data = np.ones(1000) * configs['pips_loss']
    latest_pip = 0


    def update():
        global curve, data, ptr, p

        try:
            with open(configs_file) as f:
                configs = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)
        pips = configs['pips_loss']
        if type(pips) == type(3) and pips != latest_pip:
            data = np.ones(1000) * configs['pips_loss']


        curve.setData(data)
        if ptr == 0:
            p.enableAutoRange('y', True)
        ptr += 1
        if ptr % 1000 == 0:
            print('simple: {}'.format(ptr))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()




def graph_simple_with_read(raw_line):

    # Instantiate Plot
    global curve, data, ptr, p
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('SIMPLE WITH READ')
    p = win.addPlot(title="Updating plot")
    # PLot Data
    curve = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
    data = np.frombuffer(raw_line).reshape(data_points + 1)
    # Debugging
    ptr = 0

    def update():
        # Draw Curve
        global curve, data, ptr, p
        curve.setData(data[-1001:-1])
        if ptr == 0:
            p.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
        # Debugging
        if ptr % 1000 == 0:
            print('simple_W_read updates: {}'.format(ptr))
        ptr += 1

    # Start Graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def graph_cov(data_points, raw_cov_1, raw_cov_2, raw_cov_3, cov_windows):

    global data1, data2, data3, curve1, curve2, curve3
    global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted
    # PYQT params
    win = pg.GraphicsWindow(title='COVARIANCE')
    win.resize(1000, 600)
    pg.setConfigOptions(antialias=True)
    # Instantiate plot
    p = win.addPlot(title='COVARIANCE')
    p.setYRange(-1, 1)
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#FADEC5', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#675DC2', width=1))
    p.setLabel('left', str(cov_windows[0]), color='#60F073', **{'font-size': '12pt'})
    p.setLabel('right', str(cov_windows[1]), color='#F06084', **{'font-size': '12pt'})
    p.showGrid(y=True)

    # Plot Curves
    curve1 = p.plot(pen=pg.mkPen(color='#FADEC5', width=1))
    data1 = np.frombuffer(raw_cov_1).reshape(data_points + 1)
    curve2 = p.plot(pen=pg.mkPen(color='#675DC2', width=1))
    data2 = np.frombuffer(raw_cov_2).reshape(data_points + 1)
    # curve3 = p.plot(pen=pg.mkPen(color='#42B528', width=2))
    # data3 = np.frombuffer(raw_cov_3).reshape(data_points + 1)

    # Debugging
    missed = 0
    last_plotted = 0
    latest = data1[-1]
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():

        # Don't update plot without new data
        global data1, data2, data3, curve1, curve2, curve3
        global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows
        latest = int(data1[-1])
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)

            # Update Plot
            curve1.setData(data1[data_points - (latest-300): -1])
            curve2.setData(data2[data_points - (latest-300): -1])
            # curve3.setData(data3[data_points - (latest-300): -1])
            # Frames per Second
            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format('COV', latest, datetime.now()))
                print('{} Missed   {}:     {}'.format('COV', latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how  data moves when this is in latest > last loop

    # Start graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def graph_slopes(data_points, raw_cov_1, raw_cov_2, raw_cov_3, cov_windows):

    global data1, data2, data3, curve1, curve2, curve3
    global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted
    # PYQT params
    win = pg.GraphicsWindow(title='PAIR_SLOPE')
    win.resize(1000, 600)
    # win.setWindowTitle('COVARIANCE')
    pg.setConfigOptions(antialias=True)
    # Instantiate plot
    p = win.addPlot(title='PAIR_SLOPE')
    p.showAxis('right')
    p.getAxis('left').setPen(pg.mkPen(color='#FADEC5', width=1))
    p.getAxis('right').setPen(pg.mkPen(color='#675DC2', width=1))
    p.setLabel('left', str(cov_windows[0]), color='#60F073', **{'font-size': '12pt'})
    p.setLabel('right', str(cov_windows[1]), color='#F06084', **{'font-size': '12pt'})

    # Plot Curves
    curve1 = p.plot(pen=pg.mkPen(color='#FADEC5', width=1))
    data1 = np.frombuffer(raw_cov_1).reshape(data_points + 1)
    curve2 = p.plot(pen=pg.mkPen(color='#675DC2', width=1))
    data2 = np.frombuffer(raw_cov_2).reshape(data_points + 1)
    # curve3 = p.plot(pen=pg.mkPen(color='#42B528', width=2))
    # data3 = np.frombuffer(raw_cov_3).reshape(data_points + 1)

    # Debugging
    missed = 0
    last_plotted = 0
    latest = data1[-1]
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():

        # Don't update plot without new data
        global data1, data2, data3, curve1, curve2, curve3
        global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows
        latest = int(data1[-1])
        if latest > last_plotted:

            # print('graph: {}'.format(latest, data1[-2]))

            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    # print(latest, last_plotted, number_missed, missed)

            # Update Plot
            curve1.setData(data1[data_points - (latest-300): -1])
            curve2.setData(data2[data_points - (latest-300): -1])
            # curve3.setData(data3[data_points - (latest-300): -1])
            # Frames per Second
            ptr += 1
            now = pyqt_time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                fps = fps * (1 - s) + (1.0 / dt) * s
            p.setTitle('%0.2f fps' % fps)
            # Debugging
            if latest % 100 == 0:
                print('{} Complete {}:     {}'.format('COV', latest, datetime.now()))
                print('{} Missed   {}:     {}'.format('COV', latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how  data moves when this is in latest > last loop

    # Start graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()
'''






