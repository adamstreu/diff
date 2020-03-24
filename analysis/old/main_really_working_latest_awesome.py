if True:
    import pandas as pd
    import numpy as np
    import time
    import sys
    import copy
    import datetime
    from multiprocessing import Process, Queue, RawArray
    import oandapyV20
    import oandapyV20.endpoints.pricing as pricing
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtCore
    from pyqtgraph.ptime import time as pyqt_time
    sys.path.insert(1, '/Users/user/Desktop/diff')
    from configs import oanda_api, oanda_account  # Oanda Account Parameters


"""

SOLVED!!!!!!!

        Currently plotting 30000 point on one line within .2 seconds of arriving 
            ( ~ .5 seconds later than sent )
            
GOAL:

    Tomorrow, drink some tea an watch the graph.
    Finish the few odds an end and prepare trading module.
    Maybe learn about graphing library with time
    
Next:

    
    
    get controller working
        number to submit.
        fucking pandas / pygame thing.
        
    plot where breakeven would be on bid / ask    
    
    Have to get the fucking plot behavior friencly - fuck it just wia to plot till i have my line.....
        woindoes wont be that big anyway
            
            WHEN CALUCLATING AVERAGE - ONLY USE WHAT NUMBERS ARAENT ZERO.  (OBVIOUSLY)
    
    
    mean array calculation 
    
    covariance working
    
    can we add back in a second axis - were we doing it ok or does it really slow us down...check again 
        tomorrow - but after getting up early, ollowing a graph and correcting any errors
            so that i can spend a few hours just relaxing ( get comfortable...seriously) and watching graph
    
    maybe a few more cov lines ( diff winows) ( probably small is all i need though ) 
        
Later:
    
    Can we try subplots ?....

    graph - keep with x after adjusting y
    Any pyqt tricks to speee up graph? 
        (downsampling ? ) 
    Plot break even bid ask lines
    coloring, etc.  dont get obsessed
    
    get midi or graph  or hot buttons working - one currency
    Make sure I hevn;t slowed anything down adding the second line / axis:
        double check everythin gaboiut it ( late im tiresd_ 
    How do extra currencies hold up
    maybe a dot at current level on pairs and difference

Fuckaroos:

    I do not understand sometimes while the simple with read refuses to graph.
        Must move on.


"""


if True:
    # Parameters
    bids_or_asks = 'bids'
    data_points = 5000
    global mean_windows
    mean_windows = [10, 50, 1000]
    cov_windows = [10, 50, 300]
    rolling_cov_window = 25
    currency_to_graph = 18
    # Queues and Arrays
    q_streaming = Queue()

    # Raw Arrays
    raw_diff = RawArray('d', data_points + 1)
    raw_pair = RawArray('d', data_points + 1)
    raw_cov_1 = RawArray('d', data_points + 1)
    raw_cov_2 = RawArray('d', data_points + 1)
    raw_cov_3 = RawArray('d', data_points + 1)
    raw_mean_1 = RawArray('d', data_points + 1)
    raw_mean_2 = RawArray('d', data_points + 1)
    raw_mean_3 = RawArray('d', data_points + 1)

    # Pairs
    pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD',
                   'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD',
                   'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP',
                   'EUR_HKD', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
                   'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD',
                   'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD',
                   'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD',
                   'USD_JPY']
    trading_pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'AUD_USD', 'CAD_CHF',
                           'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 'EUR_USD',
                           'NZD_CAD', 'NZD_CHF', 'NZD_USD', 'USD_CHF']
    # Currencies
    currencies_index = list(set('_'.join(pairs_index).split('_')))
    currencies_index.sort()


def price_stream(pairs_index, bids_or_asks, q, oanda_api, oanda_account):
    """
    Stream Prices from Oanda for pairs list.
    Load data into a q if passed as argument
    No sql, Que only
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
                    _pairs[pairs_index.index(ticks['instrument'])] = float(ticks[bids_or_asks][0]['price'])
                    q.put([row, _pairs])
                    # Debug
                    if row % 100 == 0:
                        print('Stream Recieved    {}:     {}'.format(row, ticks['time']))
                        print('Stream Queued      {}:     {}'.format(row, datetime.datetime.now()))
                    row += 1
                except Exception as e:
                    print('Stream | Calculation exception: {}'.format(e))


def calculations(pairs_index, currencies_index, q, data_points, currency_to_graph,
                 raw_diff, raw_pair,
                 raw_mean_1, raw_mean_2, raw_mean_3, mean_windows,
                 raw_cov_1, raw_cov_2, raw_cov_3, cov_windows):

    # Read Raw Array
    R_diff = np.frombuffer(raw_diff, dtype=np.float64).reshape(data_points + 1)
    R_pair = np.frombuffer(raw_pair, dtype=np.float64).reshape(data_points + 1)
    R_mean_1 = np.frombuffer(raw_mean_1, dtype=np.float64).reshape(data_points + 1)
    R_mean_2 = np.frombuffer(raw_mean_2, dtype=np.float64).reshape(data_points + 1)
    R_mean_3 = np.frombuffer(raw_mean_3, dtype=np.float64).reshape(data_points + 1)
    R_cov_1 = np.frombuffer(raw_cov_1, dtype=np.float64).reshape(data_points + 1)
    R_cov_2 = np.frombuffer(raw_cov_2, dtype=np.float64).reshape(data_points + 1)
    R_cov_3 = np.frombuffer(raw_cov_3, dtype=np.float64).reshape(data_points + 1)

    # Required Numpy Arrays
    currencies = np.ones((len(currencies_index), data_points))
    pairs = np.ones((len(pairs_index), data_points))
    differences = copy.deepcopy(pairs)
    calculated = copy.deepcopy(pairs)
    mean_lines = np.zeros((len(mean_windows), len(pairs_index), data_points))
    covariances = copy.deepcopy(mean_lines)

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

            # Gather most recent data from qu stream
            _pairs = q.get()
            cal_retrieve = datetime.datetime.now()
            row = _pairs[0]
            _pairs = _pairs[1]

            # Roll arrays
            pairs = np.roll(pairs, -1)
            differences = np.roll(differences, -1)
            calculated = np.roll(calculated, -1)
            mean_lines = np.roll(mean_lines, -1)
            covariances = np.roll(covariances, -1)
            currencies = np.roll(currencies, -1)
            covariances = np.roll(covariances, -1)

            # Update Pairs
            pairs[:, -1] = _pairs
            # Calculate newest and update currencies
            a = np.tile(_pairs, (len(currencies_index), 1))
            _currencies = 1 / ((a * given).sum(1) + ((1 / a) * inverse).sum(1) + 1)
            currencies[:, -1] = _currencies
            # Calculate calculated and update calculated (yikes)
            a = np.tile(_currencies, (len(pairs_index), 1))
            _calculated = (a * inverse.T).sum(1) / (a * given.T).sum(1)
            calculated[:, -1] = _calculated
            # Calculate difference and update difference
            _differences = _pairs - _calculated  # Should go in same directions now ( I suppose )
            differences[:, -1] = _differences
            # Mean lines
            # for i in range(len(mean_windows)):   # Could do with array calculation
            #     if row <= max(mean_windows) + 10:
            #             mean_lines[i, :, -1] = _differences
            #     else:
            #         mean_lines[i, :, -1] = differences[:, -(mean_windows[i]):].mean(1)
            for i in range(len(mean_windows)):  # Could do with array calculation
                mean_lines[i, :, -1] = differences[:, -(mean_windows[i]):].mean(1)
            # Correlation Coefficient
            # covariance[:, -1] = [np.corrcoef(differences[i], pairs[i])[0][1] for i in range(len(pairs_index))]
            for j in range(len(cov_windows)):  # Could do with array calculation
                # print(j)
                _covariances = [np.corrcoef(differences[i, -(cov_windows[j]):], pairs[i, -(cov_windows[j]):])[0][1] for i in range(len(pairs_index))]
                # print(len(_covariances))
                # print(_covariances)
                # print(pairs[i, -(cov_windows[j]):].shape)
                # print(covariances.shape)
                covariances[j, :, -1] = np.array(_covariances)
                # print(covariances.shape)





            # Timestamp - Calculations complete
            calc_complete = datetime.datetime.now()

            # Copy data to shared arrays
            np.copyto(R_diff[: -1], differences[currency_to_graph])
            np.copyto(R_pair[: -1], pairs[currency_to_graph])
            np.copyto(R_mean_1[: -1], mean_lines[0, currency_to_graph, :])
            np.copyto(R_mean_2[: -1], mean_lines[1, currency_to_graph, :])
            np.copyto(R_mean_3[: -1], mean_lines[2, currency_to_graph, :])
            np.copyto(R_cov_1[: -1], covariances[0, currency_to_graph, :])
            np.copyto(R_cov_2[: -1], covariances[1, currency_to_graph, :])
            np.copyto(R_cov_3[: -1], covariances[2, currency_to_graph, :])
            np.copyto(R_diff[-1:], row)
            np.copyto(R_pair[-1:], row)
            np.copyto(R_mean_1[-1:], row)
            np.copyto(R_mean_2[-1:], row)
            np.copyto(R_mean_3[-1:], row)
            np.copyto(R_cov_1[-1:], row)
            np.copyto(R_cov_2[-1:], row)
            np.copyto(R_cov_3[-1:], row)
            array_copy_time = datetime.datetime.now()

            # Debug
            if row % 100 == 0:
                # print('Stream Recieved    {}:     {}'.format(row, cal_retrieve))
                # print('Calc Load Complete {}:     {}'.format(row, datetime.datetime.now()))
                # print('Calc time          {}:     {}'.format(row, calc_complete - cal_retrieve))
                # print('Load Arrays        {}:     {}'.format(row, array_copy_time - calc_complete))
                # print('Last pair point    {}:     {}'.format(R_diff[-1], R_diff[-2]))
                # print('Calc def compl     {}:     {}'.format(R_diff[-1], datetime.datetime.now()))
                print(R_cov_1[-4:-1])


def graph_double(data_points, raw_diff, raw_pair, raw_cov, rolling_cov_window, raw_mean_1, mean_windows):

    """
    DOUBLE AXIS GRAPH.
    """

    global diff_curve, pair_curve, fps, lastTime, ptr, p, p2, last_plotted, missed, latest


    # Plot inst for mulitple try
    p = pg.plot()
    p.showAxis('right')
    p.getAxis('right').setPen(pg.mkPen(color='#E3E3E3', width=1))
    pg.setConfigOptions(antialias=True)  # Make sure this isn't bad ( don't know it )

    diff_curve = p.plot(pen=pg.mkPen(color='#ED15DF', width=2))
    diff_data = np.frombuffer(raw_diff).reshape(data_points + 1)
    pair_data = np.frombuffer(raw_pair).reshape(data_points + 1)

    # Try to add a second graph - don't know what any of this does.
    p2 = pg.ViewBox()
    p.scene().addItem(p2)
    p.getAxis('right').linkToView(p2)
    p2.setXLink(p)
    # p2.setYRange(-10, 10)
    pair_curve = pg.PlotCurveItem(pen=pg.mkPen(color='#5AED15', width=2))
    p2.addItem(pair_curve)

    def updateViews():
        global p2
        p2.setGeometry(p.getViewBox().sceneBoundingRect())
        p2.linkedViewChanged(p.getViewBox(), p2.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)


    missed = 0
    last_plotted = 0

    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():

        global diff_curve, here, fps, lastTime, ptr, p, p2, last_plotted, missed, latest

        latest = diff_data[-1]
        if latest > last_plotted: # only update on new data
             if latest - last_plotted > 1:
                 missed += latest - last_plotted
             last_plotted = latest
             pair_curve.setData(pair_data[pair_data != 1][100:-1])  # 100 on to remove wonky startup values
             diff_curve.setData(diff_data[diff_data != 1][100:-1])  # 100 on to remove wonky startup values
             if latest % 100 == 0:
                 print('plot On            {}:     {}'.format(latest, datetime.datetime.now()))
                 print('plot Number        {}:     {}'.format(latest, diff_data[-2]))
                 print('Percent missed     {}:     {}'.format(latest, missed / latest))
                 print()

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


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


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
    plot_wait = max(mean_windows) + 50
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
        latest = data[-1]
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    print(latest, last_plotted, number_missed, missed)
            # Update Plot
            curve.setData(data[data != 0][plot_wait:-1])
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
                print('{} Complete {}:     {}'.format(title, latest, datetime.datetime.now()))
                print('{} Missed   {}:     {}'.format(title, latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how fast data moves when this is in latest > last loop

    # Start graph
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
    plot_wait = max(mean_windows) + 50
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
        global curve, data, plot_wait
        global curve_mean_1, curve_mean_2, curve_mean_3, mean_curve_data_1, mean_curve_data_2, mean_curve_data_3
        global lastTime, last_plotted, missed, latest, number_missed, ptr, p, fps
        # Don't update plot without new data
        latest = data[-1]   # Why doesn't this see to work like data above ( must call it itself )  ? ?? ?
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    print(latest, last_plotted, number_missed, missed)
            # Update Plot
            curve.setData(data[data != 1][plot_wait:-1])
            curve_mean_1.setData(mean_curve_data_1[mean_curve_data_1 != 0][plot_wait:-1])
            curve_mean_2.setData(mean_curve_data_2[mean_curve_data_2 != 0][plot_wait:-1])
            curve_mean_3.setData(mean_curve_data_3[mean_curve_data_3 != 0][plot_wait:-1])
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
                print('{} Complete {}:     {}'.format(title, latest, datetime.datetime.now()))
                print('{} Missed   {}:     {}'.format(title, latest, missed / latest))

    # Start graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()


def graph_simple(): # This one is so fast

    global curve, data, ptr, p

    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('SIMPLE')

    p = win.addPlot(title="Updating plot")
    curve = p.plot(pen='y')
    data = np.random.normal(size=(10, 1000))
    ptr = 0

    def update():
        global curve, data, ptr, p
        curve.setData(data[ptr % 10])
        if ptr == 0:
            p.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
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


def graph_cov(data_points, raw_cov_1, raw_cov_2, raw_cov_3, mean_windows):

    global data1, data2, data3, curve1, curve2, curve3
    global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, last_plotted, plot_wait
    # PYQT params
    win = pg.GraphicsWindow(title='COVARIANCE')
    win.resize(1000, 600)
    win.setWindowTitle('COVARIANCE')
    pg.setConfigOptions(antialias=True)
    # Instantiate plot
    p = win.addPlot(title='COVARIANCE')
    p.showGrid(x=True, y=True)
    p.setYRange(-1, 1)
    # Plot Curves
    curve1 = p.plot(pen=pg.mkPen(color='#FADEC5', width=2))
    data1 = np.frombuffer(raw_cov_1).reshape(data_points + 1)
    # curve2 = p.plot(pen=pg.mkPen(color='#675DC2', width=2))
    # data2 = np.frombuffer(raw_cov_2).reshape(data_points + 1)
    # curve3 = p.plot(pen=pg.mkPen(color='#42B528', width=2))
    # data3 = np.frombuffer(raw_cov_3).reshape(data_points + 1)
    plot_wait = max(mean_windows) + 50
    # Debugging
    missed = 0
    last_plotted = 0
    latest = data1[-1]
    ptr = 0
    # frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():

        # Don't update plot without new data
        global data1, data2, data3, curve1, curve2, curve3
        global ptr, p, fps, lastTime, last_plotted, missed, latest, number_missed, mean_windows, plot_wait
        latest = data1[-1]
        if latest > last_plotted:
            last_plotted = latest
            # Debugging
            if latest - last_plotted > 1:
                    number_missed = (latest - last_plotted) - 1
                    missed += number_missed
                    print(latest, last_plotted, number_missed, missed)

            # Update Plot
            curve1.setData(data1[data1 != 0][plot_wait:-1])
            #curve2.setData(data2[data2 != 0][plot_wait:-1])
            # curve2.setData(data3[data3 != 0][plot_wait:-1])
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
                print('{} Complete {}:     {}'.format('COV', latest, datetime.datetime.now()))
                print('{} Missed   {}:     {}'.format('COV', latest, missed / latest))
            if ptr % 1000 == 0:
                print('single updates: {}'.format(ptr))  # can tell how  data moves when this is in latest > last loop
    # Start graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    QtGui.QApplication.instance().exec_()



if __name__ == '__main__':

    #  Streaming and Calculations
    stream = Process(target=price_stream,
                     args=(pairs_index, bids_or_asks, q_streaming, oanda_api, oanda_account))
    calc = Process(target=calculations,
                   args=(pairs_index, currencies_index, q_streaming, data_points, 18,
                         raw_diff, raw_pair,
                         raw_mean_1, raw_mean_2, raw_mean_3, mean_windows,
                         raw_cov_1, raw_cov_2, raw_cov_3, cov_windows))

    # Graphs
    graph_pair_process = Process(target=graph_single, args=(data_points, raw_pair, 'PAIR', mean_windows))
    graph_ind_w_means = Process(target=graph_indicator_with_means,
                                args=(data_points, raw_diff, 'Difference Indicator with Means',
                                      raw_mean_1, raw_mean_2, raw_mean_3, mean_windows))
    graph_cov_process = Process(target=graph_cov,
                                args=(data_points, raw_cov_1, raw_cov_2, raw_cov_3, mean_windows))
    # graph_diff_process = Process(target=graph_single, args=(data_points, raw_diff, 'DIFFERENCE'))
    # graph_simple_process = Process(target=graph_simple, args=())
    # graph_simple_read_process = Process(target=graph_simple_with_read, args=(raw_pair, ))

    # Start Processes
    stream.start()
    calc.start()
    graph_pair_process.start()
    graph_ind_w_means.start()
    # graph_cov_process.start()
    # graph_diff_process.start()
    # graph_cov_process.start()
    # graph_simple_process.start()
    # graph_simple_read_process.start()

    # Join Processes
    stream.join()
    calc.join()
    # graph_pair_process.join()
    # graph_diff_process.start()
    # graph_cov_process.join()
    # graph_simple_process.join()
    # graph_simple_read_process.join()






