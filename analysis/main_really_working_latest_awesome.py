"""

SOLVED.

"""




if True:
    import numpy as np
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

    Can I track how many graphs I miss
    Put in second line        
    How do extra currencies hold up
    coloring, etc.  dont get obsessed
    maybe a dot at current level on pairs and difference

Later:
    
    Any pyqt tricks to speee up graph? 
        (downsampling ? ) 

"""


if True:
    # Parameters
    bids_or_asks = 'bids'
    data_points = 30000
    mean_windows = [25, 500, 900]   # Not using yet
    currency_to_graph = 19

    # Queues and Arrays
    q_streaming = Queue()

    # Raw Arrays
    raw_diff = RawArray('d', data_points + 1)
    raw_pair = RawArray('d', data_points + 1)

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
    count = 1
    while True:
        for ticks in api.request(r):
            if ticks['type'] == 'PRICE' and ticks['instrument'] in pairs_index:
                try:
                    # Update Pairs with Latest price.  Set timestamp
                    _pairs[pairs_index.index(ticks['instrument'])] = float(ticks[bids_or_asks][0]['price'])
                    q.put([count, _pairs])
                    # Debug
                    if count % 100 == 0:
                        print('Stream Recieved    {}:     {}'.format(count, ticks['time']))
                        print('Stream Queued      {}:     {}'.format(count, datetime.datetime.now()))
                    count += 1
                except Exception as e:
                    print('Stream | Calculation exception: {}'.format(e))


def calculations(pairs_index, currencies_index, q, data_points, currency_to_graph, raw_diff, raw_pair):
    # Raw Array

    R_diff = np.frombuffer(raw_diff, dtype=np.float64).reshape(data_points + 1)
    R_pair = np.frombuffer(raw_pair, dtype=np.float64).reshape(data_points + 1)

    # Required Numpy Arrays
    currencies = np.ones((len(currencies_index), data_points))
    pairs = np.ones((len(pairs_index), data_points))
    differences = copy.deepcopy(pairs)
    calculated = copy.deepcopy(pairs)
    # mean_lines = np.ones((len(mean_windows), len(pairs_index), data_points))

    # Inverse and normal denominators for conversion calculation
    inverse = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
    given = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
    for r in range(inverse.shape[0]):
        for c in range(inverse.shape[1]):
            if currencies_index[r] == pairs_index[c].split('_')[0]:
                inverse[r, c] = True
            if currencies_index[r] == pairs_index[c].split('_')[1]:
                given[r, c] = True
    either = inverse | given

    # Start
    count = 0
    while True:
        if not q.empty():
            _pairs = q.get()
            cal_retrieve = datetime.datetime.now()
            count = _pairs[0]
            _pairs = _pairs[1]

            # Roll arrays
            # pairs[: -1] = pairs[1:]
            # currencies[: -1] = currencies[1:]
            # differences[: -1] = differences[1:]
            # calculated[: -1] = calculated[1:]

            pairs = np.roll(pairs, -1)
            differences = np.roll(differences, -1)
            calculates = np.roll(calculated, -1)



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
            # Timestamp
            calc_complete = datetime.datetime.now()

            np.copyto(R_diff[: -1], differences[currency_to_graph])
            np.copyto(R_pair[: -1], pairs[currency_to_graph])
            np.copyto(R_diff[-1:], count)
            np.copyto(R_pair[-1:], count)

            array_copy_time = datetime.datetime.now()

            # Debug
            if count % 100 == 0:
                print('Stream Recieved    {}:     {}'.format(count, cal_retrieve))
                print('Calc Load Complete {}:     {}'.format(count, datetime.datetime.now()))
                print('Calc time          {}:     {}'.format(count, calc_complete - cal_retrieve))
                print('Load Arrays        {}:     {}'.format(count, array_copy_time - calc_complete))
                print('Last pair point    {}:     {}'.format(R_diff[-1], R_diff[-2]))
                print('Calc def compl     {}:     {}'.format(R_diff[-1], datetime.datetime.now()))
                # print(differences[19])
            count += 1


def graph(data_points, raw_diff, raw_pair):

    global curve, p6, data, here, fps, lastTime, ptr


    # Instantiate
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('pyqtgraph example: Plotting')
    p6 = win.addPlot(title="Updating plot")
    curve = p6.plot(pen='y')
    here = 0
    data = np.frombuffer(raw_diff).reshape(data_points + 1)

    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        global curve, p6, raw_diff, raw_pair, data, here, fps, lastTime, ptr

        '''
        # curve.setData(data[ptr % 10])
        data = np.frombuffer(raw_diff).reshape(data_points + 1)
        # data = data[data != 0]
        count = data[-1]
        curve.setData(data[:-1])
        # if ptr == 0:
        # p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
        '''
        count = data[-1]
        if count > here: # only update on new data
            here = count
            curve.setData(data[data != 1][100:-1])  # 100 on to remove wonky startup values
            if count % 100 == 0:
                print('plot Number        {}:     {}'.format(count, data[-2]))
                print('plot On            {}:     {}'.format(count, datetime.datetime.now()))

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
            p6.setTitle('%0.2f fps' % fps)







    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)                   # FIND OUT MEANING OF VALUE
    QtGui.QApplication.instance().exec_()


if __name__ == '__main__':

    #  Processes
    stream = Process(target=price_stream,
                     args=(pairs_index, bids_or_asks, q_streaming, oanda_api, oanda_account))
    calc = Process(target=calculations,
                   args=(pairs_index, currencies_index, q_streaming, data_points, currency_to_graph, raw_diff, raw_pair))
    graph_process = Process(target=graph, args=(data_points, raw_diff, raw_pair))

    # Start Processes
    stream.start()
    calc.start()
    graph_process.start()
    calc.join()
    graph_process.join()
    stream.join()





