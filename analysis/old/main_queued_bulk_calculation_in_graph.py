# Import ans Notes:
if True:
    import numpy as np
    import sys
    import copy
    import datetime
    import time
    from multiprocessing import Process, Queue
    import oandapyV20
    import oandapyV20.endpoints.pricing as pricing
    import pyqtgraph as pg
    import pyqtgraph.widgets.RemoteGraphicsView
    from pyqtgraph.Qt import QtGui, QtCore
    from pyqtgraph.ptime import time as pyqt_time
    from matplotlib import pyplot as plt

    sys.path.insert(1, '/Users/user/Desktop/diff')
    from libraries.database import database_execute
    from libraries.database import database_retrieve
    from libraries.database import dictionary_insert
    from configs import oanda_api, oanda_account  # Oanda Account Parameters

    """
    
    GOAL:
            
        Have something for tomorrow
        
        
    To DO:
                    
        Get graph working with iterate update - SUCCESS ! ! ! !  Can not worry about bulk quite yet.
        Nope: only works with few points, like, 2000 ( one minute )
        OK.  This is note working for 4000000 data points though.  Taking minutes.
        
        Let run for a while: see where it starts lagging
            Started at 1:25
            
        Bulk Update
        
        
        Graphing:
        
            downsampling?  and other tricka
            Is just add data to graph line faster?
            follow x every time - allow y  ? 
            test more data points.  20000 is only 5 minutes of data.  I want an hour.  Remove a mean
            Get Graphing working as needed for tomorrow
            coloring, etc.  dont get obsessed
            maybe add other currencies if possible,
            check update speeds and graph against incoming data, then do bulk.
            some timestamp data somewhere - even just print beggining of plot
            maybe a dot at current level on pairs and differences
            Better way to start - you know - don't wiat an hour for graph to fill - 
                if graph less than data points, just graph what data we have ( [-count : ] )
            If resizsing - how to get it to autoscroll x again.
            
        
    Later:
    
        When eveything is complete and working, later, we can see if a while loop is faster
            probably not and probably not important
        

        
    
    """

# Parameters and Initializations:
if True:
    # Parameters
    bids_or_asks = 'bids'
    q = Queue()
    data_points = 5000
    mean_windows = [25, 500, 900]
    currency_to_graph = 'EUR_USD'

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
                    q.put(_pairs)
                    # Debug
                    if count % 100 == 0:
                        print('Stream Received    {}:     {}'.format(count, ticks['time']))
                        print('Stream Queued      {}:     {}'.format(count, datetime.datetime.now()))
                    count += 1
                except Exception as e:
                    print('Stream | Calculation exception: {}'.format(e))


def graph(pairs_index, currencies_index, mean_windows, data_points, q, currency_to_graph):

    # Calculation Parameters - for use within update functio
    global currencies, pairs, differences, mean_lines, _pairs, calculated, pair_to_graph_index
    currencies = np.ones((len(currencies_index), data_points))
    pairs = np.ones((len(pairs_index), data_points))
    differences = copy.deepcopy(pairs)
    calculated = copy.deepcopy(pairs)
    mean_lines = np.ones((len(mean_windows), len(pairs_index), data_points))
    _pairs = np.ones(len(pairs_index))
    pair_to_graph_index = pairs_index.index(currency_to_graph)

    # Inverse and normal denominators for conversion calculation
    global inverse, given, either
    inverse = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
    given = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
    for r in range(inverse.shape[0]):
        for c in range(inverse.shape[1]):
            if currencies_index[r] == pairs_index[c].split('_')[0]:
                inverse[r, c] = True
            if currencies_index[r] == pairs_index[c].split('_')[1]:
                given[r, c] = True
    either = inverse | given

    # Initialize Graph
    global p, ptr, lastTime, fps, p2
    pg.setConfigOptions(antialias=True)
    # pg.setConfigOption('background', '#c7c7c7')
    # pg.setConfigOption('foreground', '#000000')
    app = QtGui.QApplication([])
    p = pg.plot()
    p.setWindowTitle(currency_to_graph)
    p.setLabel('bottom', 'Bias', units='V', **{'font-size': '20pt'})
    p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))
    p.setLabel('left', currency_to_graph, units='A',
               color='#c4380d', **{'font-size': '20pt'})
    p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))

    # Initialize curves
    global curve_difference, curve_pair, curve_mean_1, curve_mean_2, curve_mean_3
    curve_difference = p.plot(pen=pg.mkPen(color='#FA5858', width=2))
    # curve_mean_1 = p.plot(pen=pg.mkPen(color='#c4380d'))
    # curve_mean_2 = p.plot(pen=pg.mkPen(color='#c4380d'))
    curve_mean_3 = p.plot(pen=pg.mkPen(color='#c4380d', width=2))
    curve_pair = pg.PlotCurveItem(pen=pg.mkPen(color='#00FFFF', width=2))

    # Axis Stuff plus other graph stuff
    p.showAxis('right')
    p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
               color='#025b94', **{'font-size': '20pt'})
    p.getAxis('right').setPen(pg.mkPen(color='#025b94', width=1))
    p2 = pg.ViewBox()
    p.scene().addItem(p2)
    p.getAxis('right').linkToView(p2)
    p2.setXLink(p)
    p2.addItem(curve_pair)
    # Update Views - need to figure out show this does later
    def updateViews():
        global p2
        p2.setGeometry(p.getViewBox().sceneBoundingRect())
        p2.linkedViewChanged(p.getViewBox(), p2.XAxis)
    updateViews()
    p.getViewBox().sigResized.connect(updateViews)
    # Print frames per second
    ptr = 0
    lastTime = pyqt_time()
    fps = None


    # Debugging / Tracking Parameters
    global number, total_records_received, update_time, start, end
    start = datetime.datetime.now()
    end = datetime.datetime.now()
    total_records_received = 0

    # Run While Loop
    while True:

        try:
            # Get data from q
            pairs_collection = []
            count = 0
            while not q.empty():
                count += 1
                pairs_collection.append(q.get())
                total_records_received += 1
                if total_records_received % 100 == 0:
                    print('Graph Plot Record  {}:     {}\n'.format(total_records_received, datetime.datetime.now()))

            if len(pairs_collection) > 0:

                # Put data into usable form
                _pairs = np.array(pairs_collection)
                _pairs = _pairs.T
                q_length = _pairs.shape[1]

                # Roll arrays
                # pairs = np.roll(pairs, -q_length)
                # currencies = np.roll(currencies, -q_length)
                # differences = np.roll(differences, -q_length)
                # calculated = np.roll(calculated, -q_length)
                # mean_lines = np.roll(mean_lines, -q_length)
                pairs[:, : -q_length] = pairs[:, q_length:]
                currencies[:, : -q_length] = currencies[:, q_length:]
                differences[:, : -q_length] = differences[:, q_length:]
                calculated[:, : -q_length] = calculated[:, q_length:]

                # Update Pairs
                pairs[:, -q_length:] = _pairs

                # Calculate newest and update currencies
                a = np.tile(_pairs.T, (len(currencies_index), 1))  # begin again with transform
                a = a.reshape(q_length, len(currencies_index), -1)
                given1 = np.tile(given, (q_length, 1, 1))
                inverse1 = np.tile(inverse, (q_length, 1, 1))
                _currencies = 1 / ((a * given1).sum(2) + ((1 / a) * inverse1).sum(2) + 1)
                currencies[:, -q_length:] = _currencies.T  # VEry likely is wrong though

                # Calculate calculated and update calculated (yikes)
                a = np.tile(_currencies.T, (len(pairs_index), 1))
                a = a.reshape(q_length, len(pairs_index), -1)
                given2 = np.tile(given.T, (q_length, 1, 1))
                inverse2 = np.tile(inverse.T, (q_length, 1, 1))
                _calculated = (a * inverse2).sum(2) / (a * given2).sum(2)
                calculated[:, -q_length:] = _calculated.T  # VEry likely is wrong though

                # Calculate difference and update difference
                _differences = _pairs - _calculated.T  # Should go in same directions now ( I suppose )
                differences[:, -q_length:] = _differences

                # Update Curves - good enough for now
                if total_records_received >= 300:
                    curve_difference.setData(differences[pair_to_graph_index, -(total_records_received - 300):])
                    curve_pair.setData(pairs[pair_to_graph_index, -(total_records_received - 300):])
                    # curve_mean_1.setData(mean_lines[0, pair_to_graph_index, -(total_records_received - 299):])
                    # curve_mean_2.setData(mean_lines[1, pair_to_graph_index, -(total_records_received - 299):])
                    # curve_mean_3.setData(mean_lines[2, pair_to_graph_index, -(total_records_received - 299):])

                    # # Print Frames Per Minute
                    # ptr += 1
                    # now = pyqt_time()
                    # dt = now - lastTime
                    # lastTime = now
                    # if fps is None:
                    #     fps = 1.0 / dt
                    # else:
                    #     s = np.clip(dt * 3., 0, 1)
                    #     fps = fps * (1 - s) + (1.0 / dt) * s
                    # p.setTitle('%0.2f fps' % fps)

                    # Update Graph Information - last step in graph update process
                    app.processEvents()  # Is this required ?

        except Exception as e:
            print(e)


if __name__ == '__main__':

    # Streaming and Calculate and Feed Database
    stream = Process(target=price_stream,
                     args=(pairs_index, bids_or_asks, q, oanda_api, oanda_account))
    # Graph Process
    g1 = Process(target=graph,
                 args=(pairs_index, currencies_index, mean_windows, data_points, q, currency_to_graph))

    # # Start Processes
    stream.start()
    g1.start()
    g1.join()



    """
    On Iteration, empty q into np.array
    Bulk update on lines
    Plot Data fresh
    
    
    # Calculation Parameters - for use within update function
    if True:
    
        # Required Numpy Arrays
        global currencies, pairs, differences, mean_lines, _pairs, calculated, pair_to_graph_index
        currencies = np.ones((len(currencies_index), data_points))
        pairs = np.ones((len(pairs_index), data_points))
        differences = copy.deepcopy(pairs)
        calculated = copy.deepcopy(pairs)
        mean_lines = np.ones((len(mean_windows), len(pairs_index), data_points))
        _pairs = np.ones(len(pairs_index))
        pair_to_graph_index = pairs_index.index(currency_to_graph)
    
        # Inverse and normal denominators for conversion calculation
        global inverse, given, either
        inverse = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
        given = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
        for r in range(inverse.shape[0]):
            for c in range(inverse.shape[1]):
                if currencies_index[r] == pairs_index[c].split('_')[0]:
                    inverse[r, c] = True
                if currencies_index[r] == pairs_index[c].split('_')[1]:
                    given[r, c] = True
        either = inverse | given
    
        # Debugging / Tracking Parameters
        global number, total_records_received, update_time, start, end
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        total_records_received = 0
    
    # Initialize Graph
    if True:
    
        # Initialize Graph
        global p, x, curve, data, curve2, data2, ptr, lastTime, fps, p2
        pg.setConfigOptions(antialias=True)
        # pg.setConfigOption('background', '#c7c7c7')
        # pg.setConfigOption('foreground', '#000000')
        app = QtGui.QApplication([])
        p = pg.plot()
        p.setWindowTitle(currency_to_graph)
        p.setLabel('bottom', 'Bias', units='V', **{'font-size': '20pt'})
        p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))
        p.setLabel('left', currency_to_graph, units='A',
                   color='#c4380d', **{'font-size': '20pt'})
        p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))
    
        # Initialize curves
        global curve_difference, curve_pair, curve_mean_1, curve_mean_2, curve_mean_3
        curve_difference = p.plot(pen=pg.mkPen(color='#FA5858', width=2))
        # curve_mean_1 = p.plot(pen=pg.mkPen(color='#c4380d'))
        # curve_mean_2 = p.plot(pen=pg.mkPen(color='#c4380d'))
        curve_mean_3 = p.plot(pen=pg.mkPen(color='#c4380d', width=2))
        curve_pair = pg.PlotCurveItem(pen=pg.mkPen(color='#00FFFF', width=2))
    
        # Axis Stuff plus other graph stuiff
        p.showAxis('right')
        p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
                   color='#025b94', **{'font-size': '20pt'})
        p.getAxis('right').setPen(pg.mkPen(color='#025b94', width=1))
        p2 = pg.ViewBox()
        p.scene().addItem(p2)
        p.getAxis('right').linkToView(p2)
        p2.setXLink(p)
        p2.addItem(curve_pair)
        # Update Views - need to figure out show this does later
        def updateViews():
            global p2
            p2.setGeometry(p.getViewBox().sceneBoundingRect())
            p2.linkedViewChanged(p.getViewBox(), p2.XAxis)
        updateViews()
        p.getViewBox().sigResized.connect(updateViews)
        # Print frames per second
        ptr = 0
        lastTime = pyqt_time()
        fps = None
    
    # Main Graph Update Sequence
    def update():
    
        # List Global Parameters
        global p, x, curve, data, curve2, data2, ptr, lastTime, fps
        global curve_difference, curve_pair, curve_mean_1, curve_mean_2, curve_mean_2
        global currencies, pairs, differences, mean_lines, calculated
        global total_records_received, number, start, end
    
        # Get most recent data from queue - What is our plot time limit before blowing up the que - there will be one.
        # WE might eventually just have to have a fail safe.  We grab so much data, some is missed and just not updated
    
        # Get data from q
        pairs_collection = []
        count = 0
        while not q.empty():
            count += 1
            pairs_collection.append(q.get())
            total_records_received += 1
            if total_records_received % 100 == 0:
                print('Graph Plot Record  {}:     {}\n'.format(total_records_received, datetime.datetime.now()))
    
    
        if len(pairs_collection) > 0:
    
            # Put data into usable form
            _pairs = np.array(pairs_collection)
            _pairs = _pairs.T
            q_length = _pairs.shape[1]
    
            # Roll arrays
            # pairs = np.roll(pairs, -q_length)
            # currencies = np.roll(currencies, -q_length)
            # differences = np.roll(differences, -q_length)
            # calculated = np.roll(calculated, -q_length)
            # mean_lines = np.roll(mean_lines, -q_length)
            pairs[:, : -q_length] = pairs[:, q_length:]
            currencies[:, : -q_length] = currencies[:, q_length:]
            differences[:, : -q_length] = differences[:, q_length:]
            calculated[:, : -q_length] = calculated[:, q_length:]
    
            # Update Pairs
            pairs[:, -q_length:] = _pairs
    
            # Calculate newest and update currencies
            a = np.tile(_pairs.T, (len(currencies_index), 1))  # begin again with transform
            a = a.reshape(q_length, len(currencies_index), -1)
            given1 = np.tile(given, (q_length, 1, 1))
            inverse1 = np.tile(inverse, (q_length, 1, 1))
            _currencies = 1 / ((a * given1).sum(2) + ((1 / a) * inverse1).sum(2) + 1)
            currencies[:, -q_length:] = _currencies.T  # VEry likely is wrong though
    
            # Calculate calculated and update calculated (yikes)
            a = np.tile(_currencies.T, (len(pairs_index), 1))
            a = a.reshape(q_length, len(pairs_index), -1)
            given2 = np.tile(given.T, (q_length, 1, 1))
            inverse2 = np.tile(inverse.T, (q_length, 1, 1))
            _calculated = (a * inverse2).sum(2) / (a * given2).sum(2)
            calculated[:, -q_length:] = _calculated.T  # VEry likely is wrong though
    
            # Calculate difference and update difference
            _differences = _pairs - _calculated.T  # Should go in same directions now ( I suppose )
            differences[:, -q_length:] = _differences
    
            # Update Curves - good enough for now
            if total_records_received >= 300:
                curve_difference.setData(differences[pair_to_graph_index, -(total_records_received - 300):])
                curve_pair.setData(pairs[pair_to_graph_index, -(total_records_received - 300):])
                # curve_mean_1.setData(mean_lines[0, pair_to_graph_index, -(total_records_received - 299):])
                # curve_mean_2.setData(mean_lines[1, pair_to_graph_index, -(total_records_received - 299):])
                # curve_mean_3.setData(mean_lines[2, pair_to_graph_index, -(total_records_received - 299):])
    
                # Print Frames Per Minute
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
    
            # Update Graph Information - last step in graph update process
            app.processEvents()   # Is this required ?
    
    
    """

    '''
    
    Iterative Update - I need a full calculation before plotting.
    In order to do this - calculations must be matrix style.
    Hoepfully, I can drain the quei MUCH quicker than it fill
    The array calculations with multi dimension
    
    
    
    # Start q loop.  Retrieve pairs data.  Start debugging.
    number = 0
    start = datetime.datetime.now()
    while not q.empty():
        _pairs = q.get()
        number += 1
        total_records_received += 1
    
    
        # Roll arrays
        pairs = np.roll(pairs, -1)
        currencies = np.roll(currencies, -1)
        differences = np.roll(differences, -1)
        calculated = np.roll(calculated, -1)
        mean_lines = np.roll(mean_lines, -1)
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
        _differences = _pairs - _calculated # Should go in same directions now ( I suppose )
        differences[:, -1] = _differences
        # Calculate Mean lines and update
        # for i in range(len(mean_windows)):
        #     mean_lines[i, :, -1] = differences[:, -mean_windows[i]:].mean(1)
        mean_lines[2, :, -1] = differences[:, -mean_windows[2]:].mean(1)
    
    
        if total_records_received >= 300:
            # Update Curves - good enough for now
            curve_difference.setData(differences[pair_to_graph_index, -(total_records_received - 299):])
            curve_pair.setData(pairs[pair_to_graph_index,-(total_records_received - 299):])
            # curve_mean_1.setData(mean_lines[0, pair_to_graph_index, -(total_records_received - 299):])
            # curve_mean_2.setData(mean_lines[1, pair_to_graph_index, -(total_records_received - 299):])
            # curve_mean_3.setData(mean_lines[2, pair_to_graph_index, -(total_records_received - 299):])
    
            # Print Frames Per Minute
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
    
        # Print Debug and Cycle Time Information
        if total_records_received % 100 == 0:
            print('Graph Plot Record  {}:     {}'.format(total_records_received, datetime.datetime.now()))
            print('Update Loop        {},        {},'.format(number, end-start))  # This is not quite correct.
        end = datetime.datetime.now()
        count += 1
    
        # Update Graph Information - last step in graph update process
        app.processEvents()   # Is this required ?
    
    '''
    '''
    # Run continuous graph update sequence
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
    QtGui.QApplication.instance().exec_()
    '''















