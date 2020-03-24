import numpy as np
import sys
import copy
import datetime
import time
from multiprocessing import Process
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import pyqtgraph as pg
import pyqtgraph.widgets.RemoteGraphicsView
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.ptime import time as pyqt_time
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.database import database_execute
from libraries.database import database_retrieve
from libraries.database import dictionary_insert
from configs import oanda_api, oanda_account

"""

Goal:
    
    Try to have the ability to run one graph tomorrow - at least to see  if this is worth any bother.
    FUCK.  SQL is kiling me.  It;s too slow.  by a long shot.
    
Next:

    Seperate Calculatios process:  
    Process data in chunks as we fall behind
        Took a short cut (iteration as opposed to array calculation ) .  Will fix later, but important
    change indicator and graph so that i grab ALL rows since seen last 0 this will take a bit of work.
    get rid of a mean line as well
        
"""

##############################################################################
# Parameters
##############################################################################

# Database Parameters
db = '/Users/user/Desktop/diff/streaming.db'
bids_or_asks = 'bids'

# Graphing Parameters
data_points = 20000
mean_windows = [25, 500, 900]

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



def price_stream(pairs_index, currencies_index, db, bids_or_asks, oanda_api, oanda_account):

    """
    Stream Prices from Oanda for pairs list.  Insert into 'pairs' table
    Load data into a q if passed as argument
    """

    ##############################################################################
    # Create Databases
    ##############################################################################

    # Create Pair based databases
    for table in ['pairs', 'differences', 'mean_1', 'mean_2', 'mean_3']:
        statement = 'create table if not exists table_name (' \
                    'id integer primary key, timestamp text not null, '
        statement += ' real not null, '.join(pairs_index)
        statement += ' real not null);'
        database_execute(db, statement.replace('table_name', table))
        database_execute(db, 'delete from {}'.format(table))

    # Create currency databases
    statement = 'create table if not exists currencies (' \
                'id integer primary key, timestamp text not null, '
    statement += ' real not null, '.join(currencies_index)
    statement += ' real not null);'
    database_execute(db, statement)
    database_execute(db, 'delete from currencies')

    # Dictionary for loading values into SQL
    pairs = dict(zip(pairs_index, [1] * len(pairs_index)))
    pairs['timestamp'] = 'begin'

    # Streaming Parameters
    api = oandapyV20.API(access_token=oanda_api)
    params = {'instruments': ','.join(pairs_index)}
    r = pricing.PricingStream(accountID=oanda_account, params=params)

    # Start Data Stream
    count = 1
    while True:
        for ticks in api.request(r):
            if ticks['type'] == 'PRICE' and ticks['instrument'] in pairs_index:
                try:

                    # Retrieve data
                    timestamp = ticks['time']
                    pairs['timestamp'] = timestamp
                    pairs[ticks['instrument']] = float(ticks[bids_or_asks][0]['price'])

                    # Insert prices into 'pairs'
                    dictionary_insert(db, 'pairs', pairs)

                    # Debug
                    if count % 100 == 0:
                        print('Oanda Received   {}:     {}'.format(count, timestamp))
                        print('SQL   Loaded     {}:     {}'.format(count, datetime.datetime.now()))
                    count += 1

                except Exception as e:
                    print('Stream | Calculation exception: {}'.format(e))


def inidicator_calculation(pairs_index, currencies_index, db, mean_windows, data_points):

    """
    Some Struff Here
    """
    count = 1

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

    # Required Arrays
    currencies = np.ones((len(currencies_index), data_points))
    pairs = np.ones((len(pairs_index), data_points))
    differences = copy.deepcopy(pairs)
    calculated = copy.deepcopy(pairs)
    mean_lines = np.ones((len(mean_windows), len(pairs_index), data_points))
    _pairs = np.ones(len(pairs_index))

    # Fetch Latest row from pairs
    row = 0
    while True:
        latest_row = database_retrieve(db, 'SELECT id from pairs ORDER BY id DESC LIMIT 1')
        if not len(latest_row) == 0:
            latest_row = int(latest_row[0][0])
            if latest_row > row:

                # Fetch data for new information
                query = database_retrieve(db, 'select * from pairs where id between {} and {}'.format(row, latest_row))
                query = query[1:]

                # Shortcut for now till I see that all workflow is working together ( haven't check graphing yet)
                for _pairs in query:
                    count += 1
                    timestamp = _pairs[1]
                    _pairs = _pairs[2:]


                    try:

                        ##############################################################################
                        # Calculate Indicator
                        ##############################################################################

                        # Roll arrays
                        differences = np.roll(differences, -1)
                        # Calculate newest and update currencies
                        a = np.tile(_pairs, (len(currencies_index), 1))
                        _currencies = 1 / ((a * given).sum(1) + ((1 / a) * inverse).sum(1) + 1)
                        # Calculate calculated and update calculated (yikes)
                        a = np.tile(_currencies, (len(pairs_index), 1))
                        _calculated = (a * inverse.T).sum(1) / (a * given.T).sum(1)
                        # Calculate difference and update difference
                        _differences = _calculated - _pairs
                        differences[:, -1] = _differences
                        # Calculate Mean lines and update
                        means = []
                        for window in mean_windows:
                            means.append(differences[:, -window:].mean(1))

                        ##############################################################################
                        # Insert Indicator into Tables
                        ##############################################################################

                        calculated_time = datetime.datetime.now()
                        for each in [('pairs', pairs_index, _pairs),
                                     ('differences', pairs_index, _differences),
                                     ('currencies', currencies_index, _currencies),
                                     ('mean_1', pairs_index, means[0]),
                                     ('mean_2', pairs_index, means[1]),
                                     ('mean_3', pairs_index, means[2])]:
                            d = dict(zip(each[1], each[2]))
                            d['timestamp'] = timestamp
                            dictionary_insert(db, each[0], d)

                        
                        # Debug
                        if count % 100 == 0:
                            print('SQL Received   {}:     {}'.format(count, timestamp))
                            print('Ind Calculated {}:     {}'.format(count, calculated_time))
                            print('Ind Loaded     {}:     {}'.format(count, datetime.datetime.now()))
                        # Update Latest Row and continue

                    except Exception as e:
                        print(' {}'.format(e))

            row = latest_row




def graph(currency_to_graph):
    # global difference_graph_line
    # global pair_graph_line
    # global mean_line_1_graph
    # global mean_line_2_graph
    # global mean_line_3_graph
    global title
    title = currency_to_graph

    #############################################################################
    # Initialize Graph
    #############################################################################

    global p, x, curve, data, curve2, data2, ptr, lastTime, fps
    global p2
    global row

    pg.setConfigOptions(antialias=True)
    # pg.setConfigOption('background', '#c7c7c7')
    # pg.setConfigOption('foreground', '#000000')

    app = QtGui.QApplication([])

    p = pg.plot()
    # p.setXRange(0, 10)
    # p.setYRange(-10, 10)
    p.setWindowTitle(title)
    p.setLabel('bottom', 'Bias', units='V', **{'font-size': '20pt'})
    p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3), showValues=False)
    p.setLabel('left', currency_to_graph, units='A',
               color='#c4380d', **{'font-size': '20pt'})
    p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3), showValues=False)
    curve = p.plot(x=[], y=[], pen=pg.mkPen(color='#c4380d'))
    curve3 = p.plot(x=[], y=[], pen=pg.mkPen(color='#0000ff', width=1))
    curve4 = p.plot(x=[], y=[], pen=pg.mkPen(color='#0080ff', width=1))
    curve5 = p.plot(x=[], y=[], pen=pg.mkPen(color='#00ffff', width=1))
    p.showAxis('right')
    p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
               color='#025b94', **{'font-size': '20pt'})
    p.getAxis('right').setPen(pg.mkPen(color='#025b94', width=3))

    p2 = pg.ViewBox()
    p.scene().addItem(p2)
    p.getAxis('right').linkToView(p2)
    p2.setXLink(p)
    # p2.setYRange(-10, 10)

    curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#025b94', width=1))

    p2.addItem(curve2)

    def updateViews():
        global p2
        p2.setGeometry(p.getViewBox().sceneBoundingRect())
        p2.linkedViewChanged(p.getViewBox(), p2.XAxis)

    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    x = np.arange(0, 10.01, 0.01)
    data = 5 + np.sin(30 * x)
    data2 = -5 + np.cos(30 * x)
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        global p, x, curve, data, curve2, data2, ptr, lastTime, fps

        curve.setData(x=x[:ptr], y=data[:ptr])
        curve2.setData(x=x[:ptr], y=data2[:ptr])
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

        #
        # # Debugging print plot time
        # if row % 100 == 0 and title == 'EUR_USD':
        #     print('Plotted {}:     {}'.format(row, datetime.datetime.now()))

        app.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
    QtGui.QApplication.instance().exec_()


if __name__ == '__main__':

    # Instantiate Processes
    stream = Process(target=price_stream, args=(pairs_index, currencies_index, db, bids_or_asks,
                                                oanda_api, oanda_account))
    calculate = Process(target=inidicator_calculation,
                        args=(pairs_index, currencies_index, db, mean_windows, data_points))
    # g1 = Process(target=graph, args=('EUR_USD', latest_row))


    # Start Processes
    stream.start()
    # calculate.start()
    # g1.start()




    """

    # Graph indicators and currencies
    g1 = Process(target=graph_indicator,
                 args=(pairs_index, currencies_index, q1, db, data_points, mean_windows,
                       pairs_index.index('EUR_USD')))
    ##############################################################################
    # Parameters
    ##############################################################################
    currencies = np.ones((len(currencies_index), data_points))
    pairs = np.ones((len(pairs_index), data_points))
    differences = copy.deepcopy(pairs)
    calculated = copy.deepcopy(pairs)
    mean_lines = np.ones((len(mean_windows), len(pairs_index), data_points))

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

    ##############################################################################
    # Indicator Calculations
    ##############################################################################

    # Retrieve Most Recent Pair Data From streaming Queue
    while True:
        if not q.empty():
            row = q.get()
            try:
                if row % 500 == 0 and title == 'EUR_USD':
                    print('Received {}:     {}'.format(row, datetime.datetime.now()))
                # Fetch Oanda pricing from databases.  Set timestamp
                statement = 'select * from pairs where id = {}'.format(row)
                _pairs = database_retrieve(db, statement)[0]
                timestamp = _pairs[1]
                _pairs = _pairs[2:]

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
                _differences = _calculated - _pairs
                differences[:, -1] = _differences

                # Calculate Mean lines and update
                for i in range(len(mean_windows)):
                    mean_lines[i, :, -1] = differences[:, -mean_windows[i]:].mean(1)

                # Prepare first few row of data for 'nicer' plotting ( good auto zoom before lots of data )
                if row < 100:
                    differences = np.tile(differences[:, -1], (differences.shape[1], 1)).T
                    pairs = np.tile(pairs[:, -1], (pairs.shape[1], 1)).T
                    ml_shape = mean_lines.shape
                    mean_lines = np.tile(mean_lines[:, :, -1].reshape(-1, 1), (1, 1, mean_lines.shape[2]))
                    mean_lines = mean_lines.reshape(ml_shape)
                # Graph lines update
                difference_graph_line = differences[currency_to_graph]
                pair_graph_line = pairs[currency_to_graph]
                mean_line_1_graph = mean_lines[0, currency_to_graph, :]
                mean_line_2_graph = mean_lines[1, currency_to_graph, :]
                mean_line_3_graph = mean_lines[2, currency_to_graph, :]
                # # update()


            # On Exception
            except Exception as e:
                print(e)

            # Debug - time test
            if row % 500 == 0 and title == 'EUR_USD':
                print('calculated {}:  {}'.format(row, datetime.datetime.now()))
    """

    """
    should onlt
    sdf
    """
    '''

    def graph():
        global p, x, curve, data, curve2, data2, ptr, lastTime, fps
        global p2
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', '#c7c7c7')
        pg.setConfigOption('foreground', '#000000')

        app = QtGui.QApplication([])

        p = pg.plot()
        p.setXRange(0, 10)
        p.setYRange(-10, 10)
        p.setWindowTitle('Current-Voltage')
        p.setLabel('bottom', 'Bias', units='V', **{'font-size': '20pt'})
        p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))
        p.setLabel('left', 'Current', units='A',
                   color='#c4380d', **{'font-size': '20pt'})
        p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))
        curve = p.plot(x=[], y=[], pen=pg.mkPen(color='#c4380d'))
        p.showAxis('right')
        p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
                   color='#025b94', **{'font-size': '20pt'})
        p.getAxis('right').setPen(pg.mkPen(color='#025b94', width=3))

        p2 = pg.ViewBox()
        p.scene().addItem(p2)
        p.getAxis('right').linkToView(p2)
        p2.setXLink(p)
        p2.setYRange(-10, 10)

        curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#025b94', width=1))
        p2.addItem(curve2)

        def updateViews():
            global p2
            p2.setGeometry(p.getViewBox().sceneBoundingRect())
            p2.linkedViewChanged(p.getViewBox(), p2.XAxis)

        updateViews()
        p.getViewBox().sigResized.connect(updateViews)

        x = np.arange(0, 10.01, 0.01)
        data = 5 + np.sin(30 * x)
        data2 = -5 + np.cos(30 * x)
        ptr = 0
        lastTime = pyqt_time()
        fps = None

        def update():
            global p, x, curve, data, curve2, data2, ptr, lastTime, fps

            curve.setData(x=x[:ptr], y=data[:ptr])
            curve2.setData(x=x[:ptr], y=data2[:ptr])
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

            app.processEvents()  ## force complete redraw for every plot.  Try commenting out to see if a different in speed occurs.

        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(0)
        QtGui.QApplication.instance().exec_()

    graph()

    '''

    """
    ##############################################################################
    # Push Prices into Sql Tables 
    ##############################################################################

    def db_statement(table, columns, values, timestamp):
        statement = 'INSERT INTO {} (timestamp, {}) values ( {}, {})'
        statement.format(table, ','.join(columns), timestamp, ','.join(values))

    database_execute(db, db_statement('currencies', pairs_index, _currencies, timestamp))
    database_execute(db, db_statement('differences', pairs_index, _differences, timestamp))
    database_execute(db, db_statement('mean_line_1', pairs_index, _differences, timestamp))
    database_execute(db, db_statement('mean_line_2', pairs_index, _differences, timestamp))
    database_execute(db, db_statement('mean_line_3', pairs_index, _differences, timestamp))

    # Update Arrays
    for q in q_arr:
        q.put(row)
     """

    """
    # Graph Parameters
    graphs = {}
    graph_lines = ['pair', 'indicator', 'mean_1', 'mean_2', 'mean_3]
    graph_colors = [(0, 0, 255), (255, 0, 0), (200, 200, 200),
                    (50, 50, 50), (100, 100, 100)]


    global p, x, curve, data, curve2, data2, ptr, lastTime, fps
    global p2
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', '#c7c7c7')
    pg.setConfigOption('foreground', '#000000')

    app = QtGui.QApplication([])

    p = pg.plot()
    p.setXRange(0, 10)
    p.setYRange(-10, 10)
    p.setWindowTitle('Current-Voltage')
    p.setLabel('bottom', 'Bias', units='V', **{'font-size': '20pt'})
    p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))
    p.setLabel('left', 'Current', units='A',
               color='#c4380d', **{'font-size': '20pt'})
    p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))
    curve = p.plot(x=[], y=[], pen=pg.mkPen(color='#c4380d'))
    p.showAxis('right')
    p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
               color='#025b94', **{'font-size': '20pt'})
    p.getAxis('right').setPen(pg.mkPen(color='#025b94', width=3))

    p2 = pg.ViewBox()
    p.scene().addItem(p2)
    p.getAxis('right').linkToView(p2)
    p2.setXLink(p)
    p2.setYRange(-10, 10)

    curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#025b94', width=1))
    p2.addItem(curve2)

    def updateViews():
        global p2
        p2.setGeometry(p.getViewBox().sceneBoundingRect())
        p2.linkedViewChanged(p.getViewBox(), p2.XAxis)

    updateViews()
    p.getViewBox().sigResized.connect(updateViews)

    x = np.arange(0, 10.01, 0.01)
    data = 5 + np.sin(30 * x)
    data2 = -5 + np.cos(30 * x)
    ptr = 0
    lastTime = pyqt_time()
    fps = None

    def update():
        global p, x, curve, data, curve2, data2, ptr, lastTime, fps

        curve.setData(x=x[:ptr], y=data[:ptr])
        curve2.setData(x=x[:ptr], y=data2[:ptr])
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

        app.processEvents()  ## force complete redraw for every plot.  Try commenting out to see if a different in speed occurs.

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
    QtGui.QApplication.instance().exec_()

        # # Graph Run
    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(50)

    # import sys
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()

    """

"""
Next:

    SINGLE PROCESS MIGHT BE NO GOOD - WE ARE WAY BEHIN HERE AT 20000 POINTS, WHICH REALLY IS
        AND WE ARE ONLY AT ONE GRAPH AND NO INDICATOR EXTRA


    Continue Getting the one process Plot working - Goal for today:   OK

        get more points in graph
        add exceion handling maybe - don't care yet.
        Look at timestmp - are we keeping up ?
            Looks like functions and plotting take about .3 seconds - not great - and that's on;y with 2000 points - we need 20000'
        get rest of pairs in graph
        Must check all matrix calculations
        fix auto adjusting plot
        Get Ready to stream for tomorrow with just the one processes

        Frames per second count - Ready to compare methods when
            combining both in a pricess - which is best, compare tonight.

        pre_load data so it's not so bad'


    and at the end dof all of this,
        the difference in price will just be because I didn' tgett all the data cause of the streaming rate ( 4 per second per instrumnt)'



Later:

    Get multiple processs working:

        Learn how to use pyqt really - but only after I'm ready with everything else
        Check scrolling plots exapmle. - worth it to jsut keep adding data ?/ (later)
        Plot all currencies together ( or seperate )
        buy sell button
        turn off on currencies
        3 mean lines are fixe right now.  mena winows ordered small to big must be

"""
"""
##############################################################################
# PArameters
###################################

    url = 'https://stream-fxpractice.oanda.com/v3/accounts/101-001-7518331-001/pricing/stream'
    auth = 'Bearer f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b'
    instruments = ('instruments', 'EUR_USD')  # 'EUR_USD,USD_CAD'
    headers = {'Authorization': auth}
    params = {(instruments)}

    session = requests.Session()
    session.trust_env = False  # Don't read proxy settings from OS
    r = session.get(url)

    r = requests.get(url, headers=headers, params=params, stream=True)
    for line in r.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(json.loads(decoded_line)['time'])



# Database Parameters
db = '/Users/user/Desktop/diff/streaming.db'

# Trade Parameters
stop_loss = 15

# Graph Parameters
graphs = {}
graph_lines = ['pair', 'indicator', 'mean_1', 'mean_2']  # 'mean_3']
graph_colors = [(0, 0, 255), (255, 0, 0), (200, 200, 200),
                (50, 50, 50), (100, 100, 100)]
data_points = 20000
mean_windows = [25, 500, 900]
bids_or_asks = 'bids'

# Setup Ques
q1 = Queue()
q2 = Queue()

##############################################################################
# Data Structures
##############################################################################


# Trading and Currency Universe
pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD',
               'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD',
               'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP',
               'EUR_HKD', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
               'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD',
               'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD',
               'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD',
               'USD_JPY']
trading_pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'AUD_USD', 'CAD_CHF']  # ,
# 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 'EUR_USD',
# 'NZD_CAD', 'NZD_CHF', 'NZD_USD', 'USD_CHF' ]
currencies_index = list(set('_'.join(pairs_index).split('_')))
currencies_index.sort()

# Create Arrays
streaming_pairs = np.ones(len(pairs_index))
currencies = np.ones((len(currencies_index), data_points))
pairs = np.ones((len(pairs_index), data_points))
differences = copy.deepcopy(pairs)
calculated = copy.deepcopy(pairs)
mean_lines = np.ones((len(mean_windows), len(pairs_index), data_points))

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

url = 'https://stream-fxpractice.oanda.com/v3/accounts/101-001-7518331-001/pricing/stream'
session = requests.Session()
session.trust_env = False  # Don't read proxy settings from OS
r = session.get(url)





    '''
    Stream Prices from Oanda for pairs list.  Insert into 'pairs' table
    Load data into a q if passed as argument
    '''

    # Create Pairs Dictionary
    pairs_dictionary = dict(zip((pairs_index), [1] * len(pairs_index)))
    pairs_dictionary['timestamp'] = str(np.datetime64('now'))

    # Create Database
    statement =  'create table if not exists pairs (' \
              'id integer primary key, timestamp text not null, '
    statement += ' real not null, '.join(pairs_index)
    statement += ' real not null);'
    database_execute(db, statement)
    database_execute(db, 'delete from pairs')

    # Streaming Parameters
    api = oandapyV20.API(access_token=oanda_api)
    params ={'instruments': ','.join(pairs_index)}
    r = pricing.PricingStream(accountID=oanda_account, params=params)

    print(api.request(r))

    # Setup qf
    q_count = 1

    # Start Data Stream
    for ticks in api.request(r):
        #rint(ticks)
        if ticks['type'] == 'PRICE':
            if ticks['instrument'] in pairs_index:

                # Update Pairs Dictionary with price and time
                price = float(ticks[bids_or_asks][0]['price'])
                pairs_dictionary[ticks['instrument']] = price
                pairs_dictionary['timestamp'] = ticks['time']

                # Insert pairs Dictionary into pairs table
                dictionary_insert(db, 'pairs', pairs_dictionary)

                # Load table record into q_co#                          # This might cause some erros - no handling
                q1.put(q_count)
                q_count += 1

         #       print('Q_count: {}'.format(q_count))


    """

"""

def indicator_calculations(q1, q2, pairs, currencies, differences,
                           calculated, mean_lines, pairs_index,
                           currencies_index):

    # Fetch Pricing Data
    while True:
        if not q1.empty():
            row = q1.get()

            # Fetch oanda pricing from databases.  Set timestamp
            statement = 'select * from pairs where id = {}'.format(row)
            _pairs = database_retrieve(db, statement)[0][2:]

            # Roll arrays
            pairs = np.roll(pairs, -1)
            currencies = np.roll(currencies, -1)
            differences = np.roll(differences, -1)
            calculated = np.roll(calculated, -1)
            mean_lines = np.roll(mean_lines, -1)

            # Update Pairs
            pairs[:, -1] = _pairs

            # Calculate newest and update currencies
            a = np.tile(pairs[:, -1], (len(currencies_index), 1))
            _currencies = 1 / ( (a * given).sum(1) + ((1 / a) * inverse).sum(1)  + 1)
            currencies[:, -1] = _currencies

            # Calculate calculated and update calculated (yikes)
            a = np.tile(_currencies, (len(pairs_index), 1))
            _calculated = (a * inverse.T).sum(1) / (a * given.T).sum(1)
            calculated[:, -1] = _calculated

            # Calculate difference and update difference
            _differences = _calculated - _pairs
            differences[:, -1] = _differences

            # Calculate Mean lines and update
            for i in range(len(mean_windows)):
                mean_lines[i, :,  -1] = differences[:, -mean_windows[i]:].mean(1)



            # Create
            # Put all into appropriate db




            # Pass Data To Graphing Function
            q2.put(row)




##############################################################################
# Graph Initialize
##############################################################################


pg.setConfigOptions(antialias=True)
# pg.setConfigOption('background', '#c7c7c7')
# pg.setConfigOption('foreground', '#000000')
app = QtGui.QApplication([])
p = pg.plot()
p.setTitle('currency')
p.setXRange(0, 7)
# p.setYRange(-10,10)
# p.setWindowTitle('Current-Voltage')
p.setLabel('bottom', 'Bias', units='V', **{'font-size': '20pt'})
p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))
# p.setLabel('left', 'Current', units='A',
#             color='#c4380d', **{'font-size':'20pt'})
p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))
curve = p.plot(x=[], y=[], pen=pg.mkPen(color='#c4380d'))
p.showAxis('right')
p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
           color='#025b94', **{'font-size': '20pt'})
p.getAxis('right').setPen(pg.mkPen(color='#025b94', width=3))

p2 = pg.ViewBox()
p.scene().addItem(p2)
p.getAxis('right').linkToView(p2)
p2.setXLink(p)
# p2.setYRange(-10,10)

curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#025b94', width=1))
p2.addItem(curve2)


def updateViews():
    global p2
    p2.setGeometry(p.getViewBox().sceneBoundingRect())
    p2.linkedViewChanged(p.getViewBox(), p2.XAxis)


updateViews()
p.getViewBox().sigResized.connect(updateViews)

x = np.arange(0, 10.01, 0.01)
data = 5 + np.sin(30 * x)
data2 = -5 + np.cos(30 * x)
ptr = 0
lastTime = time()
fps = None


##############################################################################
# Update Graph
##############################################################################
def update():
    global p, x, curve, data, curve2, data2, ptr, lastTime, fps
    if ptr < len(x):
        curve.setData(x=x[:ptr], y=data[:ptr])
        curve2.setData(x=x[:ptr], y=data2[:ptr])
        ptr += 1
        now = time()
        dt = now - lastTime
        lastTime = now
        if fps is None:
            fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            fps = fps * (1 - s) + (1.0 / dt) * s
        p.setTitle('%0.2f fps' % fps)
    else:
        ptr = 0
    app.processEvents()
"""
