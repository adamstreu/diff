from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import sys
import time
import copy
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.widgets.RemoteGraphicsView
from pyqtgraph.ptime import time
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.database import database_execute
from libraries.database import database_retrieve
from libraries.database import dictionary_insert
from libraries.currency import conversion
# from libraries.midi_controller import Midi_controller

import datetime

import time as t



import requests

import json

##############################################################################
# Notes
##############################################################################


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

    
##############################################################################
# PArameters
##############################################################################


# Database Parameters
db = '/Users/user/Desktop/diff/streaming.db'

# Trade Parameters
stop_loss = 15

# Graph Parameters
graphs = {}
graph_lines = ['pair', 'indicator', 'mean_1', 'mean_2'] # 'mean_3']
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
trading_pairs_index  = ['AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'AUD_USD', 'CAD_CHF']#, 
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
            given[r,c] = True
either = inverse | given


url = 'https://stream-fxpractice.oanda.com/v3/accounts/101-001-7518331-001/pricing/stream'
session = requests.Session()
session.trust_env = False  # Don't read proxy settings from OS
r = session.get(url)





def price_stream_to_db():
    
    

    

    url = 'https://stream-fxpractice.oanda.com/v3/accounts/101-001-7518331-001/pricing/stream'
    auth = 'Bearer f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b'
    instruments = ('instruments', 'EUR_USD') #  'EUR_USD,USD_CAD'
    headers = { 'Authorization': auth }
    params = {(instruments)}
    
    session = requests.Session()
    session.trust_env = False  # Don't read proxy settings from OS
    r = session.get(url)
    
    r = requests.get(url, headers=headers, params=params, stream=True)
    for line in r.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(json.loads(decoded_line))
            
    """
    import requests
    url = 'https://www.googleapis.com/qpxExpress/v1/trips/search?key=mykeyhere'
    payload = open("request.json")
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=payload, headers=headers)
    
    end = "https://stream-fxtrade.oanda.com/v3/accounts/101-001-7518331-001/pricing/stream?instruments=EUR_USD%2CUSD_CAD"
    end
    curl -H "Authorization: Bearer f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b" "https://stream-fxtrade.oanda.com/v3/accounts/101-001-7518331-001/pricing/stream?instruments=EUR_USD%2CUSD_CAD"
    
    r = requests.get('https://github.com/timeline.json')
    r.json()
    """
    """
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
            
        
"""
        
        
        
        
        






##############################################################################
# Graph Initialize
##############################################################################
    
    

pg.setConfigOptions(antialias=True)
# pg.setConfigOption('background', '#c7c7c7')
# pg.setConfigOption('foreground', '#000000')
app = QtGui.QApplication([])
p = pg.plot()
p.setTitle('currency')
p.setXRange(0,7)
# p.setYRange(-10,10)
# p.setWindowTitle('Current-Voltage')
p.setLabel('bottom', 'Bias', units='V', **{'font-size':'20pt'})
p.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))
# p.setLabel('left', 'Current', units='A',
#             color='#c4380d', **{'font-size':'20pt'})
p.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))
curve = p.plot(x=[], y=[], pen=pg.mkPen(color='#c4380d'))
p.showAxis('right')
p.setLabel('right', 'Dynamic Resistance', units="<font>&Omega;</font>",
            color='#025b94', **{'font-size':'20pt'})
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

x = np.arange(0, 10.01,0.01)
data = 5+np.sin(30*x)
data2 = -5+np.cos(30*x)
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
            fps = 1.0/dt
        else:
            s = np.clip(dt*3., 0, 1)
            fps = fps * (1-s) + (1.0/dt) * s
        p.setTitle('%0.2f fps' % fps)
    else:
        ptr = 0
    app.processEvents()  








# Run
if __name__ == '__main__':
    
    stream = Process(target=price_stream_to_db) #, 
    #                   args = (pairs_index, db, 'bids', q1, q2))
    stream.start() 
    stream.join()
    # t.sleep(3)

    

    ##############################################################################
    # Create SQL Tables
    ##############################################################################

    # Create Indicator Tables
    statement = 'create table if not exists table_name (' \
                'id integer primary key, timestamp text not null, '
    statement += ' real not null, '.join(pairs_index)
    statement += ' real not null);'
    for table in ['differences', 'mean_line_1', 'mean_line_2', 'mean_line_3']:
        database_execute(db, statement.replace('tables_name', table))
        database_execute(db, 'delete from {}'.format(table))

    # Create Currency Tables
    statement = 'create table if not exists currencies (' \
                'id integer primary key, timestamp text not null, '
    statement += ' real not null, '.join(currencies_index)
    statement += ' real not null);'
    database_execute(db, statement)
    database_execute(db, 'delete from currencies')

    

    # # Graph Run
    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(50)


    # import sys
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()

    

    




    
    
    


  


