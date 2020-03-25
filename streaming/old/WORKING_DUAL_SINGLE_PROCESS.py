
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
from libraries.oanda_old import price_stream_to_db
from libraries.database import database_execute
from libraries.database import database_retrieve
from libraries.database import dictionary_insert
from libraries.currency import conversion
# from libraries.midi_controller import Midi_controller
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
from configs import oanda_api, oanda_account
import datetime



##############################################################################
# Notes
##############################################################################


"""
Next:
    
    SINGLE PROCESS MIGHT BE NO GOOD - WE ARE WAY BEHIN HERE AT 20000 POINTS, WHICH REALLY IS 
        AND WE ARE ONLY AT ONE GRAPH AND NO INDICATOR EXTRA
        
    is iter as good' as oanda function'
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
        
    
Later: 
    
    Time how long it takes betweeen placing an order and it being filed
    
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



    
##############################################################################
# Graph Initialize
##############################################################################
    
    

pg.setConfigOptions(antialias=True)
# pg.setConfigOption('background', '#c7c7c7')
# pg.setConfigOption('foreground', '#000000')
app = QtGui.QApplication([])
p = pg.plot()
p.setTitle('currency')
p.setXRange(0,data_points)
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

# x = np.arange(0, 10.01,0.01)
# data = 5+np.sin(30*x)
# data2 = -5+np.cos(30*x)
ptr = 0
lastTime = time()
fps = None




##############################################################################
# Stream Data
##############################################################################


# Streaming Parameters
api = oandapyV20.API(access_token=oanda_api)
params ={'instruments': ','.join(pairs_index)} 
r = pricing.PricingStream(accountID=oanda_account, params=params)   

# Begin Stream
count = 0
for ticks in api.request(r):
    if ticks['type'] == 'PRICE':
        if ticks['instrument'] in pairs_index:
            
            
            ##############################################################
            # Retrieve Stream manipulate data
            ##############################################################
            
            # REtrieve Data
            instrument = pairs_index.index(ticks['instrument']) 
            price = float(ticks[bids_or_asks][0]['price'])
            streaming_pairs[instrument] = price
            
            # Roll arrays 
            pairs = np.roll(pairs, -1)
            currencies = np.roll(currencies, -1)
            differences = np.roll(differences, -1)
            calculated = np.roll(calculated, -1)
            mean_lines = np.roll(mean_lines, -1)
            
            # Update Pairs
            pairs[:, -1] = streaming_pairs
        
            # Calculate newest and update currencies
            a = np.tile(pairs[:, -1], (len(currencies_index), 1))
            _currencies = 1 / ( (a * given).sum(1) + ((1 / a) * inverse).sum(1)  + 1)
            currencies[:, -1] = _currencies
        
            # Calculate calculated and update calculated (yikes)
            a = np.tile(_currencies, (len(pairs_index), 1))
            _calculated = (a * inverse.T).sum(1) / (a * given.T).sum(1) 
            calculated[:, -1] = _calculated
            
            # Calculate difference and update difference 
            _differences = _calculated - streaming_pairs
            differences[:, -1] = _differences
            
            # Calculate Mean lines and update
            for i in range(len(mean_windows)):
                mean_lines[i, :,  -1] = differences[:, -mean_windows[i]:].mean(1)
                        
            
            ##############################################################
            # Update Graph
            ##############################################################

            # Bunch a stuff    
            if count == 0:
                differences[0] = np.ones(differences[0].shape) * differences[0, -1]
                pairs[0] = np.ones(pairs[0].shape) * pairs[0, -1]
            curve.setData(differences[0])
            curve2.setData(pairs[0])
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
            p.enableAutoRange('xy', True)
            
            # Has this one line missing been destroying me for hours?
            app.processEvents()  ## for

            # Debugging
            if count % 100 == 0:
                print(ticks['time'])
                print(str(datetime.datetime.now()))
                print(count)
                print()
            count += 1
                
                
##############################################################
# Fucking Finally working (for one pair)
##############################################################
    
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


"""

MISC BITS


# Run
if __name__ == '__main__':
    




    # Update graph data, plot and redraw        
    for pair in graphs:
        row = pairs_index.index(pair)
        graphs[pair]['pair'].setData(pairs[row])
        graphs[pair]['indicator'].setData(differences[row])
        graphs[pair]['mean_1'].setData(mean_lines[0, row, :])
        graphs[pair]['mean_2'].setData(mean_lines[2, row, :])
        # graphs[pair]['plot'].enableAutoRange('xy', True)
        graphs[pair]['pair'].setGeometry(graphs[pair]['pair'].vb.sceneBoundingRect())

    
    
    # Start Feeder Process
    for i in range(1, 19400):
        q.put(i)
    
    # GRaph initialization
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000,1000)
    win.setWindowTitle('pyqtgraph example: Plotting')
    # Enable antialiasing for prettier plots
    # pg.setConfigOptions(antialias=True)
    # p6 = win.addPlot(title="Updating plot")    
    # Set Curves
    # curve1 = p6.plot(np.zeros(data_points), pen=(255,0,0), name="Red curve")
    # curve2 = p6.plot(np.zeros(data_points), pen=(0,255,0), name="Green curve")
    # curve3 = p6.plot(np.zeros(data_points), pen=(0,0,255), name="Blue curve")    
    
    win.nextRow()
    
    # p4 = win.addPlot(title="Parametric, grid enabled")
    # curve4 = p4.plot(np.zeros(data_points), pen=(0,0,255), name="Blue durve")  
    
    
    # Instantiate Plots and curves
    for pair in trading_pairs_index: 
        graphs[pair] = {}
        graphs[pair]['plot'] = win.addPlot(title=pair)
        for i in range(len(graph_lines)):
            gl = graph_lines[i]
            name = pair + '_' + gl
            color = graph_colors[i]
            graphs[pair][gl] = graphs[pair]['plot'].plot(pen=color, name=name) # removed ploit data
            graphs[pair][gl].vb.sigResized.connect(updateViews)
            if gl == 'pair':
                
                ax3 = pg.AxisItem('right')
                ax3.setLabel('axis 3', color='#ff0000')
                ax3.setZValue(-10000)
                graphs[pair]['plot'].layout.addItem(ax3, 2, 3)
                # p1.layout.addItem(ax3, 2, 3)
                # graphs[pair]['plot'].getAxis('right').setLabel('axis2', color='#0000ff')
        win.nextRow()

        
    # a2 = pg.AxisItem("left")
    # a3 = pg.AxisItem("left")
    
    
    
    
    # Graph Run
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    QtGui.QApplication.instance().exec_()

    
    
    

  
"""



