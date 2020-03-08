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
sys.path.insert(1, '/Users/user/Desktop/algo')
from libraries.oanda import price_stream_to_db
from libraries.database import database_execute
from libraries.database import database_retrieve
from libraries.database import dictionary_insert
from libraries.currency import conversion
from libraries.midi_controller import Midi_controller


##############################################################################
# Notes
##############################################################################


"""
Notes:
    
    Add a buncjh more plots - how is it handled ( add back in update time - see example)
    Will probably still want to put graphing in its own pull from database
    Make sure putting data in in same order
    Check scrolling plots exapmle. - worth it to jsut keep adding data ?/ (later)
    
    
        


    REMEMBER  - CURRENCY COVERSION IS NOT EVEN RIGHT RIGH TNOW
    Do not really need to put data into sql that we aren't trading - might want to 
        exxplore it later though  ...  for now, just throw it all in'
        
    not importnant: 
        3 mean lines are fixe right now.  mena winows ordered small to big must be
"""






##############################################################################
# PArameters
##############################################################################


# Database Parameters
db = '/Users/user/Desktop/algo/streaming.db'

# Trade Parameters
stop_loss = 15

# Graph Parameters
data_points = 1000
mean_windows = [25, 500, 900]
bids_or_asks = 'bids'

# Queue
q = Queue()

# Start Feeder Process
for i in range(1, 1000):
    q.put(i)


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
trading_pairs_index  = ['AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'AUD_USD', 'CAD_CHF', 
                        'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 'EUR_USD', 
                        'NZD_CAD', 'NZD_CHF', 'NZD_USD', 'USD_CHF' ]
currencies_index = list(set('_'.join(pairs_index).split('_')))
currencies_index.sort()

# Create Arrays
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





def update():

    global db, data_points, mean_windows, q
    global pairs_index, trading_pairs_index, currencies_index
    global currencies, pairs, differences, calculated, mean_lines
    global inverse, given, either
    
    global p6, p4, curve1, curve2, curve3, curve4

    if not q.empty():
        
        # Fetch Pricing Data            (from q ) 
        row = q.get()
        
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
             
        #Graph Data
        ind_graph = differences[0]
        m1_graph = mean_lines[0, 0, :]
        m2_graph = mean_lines[2, 0, :]
        cur_graph = currencies[0]
        curve1.setData(ind_graph)
        curve2.setData(m1_graph)
        curve3.setData(m2_graph)
        curve4.setData(cur_graph)
        p6.enableAutoRange('xy', True)
        p4.enableAutoRange('xy', True)



# Run
if __name__ == '__main__':
        

    # Start Feeder Process
    for i in range(1, 19400):
        q.put(i)
    
    # GRaph initialization
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000,1000)
    win.setWindowTitle('pyqtgraph example: Plotting')
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    p6 = win.addPlot(title="Updating plot")    
    # Set Curves
    curve1 = p6.plot(np.ones(data_points), pen=(255,0,0), name="Red curve")
    curve2 = p6.plot(np.ones(data_points), pen=(0,255,0), name="Green curve")
    curve3 = p6.plot(np.ones(data_points), pen=(0,0,255), name="Blue curve")    
    
    win.nextRow()
    p4 = win.addPlot(title="Parametric, grid enabled")
    curve4 = p4.plot(np.ones(data_points), pen=(0,0,255), name="Blue durve")  
    
    # Graph Run
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    QtGui.QApplication.instance().exec_()

    
    
    

  




    
    
    


  


