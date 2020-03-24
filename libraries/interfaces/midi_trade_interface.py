import numpy as np
import pygame.midi
import sys

# Homegrown 
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.oanda_old import close_all_positions
from libraries.oanda_old import create_order

'''
Purpose: 
    
    Midi interface to open and close trades 

Input:
    
    EACH KEY ON BOARD EITHER A BUY OR SELL WITH ONE KILL BUTTON
    
'''

class Midi_controller():
        
    def __init__(self, pairs_to_graph, stop_loss=15):
        
        # Key_maps
        # self.pairs_to_graph = pairs_to_graph
        # self.sell_map = dict(zip(np.arange(29, 29 + len(list(pairs_to_graph)) * 2, 2), 
        #                     pairs_to_graph))
        # self.buy_map  = dict(zip(np.arange(72, 72 + len(list(pairs_to_graph)) * 2, 2), 
        #                     pairs_to_graph))

        self.sell_map = {57: 'EUR_USD'}
        self.buy_map  = {65: 'EUR_USD'}
        self.kill_map = 61
        
        # Trade Parameters
        self.stop_loss = 15
    
        # Print Maps for user
        print('Self.run() to start controller.  Self,quit() to stop.')
        self.maps()


    def maps(self):  
        print('\nSell:\n{}'.format(self.sell_map))
        print('\nBuy:\n{}'.format(self.buy_map))
        print('\nKill:\n{}'.format(self.kill_map))
        
    
    def and_other_function_to_list_update_parameters(self):
        pass
        
    def run(self):
        
        #Initialize midi
        pygame.midi.quit()   
        pygame.midi.init()
        input_id = pygame.midi.get_default_input_id()
        i = pygame.midi.Input( input_id )
        
        # Switch Parameters - midi provided does not give on or off differences
        position = False
        
        # Build Key press map - True if pressed, False when released
        key_on = {}
        for key in range(127):
            key_on[key] = False
        
        while True:
            
            try:
                # Read midi channel
                if i.poll():
                    midi_events = i.read(100)
                    key = midi_events[0][0][1]
            
                    # Turn key on or off
                    key_on[key] = not key_on[key]
                    
                     # Look for kill button.  Immedietely close positions.
                    if key == self.kill_map:
                        close_all_positions()
                        position = False
            
                    # Update instrument with latest key touch
                    if key in self.buy_map and not position:
                        create_order(self.buy_map[key], 'BUY', self.stop_loss)
                        position = True
                    
                    # update direction with latest key touch
                    if key in self.sell_map and not position:
                        create_order(self.sell_map[key], 'SELL', self.stop_loss)
                        position = True
            except:
                print('error')
                pygame.midi.quit()   
                pygame.midi.init()
                
                        
                
    def quit(self):
        pygame.midi.quit()   
        exit
                

    
# Previous Version - two handed press ( one on action , one on instrument)         
"""

THIS IS VERSION 1 - PROBABLY 2 IS JUST EASIER ( ONE TOUCH MAP)


import pygame.midi
import oandapyV20
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.orders as orders


###############################################################################
# Notes
###############################################################################

'''
Goal: Midi interface to place bets

Objectives:
    
    Only one bet will be open at a time.
    
    Bet will automatically be calculated to use most of margin
    
    Stop loss will be placed at a dollar value
    
Input:
    
    Row of keys for instrument
    Buy and Sell button
        These two buttons will have to be pressed together
    Kill button - closes all open positions
    
    
'''



###############################################################################
# Parameters
###############################################################################

# Oanda 
oanda_api = 'f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b'
oanda_account = '101-001-7518331-001'

#Initialize midi
pygame.midi.quit()   
pygame.midi.init()
input_id = pygame.midi.get_default_input_id()
i = pygame.midi.Input( input_id )

# Map midi keyboard buttons to instruments 
instrument_map = {61: 'EUR_USD',
                  63: 'AUD_CAD',
                  61: 'EUR_USD',
                  61: 'EUR_USD',
                  61: 'EUR_USD',
                  61: 'EUR_USD',
                  61: 'EUR_USD',
                  61: 'EUR_USD',
                  61: 'EUR_USD',
                  61: 'EUR_USD'}

# Map midi keyboard buttons to actions 
action_map = {    80: 'KILL',
                  84: 'BUY',
                  88: 'SELL'}

# Build Key press map - True if pressed, False when released
key_on = {}
for key in range(127):
    key_on[key] = False
    
# Switch Parameters
position = False
instrument_register = False
direction_register = False
instrument = ''
direction = ''
simultaneous = ''




###############################################################################
# Definitions
###############################################################################


def get_open_positions(client=oanda_api, account_id = oanda_account):
    client = oandapyV20.API(access_token=client)
    r = positions.OpenPositions(account_id)
    client.request(r)
    p = r.response
    instruments = []
    for position in p['positions']:
        if float(position['long']['units']) > 0:
            instruments.append((position['instrument'], 'long'))
        else:
            instruments.append((position['instrument'], 'short'))
    return instruments  


def close_position(client=oanda_api, account_id = oanda_account):
    instruments = get_open_positions()
    client = oandapyV20.API(access_token=client)
    for pair in instruments:
        if pair[1] == 'long':
            data = { "longUnits": "ALL" }
        else:
            data = { "shortUnits": "ALL" }
        r = positions.PositionClose(accountID=account_id,
                                     instrument=pair[0], 
                                     data = data)
        client.request(r)
        print('\nPositions closed: {}\n'.format(instrument))
        print(r.response)
    return
    

def create_order(instrument, direction, 
                 client=oanda_api, account_id = oanda_account):
    #, quantity, target, stop, account):
    client = oandapyV20.API(access_token=client)
    if direction == 'BUY':
        units = 35000
    elif direction == 'SELL':
        units = -35000
    else:
        print('Creat Order error: Direction uncertain')
    data   = {'order' : {"units": units, #quantity, 
                         "instrument": instrument, 
                         "timeInForce": "FOK", 
                         "type": "MARKET", 
                         "positionFill": "DEFAULT"}}
    '''
    'takeProfitOnFill' : {'price': str(round(target, 5)), 
                              'timeInForce' : 'GTC'},
    'stopLossOnFill':    {'price': str(round(stop, 5)), 
                              'timeInForce' : 'GTC'}}}
    '''
    r = orders.OrderCreate(account_id, data=data)
    client.request(r)    
    print('\nPositions Opened: {} {}\n'.format(instrument, direction))
    print(r.response) # int(r.response['orderCreateTransaction']['id'])
    return




###############################################################################
# Main loop and controls.  Read incoming Midi.  Open and close Oanda orders. 
###############################################################################

while True:
    
    # Read midi channel
    if i.poll():
        midi_events = i.read(100)
        key = midi_events[0][0][1]
        
        # Turn key on or off
        key_on[key] = not key_on[key]
    
        # Look for kill button.  Immedietely close positions.
        if key in action_map and action_map[key] == 'KILL':
            if key_on[key] == True:
                close_position(oanda_api, oanda_account)
                position = False
                        
        # Turn off registers when buttons are off
        if key_on[key] == False:
            if key in instrument_map:
                instrument_register = False
            if key in action_map:
                direction_register = False  

        # Update instrument with latest key touch
        if key_on[key] == True and key in instrument_map:
            instrument = instrument_map[key]
            instrument_register = True
        
        # update direction with latest key touch
        if key_on[key] == True and key in action_map:
            if action_map[key] != 'KILL':
                direction = action_map[key]
                direction_register = True
        
        # Only do anything if no positions are placed 
        # Look for both direction and instruement provided
        if not position and instrument_register and direction_register:
            create_order(instrument, direction, oanda_api, oanda_account)
            instrument_register = False
            direction_register = False
            position = True    
                
            
"""
        