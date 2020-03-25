import datetime
import numpy as np
import pandas as pd
import oandapyV20
import sys
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.forexlabs as labs
oandapyV20.endpoints.accounts.AccountInstruments
# sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.database import dictionary_insert
from libraries.database import database_execute
from libraries.database import database_retrieve
import yaml

# Import Configs File
configs_file = '/Users/user/Desktop/diff/configs.yaml'
with open(configs_file) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

# Oanda Parameters
oanda_api = configs['oanda_api']
oanda_account = configs['oanda_account']
daily_alignment = 0


def get_tradable_instruments(oanda_api=oanda_api, 
                             oanda_account=oanda_account):
    client = oandapyV20.API(access_token=oanda_api)
    params = {"instruments": ""}#"EU50_EUR,EUR_USD,US30_USD,FR40_EUR,EUR_CHF,DE30_EUR"}
    r = accounts.AccountInstruments(oanda_account, params=params)
    client.request(r)
    return r.response

def get_candles_bid_close(instrument, granularity, _from, _to,
                          da=daily_alignment, oanda_api=oanda_api):
    print('Fetching Candles.')
    client = oanda_api
    client = oandapyV20.API(access_token=client)
    params = {'from': _from,
              'to': _to,
              'granularity': granularity,
              'price': 'B',
              'count': 5000,
              'alignmentTimezone': 'America/Los_Angeles',
              'dailyAlignment': da}
    # Request Data
    coll = []
    for r in InstrumentsCandlesFactory(instrument = instrument, 
                                       params = params):
        try:
            client.request(r)
            coll.append(r.response)
        except Exception as e:
            print(e)
    # collect Returned Data into list.  Cast to floats.
    bidclose = []
    timestamp = []
    volume = []
    for i in range(len(coll)):
        for j in range(len(coll[i]['candles'])):
            bidclose.append(float(coll[i]['candles'][j]['bid']['c']))              
            timestamp.append(coll[i]['candles'][j]['time'])
            volume.append(float(coll[i]['candles'][j]['volume']))
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df[instrument] = pd.to_numeric(bidclose)
    df['volume'] = pd.to_numeric(volume)
    if not coll[i]['candles'][-1]['complete']:
        df.drop(df.last_valid_index(), inplace=True)
    return df






def fetch_account_details(oanda_api=oanda_api, oanda_account=oanda_account):

    # Fetch Account Details
    client = oandapyV20.API(access_token=oanda_api)
    r = accounts.AccountDetails(oanda_account)
    client.request(r)
    return r.response



def close_position(direction, pair):

    # Get Units to close based on direction of original trade
    if direction == 'buy':
        units = 'longUnits'
        return_fields = ['longOrderCreateTransaction',
                         'longOrderFillTransaction']
    else:
        units = 'shortUnits'
        return_fields = ['shortOrderCreateTransaction',
                         'shortOrderFillTransaction']
    # Request Parameters
    data = {units: "ALL"}
    client = oandapyV20.API(access_token=oanda_api)
    r = positions.PositionClose(accountID=oanda_account,
                                instrument=pair,
                                data=data)
    # Attempt Request and return response
    try:
        client.request(r)
        print(r.response)
        return r.response
    except:
        return r.response
        print(r.response)


def create_order(pair, direction, pips, commission, loss_target, purchase_price, oanda_api, oanda_account):
    
    def calculate_quantity(loss_target, commission, pips):
        quantity = int(loss_target / ((commission + pips) * .0001))
        if direction == 'buy':
            return quantity
        else:
            return - quantity

    def calculate_stop_loss_price(pips, direction, purchase_price):
        if direction == 'buy':
            stop_loss_price = purchase_price - (pips * .0001)
        else:
            stop_loss_price = purchase_price + (pips * .0001)
        return stop_loss_price
    
    # Calculate trade parameters
    quantity = calculate_quantity(loss_target, commission, pips)
    stop_loss_price = calculate_stop_loss_price(pips, direction, purchase_price)
    q = 30000
    if direction == 'sell':
        q = - q
    # Fix Trade Data
    data = {'order': {"units": q, #quantity,
                      "instrument": pair,
                      "timeInForce": "FOK",
                      "type": "MARKET",
                      "positionFill": "DEFAULT"}}#,
                      # 'stopLossOnFill': {'timeInForce': 'GTC',
                      #                    'price': str(round(stop_loss_price, 5))}}}

    # Place Order
    client = oandapyV20.API(access_token=oanda_api)
    r = orders.OrderCreate(oanda_account, data=data)
    try:
        client.request(r)
        return r.response
    except Exception as e:
        print(e)
        return r.response






def price_stream_nosql(pairs_index, db, bids_or_asks, q1,  oanda_api=oanda_api, oanda_account=oanda_account):
    """
    Stream Prices from Oanda for pairs list.  Insert into 'pairs' table
    Load data into a q if passed as argument
    """
    # Create Pairs Dictionary
    pairs_dictionary = dict(zip(pairs_index, [1] * len(pairs_index)))
    pairs_dictionary['timestamp'] = str(np.datetime64('now'))

    # Create Database
    statement = 'create table if not exists pairs (' \
                'id integer primary key, timestamp text not null, '
    statement += ' real not null, '.join(pairs_index)
    statement += ' real not null);'
    database_execute(db, statement)
    database_execute(db, 'delete from pairs')

    # Streaming Parameters
    api = oandapyV20.API(access_token='f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b')
    params = {'instruments': ','.join(pairs_index)}
    r = pricing.PricingStream(accountID=oanda_account, params=params)
    print('Streaming Oanda Pricing Data:')

    # Start Data Stream
    count = 1
    while True:
        try:
            for ticks in api.request(r):
                if ticks['type'] == 'PRICE':
                    if ticks['instrument'] in pairs_index:

                        # Update Pairs Dictionary with price and time
                        price = float(ticks[bids_or_asks][0]['price'])
                        pairs_dictionary[ticks['instrument']] = price
                        pairs_dictionary['timestamp'] = ticks['time']

                        # Insert pairs Dictionary into pairs table
                        dictionary_insert(db, 'pairs', pairs_dictionary)

                        # Load row into q and update count
                        q1.put(count)
                        # q2.put(count)
                        # q3.put(count)
                        # q4.put(count)
                        # q5.put(count)

                        count += 1
                        # Debug - testing timestamps
                        if count % 500 == 0:
                            print('Sent {}:        {}'.format(count, ticks['time']))
                            print('Queued {}:      {}'.format(count, (str(datetime.datetime.now()))))


        except Exception as e:
            print(e)




def price_stream(pairs_index, db, bids_or_asks, q1, #q2, q3, q4, q5,
                 oanda_api=oanda_api, oanda_account=oanda_account):

    """
    Stream Prices from Oanda for pairs list.  Insert into 'pairs' table
    Load data into a q if passed as argument
    """
    # Create Pairs Dictionary
    pairs_dictionary = dict(zip(pairs_index, [1] * len(pairs_index)))
    pairs_dictionary['timestamp'] = str(np.datetime64('now'))
    
    # Create Database
    statement = 'create table if not exists pairs (' \
                'id integer primary key, timestamp text not null, '
    statement += ' real not null, '.join(pairs_index)
    statement += ' real not null);'
    database_execute(db, statement)
    database_execute(db, 'delete from pairs')
        
    # Streaming Parameters
    api = oandapyV20.API(access_token='f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b')
    params ={'instruments': ','.join(pairs_index)} 
    r = pricing.PricingStream(accountID=oanda_account, params=params)   
    print('Streaming Oanda Pricing Data:')

    # Start Data Stream
    count = 1
    while True:
        try:
            for ticks in api.request(r):
                if ticks['type'] == 'PRICE':
                    if ticks['instrument'] in pairs_index:

                        # Update Pairs Dictionary with price and time
                        price = float(ticks[bids_or_asks][0]['price'])
                        pairs_dictionary[ticks['instrument']] = price
                        pairs_dictionary['timestamp'] = ticks['time']

                        # Insert pairs Dictionary into pairs table
                        dictionary_insert(db, 'pairs', pairs_dictionary)

                        # Load row into q and update count
                        q1.put(count)
                        # q2.put(count)
                        # q3.put(count)
                        # q4.put(count)
                        # q5.put(count)

                        count += 1
                        # Debug - testing timestamps
                        if count % 500 == 0:
                            print('Sent {}:        {}'.format(count, ticks['time']))
                            print('Queued {}:      {}'.format(count, (str(datetime.datetime.now()))))


        except Exception as e:
            print(e)



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


def close_all_positions(oanda_api=oanda_api, account_id=oanda_account):
    instruments = get_open_positions()
    client = oandapyV20.API(access_token=oanda_api)
    for pair in instruments:
        if pair[1] == 'long':
            data = { "longUnits": "ALL" }
        else:
            data = { "shortUnits": "ALL" }
        r = positions.PositionClose(account_id,
                                    instrument=pair[0],
                                    data = data)
        client.request(r)
        print('\nPositions closed: {}\n'.format(pair))
        print(r.response)
    return


"""
def create_order(instrument, direction, stop_loss, quantity=30000,
                 oanda_api=oanda_api, oanda_account=oanda_account):

    def calculate_quantity():
        pass
        

    def calculate_quantity():
        pass
    # Fix Trade Data
    quantity = quantity if direction == 'BUY' else -quantity
    data = {'order' : {"units": quantity,
                       "instrument": instrument,
                       "timeInForce": "FOK",
                       "type": "MARKET",
                       "positionFill": "DEFAULT"}}
    # Place Order
    client = oandapyV20.API(access_token=oanda_api)
    r = orders.OrderCreate(oanda_account, data=data)
    client.request(r)
    # Print Return
    print('\nPositions Opened: {} {}\n'.format(instrument, direction))
    print(r.response)

    '''
    'takeProfitOnFill' : {'price': str(round(target, 5)), 
                              'timeInForce' : 'GTC'},
    'stopLossOnFill':    {'price': str(stop_loss), 
                                               'timeInForce' : 'GTC'}}}
    '''

    return
"""


"""
###########'#
# OL
###########3


daily_alignment = 0
def get_candles(instrument, granularity, _from, _to, da=daily_alignment):
    #print('Fetching Candles.')
    client = 'f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b'
    client = oandapyV20.API(access_token=client)
    params = {'from': _from,
              'to': _to,
              'granularity': granularity,
              'price': 'BAM',
              'count': 5000,
              'alignmentTimezone': 'UTC',
              'dailyAlignment': da}
    # Request Data
    coll = []
    for r in InstrumentsCandlesFactory(instrument = instrument, 
                                       params = params):
        try:
            client.request(r)
            coll.append(r.response)
        except Exception as e:
            print(e)
    # collect Returned Data into list.  Cast to floats.
    bidlow = []
    bidhigh = []
    bidclose = []
    asklow = []
    askhigh = []
    askclose = []
    midopen = []
    midlow = []
    midhigh = []
    midclose = []
    timestamp = []
    volume = []
    for i in range(len(coll)):
        for j in range(len(coll[i]['candles'])):
            bidhigh.append(float(coll[i]['candles'][j]['bid']['h']))
            bidlow.append(float(coll[i]['candles'][j]['bid']['l']))
            bidclose.append(float(coll[i]['candles'][j]['bid']['c']))
            askhigh.append(float(coll[i]['candles'][j]['ask']['h']))
            asklow.append(float(coll[i]['candles'][j]['ask']['l']))
            askclose.append(float(coll[i]['candles'][j]['ask']['c']))               
            midopen.append(float(coll[i]['candles'][j]['mid']['o']))
            midhigh.append(float(coll[i]['candles'][j]['mid']['h']))
            midlow.append(float(coll[i]['candles'][j]['mid']['l']))
            midclose.append(float(coll[i]['candles'][j]['mid']['c']))               
            timestamp.append(coll[i]['candles'][j]['time'])
            volume.append(float(coll[i]['candles'][j]['volume']))
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['bidhigh'] = pd.to_numeric(bidhigh)
    df['bidlow'] = pd.to_numeric(bidlow)
    df['bidclose'] = pd.to_numeric(bidclose)
    df['askhigh'] = pd.to_numeric(askhigh)
    df['asklow'] = pd.to_numeric(asklow)
    df['askclose'] = pd.to_numeric(askclose)
    df['midopen'] = pd.to_numeric(midopen)
    df['midhigh'] = pd.to_numeric(midhigh)
    df['midlow'] = pd.to_numeric(midlow)
    df['midclose'] = pd.to_numeric(midclose)
    df['spread'] = df.askclose - df.bidclose
    df['volume'] = pd.to_numeric(volume)
    
    if not coll[i]['candles'][-1]['complete']:
        df.drop(df.last_valid_index(), inplace=True)

    return df


def get_candles_by_count(instrument, granularity, count, da = daily_alignment):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': count,
              'granularity': granularity,
              'price': 'BAM',
              'alignmentTimezone': 'UTC',
              'dailyAlignment': da}

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    # Assemble Dataframe
    askclose = []
    bidclose = []
    high = []
    low = []
    midhigh = []
    midlow = []    
    midclose = []
    volume = []
    close = []
    timestamp = []
    for i in range(len(coll['candles'])):
        high.append(float(coll['candles'][i]['mid']['h']))
        low.append(float(coll['candles'][i]['mid']['l']))
        midhigh.append(float(coll['candles'][i]['mid']['h']))
        midlow.append(float(coll['candles'][i]['mid']['l']))
        close.append(float(coll['candles'][i]['mid']['c']))
        midclose.append(float(coll['candles'][i]['mid']['c']))
        askclose.append(float(coll['candles'][i]['ask']['c']))
        bidclose.append(float(coll['candles'][i]['bid']['c']))
        volume.append(float(coll['candles'][i]['volume']))
        timestamp.append(coll['candles'][i]['time'])
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['midhigh'] = pd.to_numeric(midhigh)
    df['midlow'] = pd.to_numeric(midlow)
    df['high'] = pd.to_numeric(high)
    df['low'] = pd.to_numeric(low)
    df['close'] = pd.to_numeric(close)
    df['askclose'] = pd.to_numeric(askclose)
    df['midclose'] = pd.to_numeric(midclose)
    df['bidclose'] = pd.to_numeric(bidclose)
    df['spread'] = df.askclose - df.bidclose
    df['volume'] = pd.to_numeric(volume)
    return df



def get_multiple_candles_midclose(instrument_list, granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'M',
              'alignmentTimezone': 'UTC',
              }    
    instrument_dict = {}
    for instrument in instrument_list:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        coll = client.request(r)
        # Remove last candle if not complete.
        if coll['candles'][-1]['complete'] == False:
            coll['candles'].pop()
        instrument_dict[instrument] = float(coll['candles'][-1]['mid']['c'])                
    return instrument_dict


def get_multiple_candles_volume(instrument_list, granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'AB',
              'alignmentTimezone': 'UTC',
              }    
    instrument_dict = {}
    for instrument in instrument_list:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        coll = client.request(r)
        # Remove last candle if not complete.
        if coll['candles'][-1]['complete'] == False:
            coll['candles'].pop()
        instrument_dict[instrument] = float(coll['candles'][-1]['volume'])                  
    return instrument_dict



def get_multiple_candles_spread(instrument_list, granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'AB',
              'alignmentTimezone': 'UTC',
              }    
    instrument_dict = {}
    for instrument in instrument_list:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        coll = client.request(r)
        # Remove last candle if not complete.
        if coll['candles'][-1]['complete'] == False:
            coll['candles'].pop()
        instrument_dict[instrument] = float(coll['candles'][-1]['ask']['c']) \
                                    - float(coll['candles'][-1]['bid']['c'])                  
    return instrument_dict


def get_spreads():
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {
              "instrument": "EUR_USD",
              "period": 57600
              }
    r = labs.Spreads(params=params)
    client.request(r)
    print(r.response)
    

def get_time(granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'M',
              'alignmentTimezone': 'UTC',
              }    
    r = instruments.InstrumentsCandles(instrument='AUD_JPY', params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    return pd.to_datetime(coll['candles'][-1]['time'])
    


def get_orderbook(instrument, time):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'time': time, 'Accept-Datetime-Format': 'RFC3339'}
    r = instruments.InstrumentsOrderBook(instrument=instrument,
                                          params=params)
    a = client.request(r)
    #print(a)#(r.response)
    return a          
 

def create_order(instrument, direction):#, quantity, target, stop, account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    account_id = '101-001-7518331-001'
    client = oandapyV20.API(access_token=client)
    if direction.lower() == 'long':
        units = 35000
    else:
        units = -35000
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
    return int(r.response['orderCreateTransaction']['id'])


#>>> import json
#>>> from oandapyV20 import API
#>>> import oandapyV20.endpoints.trades as trades
#>>> from oandapyV20.contrib.requests import TradeCloseRequest
#>>>
#>>> accountID = "..."
#>>> client = API(access_token=...)
#>>> ordr = TradeCloseRequest(units=10000)
#>>> print(json.dumps(ordr.data, indent=4))
#{
#   "units": "10000"
#}
#>>> # now we have the order specification, create the order request
#>>> r = trades.TradeClose(accountID, tradeID=1234,
#>>>                       data=ordr.data)
#>>> # perform the request
#>>> rv = client.request(r)
#>>> print(rv)
#>>> ...




def close_position(instrument):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    account_id = '101-001-7518331-001'
    client = oandapyV20.API(access_token=client)
    data =  {
              "shortUnits": "ALL"
            }
    r = positions.PositionClose(accountID=account_id,
                                 instrument=instrument, 
                                 data = data)
    client.request(r)
    print(r.response)







def get_open_positions(account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    r = positions.OpenPositions(accountID=account)
    client.request(r)
    p = r.response
    instruments = []
    for position in p['positions']:
        instruments.append(position['instrument'])
    return instruments  


def get_transactions_range(_from, account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {"from": _from,
          "to": _from + 500,
          'type': 'ORDER_FILL'}
    r = trans.TransactionIDRange(accountID=account, params=params)
    client.request(r)
    return r.response

    
def get_most_recent_transaction(account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    r = trades.OpenTrades(accountID=account)
    client.request(r)
    _id = int(r.response['lastTransactionID'])
    return _id


def get_accounts():
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    r = accounts.AccountList()
    client.request(r)
    pprint(r.response['accounts'])
    accounts_collection = []
    for each in r.response['accounts']:
        accounts_collection.append(each['id'])
    return accounts_collection


market = ['AUD_CAD',
        'AUD_CHF',
        'AUD_HKD',
        'AUD_JPY',
        'AUD_NZD',
        'AUD_SGD',
        'AUD_USD',
        'CAD_CHF',
        'CAD_HKD',
        'CAD_JPY',
        'CAD_SGD',
        'CHF_HKD',
        'CHF_JPY',
        'CHF_ZAR',
        'EUR_AUD',
        'EUR_CAD',
        'EUR_CHF',
        'EUR_CZK',
        'EUR_DKK',
        'EUR_GBP',
        'EUR_HKD',
        'EUR_HUF',
        'EUR_JPY',
        'EUR_NOK',
        'EUR_NZD',
        'EUR_PLN',
        'EUR_SEK',
        'EUR_SGD',
        'EUR_TRY',
        'EUR_USD',
        'EUR_ZAR',
        'GBP_AUD',
        'GBP_CAD',
        'GBP_CHF',
        'GBP_HKD',
        'GBP_JPY',
        'GBP_NZD',
        'GBP_PLN',
        'GBP_SGD',
        'GBP_USD',
        'GBP_ZAR',
        'HKD_JPY',
        'NZD_CAD',
        'NZD_CHF',
        'NZD_HKD',
        'NZD_JPY',
        'NZD_SGD',
        'NZD_USD',
        'SGD_CHF',
        'SGD_HKD',
        'SGD_JPY',
        'TRY_JPY',
        'USD_CAD',
        'USD_CHF',
        'USD_CNH',
        'USD_CZK',
        'USD_DKK',
        'USD_HKD',
        'USD_HUF',
        'USD_INR',
        'USD_JPY',
        'USD_MXN',
        'USD_NOK',
        'USD_PLN',
        'USD_SAR',
        'USD_SEK',
        'USD_SGD',
        'USD_THB',
        'USD_TRY',
        'USD_ZAR',
        'ZAR_JPY']


if __name__ == '__main__':
    pass
"""