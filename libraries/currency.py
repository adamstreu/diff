import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.oanda import get_tradable_instruments
from libraries.oanda import get_candles_bid_close


'''
Get Values
Solve for usd.
Use that to compute remainder.
Compare with pair.
'''

# get All TRadeable Instruments
if False:
    instruments = get_tradable_instruments()
    pairs = [x['name'] for x in instruments['instruments']]
    currencies = list(set('_'.join(pairs).split('_')))
    currencies.sort()
    matrix = np.zeros((len(currencies), len(currencies)))
    
    '''
    for r in range(len(currencies)):
        for c in range(len(currencies)):
            # Try one iteration
            if str(currencies[c]) + '_' + str(currencies[r]) in pairs:
                matrix[r, c] = 1
            if str(currencies[r]) + '_' + str(currencies[c]) in pairs:
                matrix[r, c] = 1        
    df = pd.DataFrame(matrix, columns = currencies)
    df.insert(0, 'currencies', currencies)
    df = df.set_index('currencies', drop=True)
    '''
    



# Get Candles
if False:
    # Candle Parameters
    _from = '2019-12-25T00:00:00Z'
    _to = '2020-01-01T00:00:00Z'
    granularity = 'D'
    df = get_candles_bid_close(pairs[0], granularity, _from, _to)
    df = df.drop('volume', axis=1)
    df = df.set_index('timestamp', drop=True)
    for pair in pairs[1:]:
        _df = get_candles_bid_close(pair, granularity, _from, _to)
        _df = _df.set_index('timestamp', drop=True)
        _df = _df.drop('volume', axis=1)
        df = df.join(_df, how='inner')
    # Fill Forward missing prices, drop top row if any nans remain
    df = df.fillna(method='ffill').dropna()




# Solve for all Pairs on USD
if True:    
    
    currency_dict = dict(zip(currencies, [0] * len(currencies)))
    pairs_dict = df.iloc[0].to_dict()

    
    pair_subset = []
    for pair in list(pairs_dict.keys()):
        if 'USD' == pair.split('_')[0]:
            pair_subset.append(1 / pairs_dict[pair])
        if 'USD' == pair.split('_')[1]:
            pair_subset.append(    pairs_dict[pair])
    currency_dict['USD'] = 1 / (np.array(pair_subset).sum() + 1 )


    # Compute Remiander of prices soley using usd
    usd_subset = [pair for pair in pairs if 'USD' in pair.split('_') ]
    for pair in usd_subset:
        currency = pair.replace('USD', '').replace('_', '')
        if pair.split('_')[0] == 'USD':
            currency_dict[currency] = currency_dict['USD'] / pairs_dict[pair]
        else:
            currency_dict[currency] = currency_dict['USD'] * pairs_dict[pair]
    prices = np.array(list(currency_dict.values()))
    


    
    
    
    
            
    
    
    
    
    
    
    








def conversion_dict(pairs, currencies):
    '''
    Calculate currencies based on pairs
    Return Currency dictionary based on full set of instrument prices
    '''
        
    for currency in currencies.keys():
        # If pair contains currency, add it to the subset, making sure that
        # It is added with the currency as the denominator
        pair_subset = []
        for pair in list(pairs.keys()):
            if currency == pair.split('_')[0]:
                pair_subset.append(1 / pairs[pair])
            if currency == pair.split('_')[1]:
                pair_subset.append(    pairs[pair])
        currencies[currency] = 1 / (np.array(pair_subset).sum() + 1 )
    return currencies 


def conversion(pairs, currency_subsets):
    currencies_dict = dict(zip(currency_subsets.keys(), [1] * len(currency_subsets.keys())))
    for currency in currency_subsets.keys():
        denominator = 0
        for i in currency_subsets[currency]['inverse']:
            denominator += (1 / pairs[i])
        for i in currency_subsets[currency]['provided']:
            denominator += pairs[i]
        currencies_dict[currency] = 1 / denominator
    return currencies_dict