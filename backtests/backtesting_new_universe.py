import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.oanda import get_candles_bid_close
from libraries.oanda import get_tradable_instruments

"""
Backtesting currencies and Indicators and Look for major trends:
        
    1.  Call candles for over multiple years
    2.  Calculate Currencies and Indicator.
    3.  go through Data and Graph.
    4.  Report Findings.
    5.  Save in convenient fashion for reuse - answerting different questions.
    6.  Save pulled candles onto disk so do not need to call Oanda every time.


Questions:
    
    What is volume in candle?


"""


# Candle Parameters
_from = '2019-12-01T00:00:00Z'
_to = '2020-01-01T00:00:00Z'
granularity = 'H1'

# Pairs
instruments = get_tradable_instruments()
pairs_index = [x['name'] for x in instruments['instruments']]

# Currencies
currencies_index = list(set('_'.join(pairs_index).split('_')))
currencies_index.sort()
currency_dict = dict(zip(currencies_index, [0] * len(currencies_index)))



# Call Candles and assemble in to appropriate Dataframe, Calculate Indicators
##############################################################################
if True:
    
    # GEt candles
    pairs = get_candles_bid_close(pairs_index[0], granularity, _from, _to)
    pairs = pairs.drop('volume', axis=1)
    pairs = pairs.set_index('timestamp', drop=True)
    for pair in pairs_index[1:]:
        df = get_candles_bid_close(pair, granularity, _from, _to)
        df = df.set_index('timestamp', drop=True)
        df = df.drop('volume', axis=1)
        pairs = pairs.join(df, how='inner')
    # Fill Forward missing prices, drop top row if any nans remain
    pairs = pairs.fillna(method='ffill').dropna()

    # Calculate Currencies and Indicator for each timestamp
    def conversion(pairs_dict, currency_dict):
        # Calculate USD Value
        pair_subset = []
        for pair in list(pairs_dict.keys()):
            if 'USD' == pair.split('_')[0]:
                pair_subset.append(1 / pairs_dict[pair])
            if 'USD' == pair.split('_')[1]:
                pair_subset.append(    pairs_dict[pair])
        currency_dict['USD'] = 1 / (np.array(pair_subset).sum() + 1 )
        # Compute remaining Currencies using USD
        usd_subset = [pair for pair in pairs if 'USD' in pair.split('_') ]
        for pair in usd_subset:
            currency = pair.replace('USD', '').replace('_', '')
            if pair.split('_')[0] == 'USD':
                currency_dict[currency] = currency_dict['USD'] / pairs_dict[pair]
            else:
                currency_dict[currency] = currency_dict['USD'] * pairs_dict[pair]
        # Return Currencies Dicctionary
        return currency_dict
    
    # Calculate Currrencies from each pairs row
    currencies = pd.DataFrame(columns = currencies_index)
    for index, row in pairs.iterrows():
        p = row.to_dict()
        currencies.loc[index] = conversion(p, currency_dict)
                
    # Rcalculate Pairs based on currency prices
    calculated = pairs.copy()
    for column in calculated:
        left = column.split('_')[0]
        right = column.split('_')[1]
        calculated[column] = currencies[left] / currencies[right]
        
    # Calculate Difference Indicator   
    differences = calculated - pairs
            
        
        
        
        
    
    
    
    
    
    







# Difference test 1
##############################################################################
if False:
 
      
    curr = 'EUR_USD'
    bins = pd.cut(differences[curr], 50, labels=False)
    greater_than_0_index = calculated[curr] > pairs[curr]
    diff_greater = differences[curr].values[greater_than_0_index]
    pair_greater = pairs[curr].values[greater_than_0_index]


    # Simple question - if it was higher, was the next ( what about the last)
    _greater = pairs[curr][greater_than_0_index].values < pairs[curr][np.roll(greater_than_0_index, 1)].values
    print(_greater.mean())
    
    # The answer is no - what about if we bin the sizes
    
    bins = pd.cut(differences[curr][greater_than_0_index].values, 50, labels=False)
    
    
    
    _bins = []
    _count = []
    _greater_than = []
    for i in range(bins.max() + 1):
        _bins.append(i)
        _count.append((bins == i).sum())
        # greater_than = (pairs[curr].values[bins == i] < \
        #                 pairs[curr].values[np.roll(bins == i, 1)]).mean()
        greater_than = (pair_greater[bins == i] < \
                        pair_greater[np.roll(bins == i, 1)]).mean()
        _greater_than.append(greater_than)
        
    plt.scatter(_count, _greater_than)
    


    





















