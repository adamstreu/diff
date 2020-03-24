import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, '/Users/user/Desktop/diff')
from libraries.oanda import get_candles_bid_close

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
_from = '2019-06-01T00:00:00Z'
_to = '2020-01-01T00:00:00Z'
granularity = 'M1'

# Pairs
pairs_index = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD',
                'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD',
                'CHF_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP',
                'EUR_HKD', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_AUD',
                'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD',
                'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD',
                'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD',
                'USD_JPY']

# Currencies
currencies_index = list(set('_'.join(pairs_index).split('_')))
currencies_index.sort()



# Call Candles and assemble in to appropriate Dataframe
##############################################################################
if False:
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
##############################################################################
if False
    def conversion_dict(pairs, currencies):
        '''
        Calculate currencies based on pairs
        Return Currency dictionary based on full set of instrument prices
        Input: pairs:      dictionary:       currenct value of each pair 
               currencies: dictionary:       values can be zero
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
    
    currency_dict = dict(zip(currencies_index, [0] * len(currencies_index)))
    currencies = pd.DataFrame(columns = currencies_index)
    for index, row in pairs.iterrows():
        p = row.to_dict()
        currencies.loc[index] = conversion_dict(p, currency_dict)
        
 
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
if True:
    '''
    for a currency:
        Bin amount differences were positive 
        for each bin:
            how many times did it happen?
            what was the percent of time that the next days price was higher?
        
    '''
    
    # Only choose those days where the difference was greater than zero
    
    # This is wrong.
    
    '''
    curr = 'EUR_USD'
    greater_than_0_index = differences[curr].values > 0
    diff_greater = differences[curr].values[greater_than_0_index]
    pair_greater = pairs[curr].values[greater_than_0_index]
    bins = pd.cut(diff_greater, 50, labels=False)
    
    
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
    '''
    
    
    
    
    # WY DOES E_USD RETURN ON 300 RESUTLS  ?? ? ? 
    # ANYWAY - TIRED NOW.
    
    
    
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
    


    





















