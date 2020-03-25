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


"""







# Call Candles and assemble in to appropriate Dataframe, Calculate Indicators
##############################################################################
if False:
    
    # Pairs
    instruments = get_tradable_instruments()
    pairs_index = [x['name'] for x in instruments['instruments']]
    
    # Currencies
    currencies_index = list(set('_'.join(pairs_index).split('_')))
    currencies_index.sort()
    currency_dict = dict(zip(currencies_index, [0] * len(currencies_index)))

    # Candle Parameters
    _from = '2019-12-01T00:00:00Z'
    _to = '2020-01-01T00:00:00Z'
    granularity = 'H1'

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
            
        
        
        
        
    
    
    
    
    
# Streaming Build
##############################################################################

# Liek from Q
_pairs = np.array(pairs.iloc[0].to_list())

# Create Subset
usd_inverse = []
usd_provided = []
for pair in list(pairs_index):
    if 'USD' == pair.split('_')[0]:
        usd_inverse.append(True)        
        usd_provided.append(False)
    elif 'USD' == pair.split('_')[1]:
        usd_inverse.append(False)        
        usd_provided.append(True)
    else:
        usd_inverse.append(False)        
        usd_provided.append(False)
usd_provided = np.array(usd_provided)
usd_inverse = np.array(usd_inverse)

p = [x.replace('_USD', '') for x in np.array(pairs_index)[usd_provided]]
i = [x.replace('USD_', '') for x in np.array(pairs_index)[usd_inverse]]
currencies_index = ['USD'] + p + i




# IN LOOP # # # #


# Calculate USD
usd = 1 / (1 + _pairs[usd_provided].sum() + (1 / _pairs[usd_inverse]).sum())

# Build Currencies Index
_currencies = np.array([usd] + list(usd * _pairs[usd_provided]) + list(usd / _pairs[usd_inverse]) )

# Build Calculated.



# Make currency Mask
currency_nominator_mask = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
currency_denominator_mask = np.zeros((len(currencies_index), len(pairs_index))).astype(bool)
for i in range(len(pairs_index)):
    nom = pairs_index[i].split('_')[0]
    den = pairs_index[i].split('_')[1]
    currency_nominator_mask[currencies_index.index(nom), i] = True
    currency_denominator_mask[currencies_index.index(den), i] = True


# Build Currency mask out
a = np.tile(_currencies, (len(pairs_index), 1)).T
_calculated = (a * currency_nominator_mask).sum(0) / (a * currency_denominator_mask).sum(0)

# Calculate Differences
_pairs - _calculated


'''
# Create usd array to multiply by _currencies
a = [_currencies[0]] + [_currencies[0]] * usd_provided.sum() + [1 / _currencies[0]] * usd_inverse.sum()
a = np.array(a)
'''

'''
[EUR] = [EUR_USD] * [   USD   ] 
[CAD] = [USD_CAD] * [ 1 / USD ]



# Inverse and normal denominators for conversion calculation
inverse = np.zeros((len(self.currencies_index), len(self.pairs_index))).astype(bool)
given = np.zeros((len(self.currencies_index), len(self.pairs_index))).astype(bool)
for r in range(inverse.shape[0]):
    for c in range(inverse.shape[1]):
        if self.currencies_index[r] == self.pairs_index[c].split('_')[0]:
            inverse[r, c] = True
        if self.currencies_index[r] == self.pairs_index[c].split('_')[1]:
        given[r, c] = True
'''               




'''
# Calculate newest and update currencies
a = np.tile(_pairs, (len(self.currencies_index), 1))
_currencies = 1 / ((a * given).sum(1) + ((1 / a) * inverse).sum(1) + 1)
'''


















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
    


    





















