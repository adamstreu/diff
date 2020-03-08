import numpy as np


def conversion_old(pairs, currencies):
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