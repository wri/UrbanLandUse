import pandas as pd
import numpy as np

# loss function stuff

# get category counts from chip catalog
def calc_category_counts(df,remapping='standard'):
    df_counts = df['lulc'].value_counts()
    cats = df_counts.index.values
    cats.sort()
    counts_dict = {}
    for c in cats:
        counts_dict[c] = df_counts.loc[c]
    if remapping is None:
        return counts_dict
    else:
        if isinstance(remapping, str):
            if remapping.lower() == 'standard' or remapping.lower() == 'residential':
                remapping = {0:0,1:1,2:2,3:2,4:2,5:2,6:6}
            else:
                raise ValueError('Unrecognized remapping identifier: ',remapping)
        assert isinstance(remapping, dict)
        recount_dict = {}
        for k in sorted(counts_dict.iterkeys()):
            if remapping[k] not in recount_dict:
                recount_dict[remapping[k]] = 0
            recount_dict[remapping[k]] += counts_dict[k]
        return recount_dict

# calculate weights from category counts
def calc_category_weights(category_counts,log=False,mu=1.0):
    assert isinstance(category_counts,dict)
    n_samples = sum(category_counts.values())
    category_weights = {}
    for k in sorted(category_counts.iterkeys()):
        # for the moment skipping scenario where one or more cats have zero samples
        count = category_counts[k]
        score = n_samples / float(count)
        if log:
            score = math.log(mu*score)
        category_weights[k] = score
    return category_weights

# normalize provided category weights
def normalize_category_weights(category_weights,max_score=None):
    assert isinstance(category_weights,dict)
    min_weight = min(category_weights.values())
    normalized_weights = {}
    for k in sorted(category_weights.iterkeys()):
        normalized_weights[k] = category_weights[k]/min_weight
        if max_score is not None:
            normalized_weights[k] = min(max_score, normalized_weights[k])
    return normalized_weights

# single function to wrap previous three: catalog -> normalized weights
def generate_category_weights(df,remapping='standard',log=False,mu=1.0,max_score=None):
    cat_counts = calc_category_counts(df,remapping=remapping)
    cat_weights = calc_category_weights(cat_counts,log=log,mu=1.0)
    cat_normed = normalize_category_weights(cat_weights,max_score=max_score)
    return cat_normed