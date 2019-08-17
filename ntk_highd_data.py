#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to analyze high dimensional data

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1


List of functions/class in ntk_highd_data_to_lowd
highd_data_to_lowd(data, method=2)
'''


# Load Ecube HS data and grab data within range
def highd_data_to_lowd(data, method=2):

    '''
    high dimensional data to low-dimensional
    This is just a interface please modify parameters to match your needs
    highd_data_to_lowd(data, method=2)
    data - high dimensional data
    method (deafult 2) 1: umap, 2 :tsne

    returns embedding
    '''

    if method == 1:
        try:
            import umap
        except ImportError:
            raise ImportError('Run : conda install -c conda-forge umap-learn')

        embedding = umap.UMAP(n_neighbors=40, n_components=2,
                              metric='euclidean', n_epochs=None,
                              learning_rate=1.0, init='spectral',
                              min_dist=0.2, spread=1.0, set_op_mix_ratio=1.0,
                              local_connectivity=1.0, repulsion_strength=1.0,
                              negative_sample_rate=5, transform_queue_size=4.0,
                              a=None, b=None, random_state=None,
                              metric_kwds=None, angular_rp_forest=False,
                              target_n_neighbors=-1,
                              target_metric='categorical',
                              target_metric_kwds=None,
                              target_weight=0.5, transform_seed=42,
                              verbose=True).fit_transform(data[:, 0:])

    elif method == 2:
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError('Run : conda install scikit-learn')
        embedding = TSNE(n_components=2, perplexity=30.0,
                         early_exaggeration=12.0, learning_rate=200.0,
                         n_iter=3000, n_iter_without_progress=300,
                         min_grad_norm=1e-07, metric='euclidean',
                         init='random',
                         verbose=0, random_state=None,
                         method='barnes_hut', angle=0.5
                         ).fit_transform(data[:, 0:])

    return embedding
