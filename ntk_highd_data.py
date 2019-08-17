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
highd_data_umap(data, n_neighbors=40, n_components=2,
                metric='euclidean', min_dist=0.2,
                verbose=True):
highd_data_tsne(data, perplexity=30.0, n_components=2,
                metric='euclidean', n_iter=3000,
                verbose=True):
'''


# Load Ecube HS data and grab data within range
def highd_data_umap(data, n_neighbors=40, n_components=2,
                    metric='euclidean', min_dist=0.2,
                    verbose=True):

    '''
    high dimensional data to low-dimensional
    This is just a interface please modify parameters to match your needs
    highd_data_to_lowd(data, method=2)
    data - high dimensional data
    https://umap-learn.readthedocs.io/en/latest/

    returns embedding
    '''

    try:
        import umap
    except ImportError:
        raise ImportError('Run : conda install -c conda-forge umap-learn')

    embedding = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                          metric=metric, n_epochs=None,
                          learning_rate=1.0, init='spectral',
                          min_dist=min_dist, spread=1.0, set_op_mix_ratio=1.0,
                          local_connectivity=1.0, repulsion_strength=1.0,
                          negative_sample_rate=5, transform_queue_size=4.0,
                          a=None, b=None, random_state=None,
                          metric_kwds=None, angular_rp_forest=False,
                          target_n_neighbors=-1,
                          target_metric='categorical',
                          target_metric_kwds=None,
                          target_weight=0.5, transform_seed=42,
                          verbose=verbose).fit_transform(data[:, 0:])

    return embedding


def highd_data_tsne(data, perplexity=30.0, n_components=2,
                    metric='euclidean', n_iter=3000,
                    verbose=True):

    '''
    high dimensional data to low-dimensional
    This is just a interface please modify parameters to match your needs
    highd_data_to_lowd(data, method=2)
    data - high dimensional data
    scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    returns embedding
    '''

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError('Run : conda install scikit-learn')
    embedding = TSNE(n_components=n_components, perplexity=perplexity,
                     early_exaggeration=12.0, learning_rate=200.0,
                     n_iter=n_iter, n_iter_without_progress=300,
                     min_grad_norm=1e-07, metric=metric,
                     init='random',
                     verbose=0, random_state=None,
                     method='barnes_hut', angle=0.5
                     ).fit_transform(data[:, 0:])

    return embedding
