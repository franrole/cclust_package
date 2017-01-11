# -*- coding: utf-8 -*-

"""
The :mod:`coclust.visualization.termClusters` module provides functions to
visualize cluster of terms.
"""


import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


logger = logging.getLogger(__name__)


def plot_cluster_top_terms(in_data, all_terms, nb_top_terms, model):
    """Plot the top terms for each cluster.

    Parameters
    ----------
    in_data : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
    all_terms: list of string
        list of all terms from the original data set
    nb_top_terms: int
        number of top terms to be displayed per cluster
    model: :class:`coclust.coclustering.BaseDiagonalCoclust`
        a co-clustering model


    Example
    -------
    >>> plot_cluster_top_terms(in_data, all_terms, nb_top_terms, model)

    .. plot::

        from coclust.visualization.termClusters import plot_cluster_top_terms
        from coclust.io.io import load_doc_term_data
        from coclust.evaluation.partitionEvaluation import best_modularity_partition

        path = '../../datasets/classic3_coclustFormat.mat'
        doc_term_data = load_doc_term_data(path)

        min_cluster_nbr = 2
        max_cluster_nbr = 9
        range_n_clusters = range(min_cluster_nbr, (max_cluster_nbr + 1))

        best_coclustMod_model, _ = \
            best_modularity_partition(doc_term_data['doc_term_matrix'],
                                      range_n_clusters, 1)
        n_terms = 10
        plot_cluster_top_terms(doc_term_data['doc_term_matrix'],
                               doc_term_data['term_labels'],
                               n_terms,
                               best_coclustMod_model)

    """

    if all_terms is None:
        logger.warning("Term labels cannot be found. Use input argument "
                       "'term_labels_filepath' in function "
                       "'load_doc_term_data' if term labels are available.")
        return

    x_label = "number of occurences"
    plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.200)
    plt.suptitle("      Top %d terms" % nb_top_terms, size=15)
    number_of_subplots = model.n_clusters

    for i, v in enumerate(range(number_of_subplots)):
        # Get the row/col indices corresponding to the given cluster
        row_indices, col_indices = model.get_indices(v)
        # Get the submatrix corresponding to the given cluster
        cluster = model.get_submatrix(in_data, v)
        # Count the number of each term
        p = cluster.sum(0)
        t = p.getA().flatten()
        # Obtain all term names for the given cluster
        tmp_terms = np.array(all_terms)[col_indices]
        # Get the first n terms
        max_indices = t.argsort()[::-1][:nb_top_terms]

        pos = np.arange(nb_top_terms)

        v = v + 1
        ax1 = plt.subplot(number_of_subplots, 1, v)
        ax1.barh(pos, t[max_indices][::-1])
        ax1.set_title("Cluster %d (%d terms)" % (v, len(col_indices)), size=11)

        plt.yticks(.4 + pos, tmp_terms[max_indices][::-1], size=9.5)
        plt.xlabel(x_label, size=9)
        plt.margins(y=0.05)
        #_remove_ticks()
        plt.tick_params(axis='both', which='both', bottom='on', top='off',
                        right='off', left='off')

    # Tight layout often produces nice results
    # but requires the title to be spaced accordingly
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.show()


def get_term_graph(X, model, terms, n_cluster, n_top_terms=10, n_neighbors=2,
                   stopwords=[]):
    """Get a graph of terms.

    Parameters
    ----------
    X:
        input matrix
    model: :class:`coclust.coclustering.BaseDiagonalCoclust`
        a co-clustering model
    terms: list of string
        list of terms
    n_cluster: int
        Id of the cluster
    n_top_terms: int, optional, default: 10
        Number of terms
    n_neighbors: int, optional, default: 2
        Number of neighbors
    stopwords: list of string, optional, default: []
        Words to remove

    """

    # The dictionary to be returned
    graph = {"nodes": [], "links": []}

    if terms is None:
        logger.warning("Term labels cannot be found. Use input argument "
                       "'term_labels_filepath' in function "
                       "'load_doc_term_data' if term labels are available.")
        return graph

    # get submatrix and local kist of terms
    row_indices, col_indices = model.get_indices(n_cluster)
    cluster = model.get_submatrix(X, n_cluster)
    terms = np.array(terms)[col_indices]

    # identify most frequent words
    p = cluster.sum(0)
    t = p.getA().flatten()
    top_term_indices = t.argsort()[::-1][:n_top_terms]

    # create tt sim matrix
    cluster_norm = normalize(cluster, norm='l2', axis=0, copy=True)
    sim = cluster_norm.T * cluster_norm

    # to be able to compute the final index of a neighbor which is also a
    # top term
    d = {t: i for i, t in enumerate(top_term_indices)}

    # identify best neighbors of frequent terms
    pointed_by = dict()
    graph = {"nodes": [], "links": []}
    all_neighbors = set()
    links = []
    for idx_tt, t in enumerate(top_term_indices):
        best_neighbors = np.argsort(sim.toarray()[t])[::-1][:n_neighbors]
        for n in best_neighbors:
            if len(stopwords) > 0:
                if terms[n] in stopwords:
                    continue
            if (terms[n].endswith("ed") or terms[n].endswith("ing") or
                    terms[n].endswith("ly")):
                continue

            # if  terms[dico_tt[n]].lower() in stopwords: continue
            if t == n:
                continue
            if n in top_term_indices and t in pointed_by.get(n, []):
                # t was already pointed by n
                continue
            if n in top_term_indices:
                # n will be able to check that is has been pointed by t
                pointed_by.setdefault(t, []).append(n)
            else:
                # a "pure" neighbor
                all_neighbors.add(n)
            if n in top_term_indices:
                # n is a (not yet handled) top term. Lookup in dictionary to
                # find the d3 index.
                # Also record original indices using couples.
                links.append(((idx_tt, t), (d[n], n)))
            else:
                # n is a pure neighbor. Compute its d3 index by an addition
                # use indices suitable for d3 links
                links.append(((idx_tt, t),
                              (len(top_term_indices) + len(all_neighbors) - 1,
                               n)))

    for top_term in top_term_indices:
        graph["nodes"].append({"name": terms[top_term], "group": 0})

    for neighbor in all_neighbors:
        graph["nodes"].append({"name": terms[neighbor], "group": 1})

    for a, b in links:
        graph["links"].append({"source": a[0],
                               "target": b[0],
                               "value": sim[a[1], b[1]]})
    return graph


def plot_cluster_sizes(model):
    """Plot the sizes of the clusters.

    Parameters
    ----------
    model: :class:`coclust.coclustering.BaseDiagonalCoclust`
        a co-clustering model

    Example
    -------
    >>> plot_cluster_sizes(model)

    .. plot::

        from coclust.visualization.termClusters import plot_cluster_sizes
        from coclust.io.io import load_doc_term_data
        from coclust.coclustering.CoclustMod import CoclustMod
        model = CoclustMod(n_clusters=3)
        matrix = load_doc_term_data('../../datasets/classic3.csv')['doc_term_matrix']
        model.fit(matrix)
        plot_cluster_sizes(model)

    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    prop_list = list(plt.rcParams['axes.prop_cycle'])
    colors = [prop_list[0]['color'], prop_list[1]['color']]
    x = []
    y = []
    for i in range(model.n_clusters):
        number_of_rows, number_of_columns = model.get_shape(i)
        x.append(number_of_rows)
        y.append(number_of_columns)
    data = [x, y]
    shift = .8 / len(data * 2)
    location = np.arange(model.n_clusters)
    legend_rects = []
    for i in range(2):
        cols = ax.bar(location + i * shift, data[i], width=shift,
                      color=colors[i % len(colors)], align='center')
        legend_rects.append(cols[0])
        for c in cols:
            h = c.get_height()
            ax.text(c.get_x() + c.get_width() / 2., h + 5, '%d' % int(h),
                    ha='center', va='bottom')
    ax.set_xticks(location + (shift / 2.))
    ax.set_xticklabels(['coclust-' + str(i) for i in range(model.n_clusters)])
    plt.xlabel('Co-clusters')
    plt.ylabel('Sizes')
    plt.tight_layout()
    ax.legend(legend_rects, ('Rows', 'Columns'))

    #_remove_ticks()
    plt.tick_params(axis='both', which='both', bottom='on', top='off',
                    right='off', left='off')
    plt.show()
