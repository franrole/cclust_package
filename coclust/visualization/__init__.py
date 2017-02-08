# -*- coding: utf-8 -*-

"""
The :mod:`coclust.visualization` module provides functions to visualize
different measures or data.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from sklearn.preprocessing import normalize


logger = logging.getLogger(__name__)
plt.style.use('ggplot')


def plot_max_modularities(max_modularities, range_n_clusters):
    """Plot all max modularities obtained after a series of evaluations. The
    best partition is indicated in the graph and main title.

    Parameters
    ----------
    max_modularities: list of float
        Final modularities for all evaluated partitions
    range_n_clusters: list
        Number of clusters for which the algorithm is to be executed

    Example
    -------
    >>> plot_max_modularities(all_max_modularities, range_n_clusters)

    .. plot::

        from coclust.visualization import plot_max_modularities
        from coclust.io.data_loading import load_doc_term_data
        from coclust.evaluation.internal import best_modularity_partition

        path = '../../../datasets/classic3_coclustFormat.mat'
        doc_term_data = load_doc_term_data(path)

        min_cluster_nbr = 2
        max_cluster_nbr = 9
        range_n_clusters = range(min_cluster_nbr, (max_cluster_nbr + 1))

        best_coclustMod_model, all_max_modularities = \
            best_modularity_partition(doc_term_data['doc_term_matrix'],
                                      range_n_clusters, 1)

        plot_max_modularities(all_max_modularities, range_n_clusters)
    """

    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(max_modularities)), 1)
    plt.xticks(np.arange(0, len(max_modularities) + 1, 1))
    labels = range_n_clusters
    ax.set_xticklabels(labels)

    # Plot all max modularities
    plt.plot(max_modularities, marker='o')

    # Set the axis titles
    plt.ylabel("Final Modularity", size=10)
    plt.xlabel("Number of clusters", size=10)

    # Set the axis limits
    plt.xlim(-0.5, (len(max_modularities) - 0.5))
    plt.ylim((min(max_modularities) - 0.05 * min(max_modularities)),
             (max(max_modularities) + 0.05 * max(max_modularities)))

    # Set the main plot titlee
    plt.title("\nMax. modularity for %d clusters (%.4f)\n" %
              (range_n_clusters[max_modularities.index(max(max_modularities))],
               max(max_modularities)), size=12)

    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')

    # Plot a dashed vertical line at best partition
    plt.axvline(np.argmax(max_modularities), linestyle="dashed")
    plt.show()


def plot_intermediate_modularities(model):
    """Plot all intermediate modularities for a model.

    Parameters
    ----------
    model: :class:`coclust.coclustering.CoclustMod`
        Fitted model

    Example
    -------
    >>> plot_intermediate_modularities(model)

    .. plot::

        from coclust.visualization import plot_intermediate_modularities
        from coclust.io.data_loading import load_doc_term_data
        from coclust.coclustering import CoclustMod
        model = CoclustMod(n_clusters=3)
        matrix = load_doc_term_data('../../../datasets/classic3.csv')['doc_term_matrix']
        model.fit(matrix)
        plot_intermediate_modularities(model)
    """

    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(model.modularities) + 1), 1)
    plt.xticks(np.arange(0, len(model.modularities) + 1, 1))
    ax.set_xticklabels(labels)

    # Plot all intermdiate modularities
    plt.plot(model.modularities, marker='o')

    # Set the axis titles
    plt.ylabel("Modularities", size=10)
    plt.xlabel("Iterations", size=10)

    # Set the axis limits
    plt.xlim(-0.5, (len(model.modularities)-0.5))
    plt.ylim((min(model.modularities) - 0.05 * min(model.modularities)),
             (max(model.modularities) + 0.05 * max(model.modularities)))

    # Set the main plot titlee
    plt.title("\nIntermediate modularities for %d clusters\n"
              % (model.n_clusters), size=12)

    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')

    # Plot a dashed horizontal line around max modularity
    plt.axhline(max(model.modularities), linestyle="dashed")
    plt.axhline((max(model.modularities)-model.tol), linestyle="dashed")
    plt.show()


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

        from coclust.visualization import plot_cluster_top_terms
        from coclust.io.data_loading import load_doc_term_data
        from coclust.evaluation.internal import best_modularity_partition

        path = '../../../datasets/classic3_coclustFormat.mat'
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

        from coclust.visualization import plot_cluster_sizes
        from coclust.io.data_loading import load_doc_term_data
        from coclust.coclustering import CoclustMod
        model = CoclustMod(n_clusters=3)
        matrix = load_doc_term_data('../../../datasets/classic3.csv')['doc_term_matrix']
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


def _remove_ticks():
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')


def plot_reorganized_matrix(X, model, precision=0.8, markersize=0.9):
    """Plot the reorganized matrix.

    Parameters
    ----------
    X: matrix
        Data matrix
    model:
        Fitted co-clustering model
    precision: float, optional
        values greater than `precision` will be plotted
    markersize: float
        marker size
    Example
    -------
    >>> plot_reorganized_matrix(X, model)

    .. plot::

        from coclust.visualization import plot_reorganized_matrix
        from coclust.io.data_loading import load_doc_term_data
        from coclust.coclustering import CoclustMod
        path = '../../../datasets/classic3.csv'
        model = CoclustMod(n_clusters=3)
        X = load_doc_term_data(path)['doc_term_matrix']
        model.fit(X)
        plot_reorganized_matrix(X, model)
    """

    row_indices = np.argsort(model.row_labels_)
    col_indices = np.argsort(model.column_labels_)
    X_reorg = X[row_indices, :]
    X_reorg = X_reorg[:, col_indices]
    plt.spy(X_reorg, precision=precision, markersize=markersize)
    _remove_ticks()
    plt.show()


def plot_convergence(criteria, criterion_name, marker='o'):
    """ Plot the convergence of a given criteria.

    Parameters
    ----------
    criteria: array-like
        Criteria values
    criterion_name: str
        Name of the criteria
    marker:
        Marker

    Example
    -------
    >>> plot_convergence(modularities, "Modularity")

    .. plot::

        from coclust.visualization import plot_convergence
        from coclust.io.data_loading import load_doc_term_data
        from coclust.coclustering import CoclustMod
        path = '../../../datasets/classic3.csv'
        model = CoclustMod(n_clusters=3)
        X = load_doc_term_data(path)['doc_term_matrix']
        model.fit(X)
        plot_convergence(model.modularities, "Modularity")
    """
    plt.plot(criteria, marker=marker)
    plt.ylabel(criterion_name)
    plt.xlabel('Iterations')
    _remove_ticks()
    plt.show()


def plot_confusion_matrix(cm, colormap=plt.get_cmap(), labels='012'):
    """Plot a confusion matrix.

    Parameters
    ----------
    cm: array
        Confusion matrix
    colormap: :class:`matplotlib.colors.Colormap`
        Color map
    labels:
        Labels

    Example
    -------
    >>> plot_confusion_matrix(cm)

    .. plot::

        from sklearn.metrics import confusion_matrix
        from coclust.visualization import plot_confusion_matrix
        from coclust.io.data_loading import load_doc_term_data
        from coclust.coclustering import CoclustMod
        path = '../../../datasets/classic3_coclustFormat.mat'
        model = CoclustMod(n_clusters=3)
        data = load_doc_term_data(path)
        X = data['doc_term_matrix']
        labels = data['doc_labels']
        model.fit(X)
        cm = confusion_matrix(labels, model.row_labels_)
        plot_confusion_matrix(cm)

    """

    conf_arr = np.array(cm)

    norm_conf_arr = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf_arr.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf_arr), cmap=colormap,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in np.arange(width):
        for y in np.arange(height):
            ax.annotate(str(conf_arr[x][y]),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        path_effects=[PathEffects.withStroke(linewidth=3,
                                                             foreground="w",
                                                             alpha=0.7)])

    fig.colorbar(res)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    _remove_ticks()
    plt.show()


def plot_delta_kl(model, colormap=plt.get_cmap(),
                  labels='012'):
    """Plot the delta values of the Information-Theoretic Co-clustering.

    Parameters
    ----------
    model: :class:`coclust.coclustering.CoClustInfo`
        The fitted co-clustering model
    colormap: :class:`matplotlib.colors.Colormap`
        Color map
    labels:
        Labels

    Example
    -------
    >>> plot_delta_kl(model)

    .. plot::

        from coclust.visualization import plot_delta_kl
        from coclust.io.data_loading import load_doc_term_data
        from coclust.coclustering import CoclustInfo
        path = '../../../datasets/classic3.csv'
        model = CoclustInfo(n_row_clusters=3, n_col_clusters=3)
        X = load_doc_term_data(path)['doc_term_matrix']
        model.fit(X)
        plot_delta_kl(model)
    """
    delta = model.delta_kl_

    delta_arr = np.round(np.array(delta), decimals=3)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(delta_arr), cmap=colormap,
                    interpolation='nearest')

    width, height = delta_arr.shape

    for x in np.arange(width):
        for y in np.arange(height):
            nb_docs = len(model.get_row_indices(x))
            nb_terms = len(model.get_col_indices(y))
            ax.annotate(str(delta_arr[x][y]) + "\n(%d,%d)" %
                        (nb_docs, nb_terms),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        path_effects=[PathEffects.withStroke(linewidth=3,
                                                             foreground="w",
                                                             alpha=0.7)])

    fig.colorbar(res)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])

    ax = plt.gca()
    ax.grid(False)

    _remove_ticks()
    plt.show()
