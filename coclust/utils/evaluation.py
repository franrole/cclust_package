# -*- coding: utf-8 -*-

"""
Evaluation
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

from __future__ import print_function
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

plt.style.use('ggplot')


def _remove_ticks():
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')


def plot_criterion(values, ylabel):
    plt.plot(values, marker='o')
    plt.ylabel(ylabel)
    plt.xlabel('Iterations')
    _remove_ticks()
    plt.show()


def plot_reorganized_matrix(X, model, precision=0.8, markersize=0.9):
    row_indices = np.argsort(model.row_labels_)
    col_indices = np.argsort(model.column_labels_)
    X_reorg = X[row_indices, :]
    X_reorg = X_reorg[:, col_indices]
    plt.spy(X_reorg, precision=precision, markersize=markersize)
    _remove_ticks()
    plt.show()


def plot_convergence(criteria, criterion_name, marker='o'):
    plt.plot(criteria, marker=marker)
    plt.ylabel(criterion_name)
    plt.xlabel('Iterations')
    _remove_ticks()
    plt.show()


def plot_confusion_matrix(cm, colormap=plt.get_cmap(), labels='012'):
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


def plot_delta_kl(delta, model, colormap=plt.get_cmap(),
                  labels='012'):

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


def plot_top_terms(model, X, terms, n_cluster, n_terms=10,
                   x_label="number of occurences"):
    row_indices, col_indices = model.get_indices(n_cluster)
    cluster = model.get_submatrix(X, n_cluster)

    p = cluster.sum(0)

    terms = np.array(terms)[col_indices]

    t = p.getA().flatten()

    n = n_terms
    max_indices = t.argsort()[::-1][:n]

    plt.figure()
    pos = np.arange(n) + .5

    plt.barh(pos, t[max_indices][::-1])
    plt.yticks(.4 + pos, terms[max_indices][::-1])

    plt.xlabel(x_label)
    plt.margins(y=0.05)
    _remove_ticks()
    plt.show()


def plot_cluster_sizes(model):
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

    _remove_ticks()
    plt.show()


def print_NMI_and_ARI(true_labels, predicted_labels):
    print("NMI:", nmi(true_labels, predicted_labels))
    print("ARI:", adjusted_rand_score(true_labels, predicted_labels))


def get_term_graph(X, model, terms, n_cluster, n_top_terms=10, n_neighbors=2,
                   stopwords=[]):
    # The dictionary to be returned
    graph = {"nodes": [], "links": []}

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


def accuracy(X, nb_clusters, true_row_labels, predicted_row_labels):
    try:
        accuracy = _true_accuracy(X, nb_clusters, true_row_labels,
                                  predicted_row_labels)
        return accuracy
    except ImportError as e:
        logging.error(e)
        logging.error("Fallback to approximate accuracy, install Munkres for "
                      "true accuracy.")
        return _approximate_accuracy(X, nb_clusters, true_row_labels,
                                     predicted_row_labels)


def _true_accuracy(X, nb_clusters, true_row_labels, predicted_row_labels):
    from munkres import Munkres, make_cost_matrix
    m = Munkres()
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    s = np.max(cm)
    cost_matrix = make_cost_matrix(cm, lambda cost: s - cost)
    indexes = m.compute(cost_matrix)
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value

    return(total * 1. / np.sum(cm))


def _approximate_accuracy(X, nb_clusters, true_row_labels,
                          predicted_row_labels):
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    n_rows = len(true_row_labels)
    total = 0
    for i in range(nb_clusters):
        if len(cm) == 0:
            break
        max_value = np.amax(cm)
        r_indices, c_indices = np.where(cm == max_value)
        total = total + max_value
        cm = np.delete(cm, r_indices[0], 0)
        cm = np.delete(cm, c_indices[0], 1)
    accuracy = (total) / (n_rows * 1.)

    return accuracy
