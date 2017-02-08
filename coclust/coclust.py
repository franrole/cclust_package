# -*- coding: utf-8 -*-

"""
coclusts script
"""

# Author: Francois Role <francois.role@gmail.com>
#         Stanislas Morbieu <stanislas.morbieu@gmail.com>

# License: BSD 3 clause

from __future__ import print_function

import argparse
import logging
import sys

import numpy as np
import scipy.sparse as sp


def get_parsers():
    main_parser = argparse.ArgumentParser(prog='coclust')

    subparsers = main_parser.add_subparsers(help='choose the algorithm to use',
                                            dest='subparser_name')

    coclust_nb_parser = argparse.ArgumentParser(prog='coclust-nb')

    # create the parser for the "modularity" command
    parser_modularity = subparsers.add_parser('modularity',
                                              help='use the modularity based \
                                              algorithm')

    # create the parser for the "specmodularity" command
    parser_spec_modularity = subparsers.add_parser('specmodularity',
                                                   help='use the spectral \
                                                   modularity based algorithm')

    # create the parser for the "info" command
    parser_info = subparsers.add_parser('info')

    parser_list = [parser_modularity, parser_spec_modularity, parser_info,
                   coclust_nb_parser]
    for parser in parser_list:
        # input args
        input_group = parser.add_argument_group('input')
        input_group.add_argument('INPUT_MATRIX', help='matrix file path')
        input_params_group = input_group.add_mutually_exclusive_group()
        input_params_group.add_argument('-k', '--matlab_matrix_key',
                                        default=None, help='if not set, csv \
                                        input is considered')
        input_params_group.add_argument('-sep', '--csv_sep', default=",",
                                        help='if not set, "," is considered; \
                                        use "\\t" for tab-separated values')

        # output args
        output_group = parser.add_argument_group('output')
        output_group.add_argument('--output_row_labels', help='file path for \
                                  the predicted row labels')
        output_group.add_argument('--output_column_labels', help='file path \
                                  for the predicted column labels')
        if parser == parser_modularity or parser == info:
            output_group.add_argument('--output_fuzzy_row_labels', default=2,
                                      type=int, help='file \
                                      path for the predicted fuzzy row labels')
            output_group.add_argument('--output_fuzzy_column_labels',
                                      default=2, type=int,
                                      help='file path for the predicted fuzzy \
                                      column labels')
            output_group.add_argument('--convergence_plot', help='file path \
                                      for the convergence plot')
        output_group.add_argument('--reorganized_matrix', help='file path for \
                                  the reorganized matrix')

        # parameter args
        parameters_group = parser.add_argument_group('algorithm parameters')
        if parser == coclust_nb_parser:
            parameters_group.add_argument('--from', default=2, type=int,
                                          help='minimum number of co-clusters')
            parameters_group.add_argument('--to', default=10, type=int,
                                          help='maximum number of co-clusters')
        elif parser == parser_info:
            parameters_group.add_argument('-K', '--n_row_clusters',
                                          help='number of row clusters',
                                          default=2, type=int)
            parameters_group.add_argument('-L', '--n_col_clusters',
                                          help='number of column clusters',
                                          default=2, type=int)
        else:
            parameters_group.add_argument('-n', '--n_coclusters',
                                          help='number of co-clusters',
                                          default=2, type=int)
        parameters_group.add_argument('-m', '--max_iter', type=int, default=15,
                                      help='maximum number of iterations')
        parameters_group.add_argument('-e', '--epsilon', type=float,
                                      default=1e-9, help='stop if the \
                                      criterion (modularity) variation in an \
                                      iteration is less than EPSILON')

        # init and runs args
        init_group = parameters_group.add_mutually_exclusive_group()
        if parser == parser_modularity or parser == parser_info:
            init_group.add_argument('-i', '--init_row_labels', default=None,
                                    help='file containing the initial row \
                                    labels, if not set random initialization \
                                    is performed')

        init_group.add_argument('--n_runs', type=int, default=1,
                                help='number of runs')

        parameters_group.add_argument('--seed', type=int, default=None,
                                      help='set the random state, useful for \
                                      reproductible results')

        # evaluation and visu args
        evaluation_group = parser.add_argument_group('evaluation parameters')
        if parser != coclust_nb_parser:
            evaluation_group.add_argument('-l', '--true_row_labels',
                                          default=None,
                                          help='file containing the true \
                                          row labels')
        evaluation_group.add_argument("--visu", action="store_true",
                                      help="Plot modularity values and \
                                      reorganized matrix (requires Numpy, \
                                      SciPy and matplotlib).")

    return {'coclust': main_parser,
            'coclust-nb': coclust_nb_parser}


def get_coclust_parser():
    return get_parsers()['coclust']


def get_coclust_nb_parser():
    return get_parsers()['coclust-nb']


def modularity_by_number_of_clusters(args, modularities):
    if args['visu']:
        try:
            import matplotlib.pyplot as plt
            plt.plot(modularities, marker='o',
                     xdata=range(args['from'], args['to'] + 1))
            plt.title("Evolution of modularity")
            plt.ylabel('Lc')
            plt.xlabel('Number of clusters')
            plt.show()

        except ImportError:
            logging.error("--visu option requires Matplotlib")


def main_coclust_nb():
    parser = get_coclust_nb_parser()
    args = vars(parser.parse_args())
    X = get_data_matrix(args)

    from .coclustering import CoclustMod
    modularity = -np.inf
    best_model = CoclustMod()

    modularities = []
    for n_coclusters in range(args['from'], args['to'] + 1):
        model = CoclustMod(n_clusters=n_coclusters, max_iter=args['max_iter'],
                           n_init=args['n_runs'], random_state=args['seed'])
        model.fit(X)
        modularities.append(model.modularity)

        if (model.modularity > modularity):
            modularity = model.modularity
            best_model = model

    print("The best number of co-clusters is", best_model.n_clusters,
          "with a modularity of", best_model.modularity)

    process_output_labels(args, best_model)

    # 3) show convergence and reorganised matrix
    args['subparser_name'] = "modularity"
    process_visualization(args, best_model, X)
    modularity_by_number_of_clusters(args, modularities)


def main_coclust():
    parser = get_coclust_parser()
    args = vars(parser.parse_args())
    if (args['subparser_name'] == "modularity"):
        modularity(args)
    elif (args['subparser_name'] == "specmodularity"):
        spec_modularity(args)
    elif (args['subparser_name'] == "info"):
        info(args)


def get_data_matrix(args):
    # 1) read the provided matlab matrix or build a matrix from a file in
    # sparse format

    if args['matlab_matrix_key'] is not None:
        # matlab input
        from scipy.io import loadmat
        matlab_dict = loadmat(args['INPUT_MATRIX'])
        key = args['matlab_matrix_key']
        try:
            X = matlab_dict[key]
        except:
            logging.error("Matlab key %s not found" % key)
            sys.exit(1)

    else:
        # csv file (matrix market format)
        if args['csv_sep'] == "\\t":
            args['csv_sep'] = "\t"
        with open(args['INPUT_MATRIX'], 'r') as f:
            f_line = f.readline().strip()
            t_line = f_line.split(args['csv_sep'])
            X = sp.lil_matrix((int(t_line[0]), int(t_line[1])))
            for i, l in enumerate(f):
                l = l.strip()
                t = l.split(args['csv_sep'])
                r, c, v = int(t[0]), int(t[1]), int(t[2])
                try:
                    X[r, c] = v
                except Exception as e:
                    logging.error(e)
                    logging.error("Problem with line %s" % i)
                    sys.exit(1)

    return X


def process_output_labels(args, model):
    if args['output_row_labels']:
        with open(args['output_row_labels'], 'w') as f:
            f.write(" ".join([str(i) for i in model.row_labels_]))
    else:
        print("*****", "row labels", "*****")
        print(model.row_labels_)

    if args['output_column_labels']:
        with open(args['output_column_labels'], 'w') as f:
            f.write(" ".join([str(i) for i in model.column_labels_]))
    else:
        print("******", "column labels", "*****")
        print(model.column_labels_)

    if args.get('output_fuzzy_row_labels', None):
        print("Save first cols of ordered BW")

    if args.get('output_fuzzy_column_labels', None):
        print("Save first cols of ordered BtZ")


def process_visualization(args, model, X):
    if args['visu']:
        try:
            import matplotlib.pyplot as plt
            if (args['subparser_name'] == "info"):
                plt.plot(model.criterions, marker='o')
                plt.title("Evolution of criterion")
            if (args['subparser_name'] == "modularity"):
                plt.plot(model.modularities, marker='o')
                plt.title("Evolution of modularity")
            plt.ylabel('Lc')
            plt.xlabel('Iterations')
            plt.show()

            X = sp.csr_matrix(X)
            X_reorg = X[np.argsort(model.row_labels_)]
            X_reorg = X_reorg[:, np.argsort(model.column_labels_)]
            plt.spy(X_reorg, precision=0.8, markersize=0.9)
            plt.title("Reorganized matrix")
            plt.show()
        except ImportError:
            logging.error("--visu option requires Matplotlib")


def process_evaluation(args, model):
    if args['true_row_labels']:
        try:
            with open(args['true_row_labels'], 'r') as f:
                labels = f.read().split()

            from sklearn.metrics.cluster import normalized_mutual_info_score
            from sklearn.metrics.cluster import adjusted_rand_score
            from sklearn.metrics import confusion_matrix

            n = normalized_mutual_info_score(labels, model.row_labels_)
            ari = adjusted_rand_score(labels, model.row_labels_)
            cm = confusion_matrix(labels, model.row_labels_)

            print("nmi ==>" + str(n))
            print("adjusted rand index ==>" + str(ari))
            print()
            print(cm)
        except Exception as e:
            logging.error("--true_row_labels option (evaluation) exception:\
                          %s" % e)


def spec_modularity(args):
    X = get_data_matrix(args)

    from .coclustering import CoclustSpecMod
    model = CoclustSpecMod(n_clusters=args['n_coclusters'],
                           max_iter=args['max_iter'],
                           n_init=args['n_runs'],
                           tol=args['epsilon'],
                           random_state=args['seed'])
    model.fit(X)

    process_output_labels(args, model)
    # TODO: visualisation
    process_evaluation(args, model)


def modularity(args):
    # 1) Initialization options
    X = get_data_matrix(args)

    if args['init_row_labels']:
        W = sp.lil_matrix(np.loadtxt(args['init_row_labels']), dtype=float)
    else:
        W = None

    # 2) perform co-clustering

    from .coclustering import CoclustMod
    model = CoclustMod(n_clusters=args['n_coclusters'], init=W,
                       max_iter=args['max_iter'], n_init=args['n_runs'],
                       tol=args['epsilon'],
                       random_state=args['seed'])
    model.fit(X)

    process_output_labels(args, model)

    # 3) show convergence and reorganised matrix
    process_visualization(args, model, X)

    # 4) evaluate using gold standard (if provided)
    process_evaluation(args, model)


def info(args):
    # 1) Initialization options
    X = get_data_matrix(args)

    if args['init_row_labels']:
        W = sp.lil_matrix(np.loadtxt(args['init_row_labels']), dtype=float)
    else:
        W = None

    # 2) perform co-clustering

    from .coclustering import CoclustInfo
    model = CoclustInfo(n_row_clusters=args['n_row_clusters'],
                        n_col_clusters=args['n_col_clusters'], init=W,
                        max_iter=args['max_iter'], n_init=args['n_runs'],
                        tol=args['epsilon'])
    model.fit(X)

    process_output_labels(args, model)

    # 3) show convergence and reorganised matrix
    process_visualization(args, model, X)

    # 4) evaluate using gold standard (if provided)
    process_evaluation(args, model)
