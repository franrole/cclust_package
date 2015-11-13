import argparse
import numpy as np
import scipy.sparse as sp
import sys


def get_parser():

    main_parser = argparse.ArgumentParser(prog='coclust')

    subparsers = main_parser.add_subparsers(help='choose the algorithm to use',
                                            dest='subparser_name')

    # create the parser for the "modularity" command
    parser_modularity = subparsers.add_parser('modularity',
                                              help='use the modularity based \
                                              algorithm')

    # create the parser for the "specmodularity" command
    parser_spec_modularity = subparsers.add_parser('specmodularity',
                                                   help='use the spectral \
                                                   modularity based algorithm')

    parser_list = [parser_modularity, parser_spec_modularity]
    for parser in parser_list:
        # input args
        input_group = parser.add_argument_group('input')
        input_group.add_argument('INPUT_MATRIX', help='matrix file path')
        input_params_group = input_group.add_mutually_exclusive_group()
        input_params_group.add_argument('-k', '--matlab_matrix_key',
                                        default=None, help='if not set, csv \
                                        input is considered')
        input_params_group.add_argument('-sep', '--csv_sep', default=None,
                                        help='if not set, "," is considered')
                                        
        # output args
        output_group = parser.add_argument_group('output')
        output_group.add_argument('--output_row_labels',
                                  help='file path for the predicted row labels')
        output_group.add_argument('--output_column_labels', help='file path \
                                  for the predicted column labels')
        if parser == parser_modularity:
            output_group.add_argument('--output_fuzzy_row_labels', help='file \
                                      path for the predicted fuzzy row labels')
            output_group.add_argument('--output_fuzzy_column_labels',
                                      help='file path for the predicted fuzzy \
                                      column labels')
            output_group.add_argument('--convergence_plot', help='file path \
                                      for the convergence plot')
        output_group.add_argument('--reorganized_matrix', help='file path for \
                                  the reorganized matrix')
                                  
        # parameter args
        parameters_group = parser.add_argument_group('algorithm parameters')
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
        if parser == parser_modularity:
            init_group.add_argument('-i', '--init_row_labels', default=None,
                                    help='file containing the initial row \
                                    labels, if not set random initialization \
                                    is performed')
        init_group.add_argument('--n_runs', type=int, default=1,
                                help='number of runs')
                                
        # evaluation and visu args
        evaluation_group = parser.add_argument_group('evaluation parameters')
        evaluation_group.add_argument('-l', '--true_row_labels', default=None,
                                      help='file containing the true \
                                      row labels')
        evaluation_group.add_argument("--visu", action="store_true",
                                      help="Plot modularity values and \
                                      reorganized matrix (requires numpy/scipy \
                                      and matplotlib).")

    return main_parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if (args.subparser_name == "modularity"):
        modularity(args)
    elif (args.subparser_name == "specmodularity"):
        spec_modularity(args)


def get_data_matrix(args):
    # 1) read the provided matlab matrix or build a matrix from a file in
    # sparse format

    if args.matlab_matrix_key is not None:
        # matlab input
        from scipy.io import loadmat
        matlab_dict = loadmat(args.INPUT_MATRIX)
        X = matlab_dict[args.matlab_matrix_key]
    else:
        # csv file (matrix market format)
        with open(args.INPUT_MATRIX, 'r') as f:
            f_line = f.readline().strip()
            t_line = f_line.split(',')
            X = sp.lil_matrix((t_line[0], t_line[1]))
            for i, l in enumerate(f):
                l = l.strip()
                t = l.split(',')
                r, c, v = int(t[0]), int(t[1]), int(t[2])
                try:
                    X[r, c] = v
                except Exception as e:
                    print(e)
                    print("problem with line", i)
                    sys.exit(0)

    return X


def process_output_labels(args, model):
    if args.output_row_labels :
        with open(args.output_row_labels,'w') as f :
            f.write(" ".join([str(i) for i in model.row_labels_]))
    else :
        print("*****", "row labels", "*****")
        print(model.row_labels_)
	
    if args.output_column_labels :
        with open(args.output_column_labels,'w') as f :
            f.write(" ".join([str(i) for i in model.column_labels_]))
    else :
        print("*****", "row labels", "*****")
        print(model.column_labels_)	
	


def process_visualization(args, model, X):
    if args.visu:
        try:
            import matplotlib.pyplot as plt
            plt.plot(model.modularities, marker='o')
            plt.ylabel('Lc')
            plt.xlabel('Iterations')
            plt.title("Evolution of modularity")
            plt.show()

            X = sp.csr_matrix(X)
            X_reorg = X[np.argsort(model.row_labels_)]
            X_reorg = X_reorg[:, np.argsort(model.column_labels_)]
            plt.spy(X_reorg, precision=0.8, markersize=0.9)
            plt.title("Reorganized matrix")
            plt.show()
        except Exception as e:
            print("Exception concerning the --visu option", e)
            print("This option requires Numpy/Scipy as well as Matplotlib.")


def process_evaluation(args, model):
    if args.true_row_labels:
        try:
            with open(args.true_row_labels, 'r') as f:
                labels = f.read().split()

            from sklearn.metrics.cluster import normalized_mutual_info_score
            from sklearn.metrics.cluster import adjusted_rand_score
            from sklearn.metrics import confusion_matrix

            n = normalized_mutual_info_score(labels, model.row_labels_)
            ari = adjusted_rand_score(labels, model.row_labels_)
            cm = confusion_matrix(labels, model.row_labels_)
            # accuracy=(total)/(nb_rows*1.)

            print("nmi ==>" + str(n))
            print("adjusted rand index ==>" + str(ari))
            print()
            print(cm)
        except Exception as e:
            print("Exception concerning the --true_row_labels option \
                  (evaluation)", e)
            print("This option requires Numpy/Scipy, Matplotlib and \
                  scikit-learn.")


def spec_modularity(args):
    X = get_data_matrix(args)

    from .CoclustSpecMod import CoclustSpecMod
    model = CoclustSpecMod(n_clusters=args.n_coclusters, max_iter=args.max_iter, n_runs=args.n_runs,epsilon=args.epsilon)
    model.fit(X)

    process_output_labels(args, model)
    # TODO: visualisation
    process_evaluation(args, model)


def modularity(args):
    # 1) Initialization options
    X = get_data_matrix(args)

    if args.init_row_labels:
        W = sp.lil_matrix(np.loadtxt(args.init_row_labels), dtype=float)
    else:
        W = None

    # 2) perform co-clustering

    from .CoclustMod import CoclustMod
    model = CoclustMod(n_clusters=args.n_coclusters, init=W,
                       max_iter=args.max_iter, n_runs=args.n_runs)
    model.fit(X)

    process_output_labels(args, model)

    # 3) show convergence and reorganised matrix
    process_visualization(args, model, X)

    # 4) evaluate using gold standard (if provided)
    process_evaluation(args, model)
