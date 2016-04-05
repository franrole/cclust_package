Basic Usage Examples
====================

CoclustMod
~~~~~~~~~~

In the following example, the CSTR dataset is loaded from a Matlab matrix using
the SciPy library. The data is stored in X and a co-clustering model using
direct maximisation of the modularity is then fitted with 4 clusters. The
modularity is printed and the predicted row labels and column labels are
retrieved for further exploration or evaluation.

.. code-block:: python

    from scipy.io import loadmat
    from coclust.CoclustMod import CoclustMod

    file_name = "../datasets/cstr.mat"
    matlab_dict = loadmat(file_name)
    X = matlab_dict['fea']

    model = CoclustMod(n_clusters=4)
    model.fit(X)

    print(model.modularity)
    predicted_row_labels = model.row_labels_
    predicted_column_labels = model.column_labels_

For example, the normalized mutual information score is computed using the
scikit-learn library:

.. code-block:: python

    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

    true_row_labels = matlab_dict['gnd'].flatten()

    print(nmi(true_row_labels, predicted_row_labels))

CoclustSpecMod
~~~~~~~~~~~~~~

Here, the Classic3 dataset is imported as a CSV. The first line of the file is
the number of rows followed by the number of columns and the number of clusters
the model is fitted with. The other lines are tuples of row number, column
number and value of that entry. The spectral modularity based model is fitted
and the predicted row labels retrieved. The shapes of the predicted clusters
are printed.

.. code-block:: python

    from __future__ import print_function
    import scipy.sparse as sp
    import csv
    from coclust.CoclustSpecMod import CoclustSpecMod

    file_name = "../datasets/classic3.csv"
    csv_file = open(file_name, 'rb')
    csv_reader = csv.reader(csv_file, delimiter=",")

    nb_row, nb_col, nb_clusters = map(int, csv_reader.next())
    X = sp.lil_matrix((nb_row, nb_col))

    for row in csv_reader:
        i, j, v = map(int, row)
        X[i, j] = v

    model = CoclustSpecMod(n_clusters=nb_clusters)
    model.fit(X)

    predicted_row_labels = model.row_labels_

    for i in range(nb_clusters):
        number_of_rows, number_of_columns = model.get_shape(i)
        print("Cluster", i, "has", number_of_rows, "rows and",
              number_of_columns, "columns.")

CoclustInfo
~~~~~~~~~~~

.. code-block:: python

    from coclust.CoclustInfo import CoclustInfo

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.metrics.cluster import normalized_mutual_info_score

    categories = [
        'rec.motorcycles',
        'rec.sport.baseball',
        'comp.graphics',
        'sci.space',
        'talk.politics.mideast'
    ]

    ng5 = fetch_20newsgroups(categories=categories, shuffle=True)

    true_labels = ng5.target

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('coclust', CoclustInfo()),
    ])

    pipeline.set_params(coclust__n_clusters=5)
    pipeline.fit(ng5.data)

    predicted_labels = pipeline.named_steps['coclust'].row_labels_

    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    print(nmi)
