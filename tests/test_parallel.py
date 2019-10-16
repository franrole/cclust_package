import importlib
import numpy as np
import coclust.coclustering

importlib.reload(coclust.coclustering.coclust_mod)

from coclust.coclustering.coclust_mod import _fit_single, CoclustMod


X = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
model = CoclustMod(n_clusters=2, n_jobs=2)

model.fit(X)

print(model.row_labels_); print(model.column_labels_); 






