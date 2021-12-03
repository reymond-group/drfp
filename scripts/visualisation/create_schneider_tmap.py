import os
import pickle
import numpy as np
import tmap as tm
from annoy import AnnoyIndex
from faerun import Faerun
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib.pyplot as plt

CFG_TMAP = tm.LayoutConfiguration()
CFG_TMAP.k = 50
CFG_TMAP.kc = 50
CFG_TMAP.sl_scaling_min = 1.0
CFG_TMAP.sl_scaling_max = 1.0
CFG_TMAP.sl_repeats = 1
CFG_TMAP.sl_extra_scaling_steps = 2
CFG_TMAP.placer = tm.Placer.Barycenter
CFG_TMAP.merger = tm.Merger.LocalBiconnected
CFG_TMAP.merger_factor = 2.0
CFG_TMAP.merger_adjustment = 0
CFG_TMAP.fme_iterations = 1000
CFG_TMAP.sl_scaling_type = tm.ScalingType.RelativeToDesiredLength
CFG_TMAP.node_size = 1 / 45
CFG_TMAP.mmm_repeats = 1

def main():
    dims = 2048
    tmp_file = "knn.pkl"

    X_train, y_train, _ = pickle.load(open("../../data/schneider50k_train.pkl", "rb"))
    X_test, y_test, _ = pickle.load(open("../../data/schneider50k_test.pkl", "rb"))

    X = []
    y = []

    X.extend(X_train)
    X.extend(X_test)

    X = np.array(X)

    y.extend(y_train)
    y.extend(y_test)

    y_values = [int(ytem.split(".")[0]) for ytem in y]

    knn = []

    if os.path.isfile(tmp_file):
        knn = pickle.load(open(tmp_file, "rb"))
    else:
        annoy = AnnoyIndex(dims, metric="angular")

        for i, v in enumerate(X):
            annoy.add_item(i, v)

        annoy.build(10)

        for i in range(len(X)):
            for j in annoy.get_nns_by_item(i, 10):
                knn.append((i, j, cosine_distance(X[i], X[j])))

        with open(tmp_file, "wb+") as f:
            pickle.dump(knn, f)

    x, y, s, t, _ = tm.layout_from_edge_list(len(X), knn, config=CFG_TMAP)

    # Plot the edges
    for i in range(len(s)):
        plt.plot(
            [x[s[i]], x[t[i]]],
            [y[s[i]], y[t[i]]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
            zorder=1,
        )

    plt.scatter(x, y, s=2.5, c=y_values, zorder=2, cmap="tab10")
    plt.show()



if __name__ == "__main__":
    main()
