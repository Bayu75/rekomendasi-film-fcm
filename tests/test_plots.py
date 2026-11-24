"""Setiap fungsi grafik di src/plots.py menghasilkan Figure tanpa memanggil Streamlit."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import plots

rng = np.random.default_rng(0)
U = rng.dirichlet(np.ones(5), size=60)
X = rng.random((60, 6))
X2d = rng.normal(size=(60, 2))
V2d = rng.normal(size=(5, 2))


def test_semua_grafik():
    figs = [
        plots.plot_objective([10, 8, 7.5, 7.4]),
        plots.plot_objective([]),
        plots.plot_select_c(pd.DataFrame({"c": [3, 4, 5], "fpc": [.6, .55, .5], "silhouette": [.3, .28, .2]}), 4),
        plots.plot_membership_heatmap(U, [f"F{i}" for i in range(60)], n=20),
        plots.plot_pca_variance(np.array([.5, .3, .1])),
        plots.plot_membership_curve(X, U, 2, "Popularitas"),
        plots.plot_clusters(X2d, U, V2d, q_idx=3, rec_idx=[1, 2, 4]),
        plots.plot_clusters(X2d, U, V2d, q_idx=None, rec_idx=[1], query_xy=(0, 0)),
        plots.plot_query_membership(U[0]),
        plots.plot_cluster_radar(rng.random((5, 5)) * 10),
        plots.plot_precision(pd.DataFrame({"k": [5, 10], "precision_fuzzy": [.9, .85], "precision_baseline": [.88, .8]})),
        plots.plot_cluster_quality([10, 20, 30, 40, 50], [.3, .2, .1, .4, .5]),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)


def test_warna_cluster_8():
    assert len(plots.CLUSTER_COLORS) == 8
    assert plots.cluster_color(9) == plots.CLUSTER_COLORS[1]
