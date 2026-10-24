import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cropdx.metrics import confusion, grouped_stratified_split, macro_f1, report_dict


def test_split_properties():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(6), 200); groups = np.arange(len(labels))
    groups[:30] = groups[0]; groups[300:340] = 999          # two big multi-class-ish groups
    s = grouped_stratified_split(groups, labels, [.7, .15, .15], 42)
    for g in np.unique(groups):
        assert len(set(s[groups == g])) == 1                 # no group straddles splits
    for c in range(6):
        share = np.bincount(s[labels == c], minlength=3) / (labels == c).sum()
        assert share.min() > 0 and abs(share[0] - .7) < .15
    assert set(s) == {0, 1, 2}


def test_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1); y = rng.integers(0, 5, 500); p = np.where(rng.random(500) < .8, y, rng.integers(0, 5, 500))
    names = list("abcde")
    assert np.array_equal(confusion(y, p, 5), sk.confusion_matrix(y, p, labels=range(5)))
    assert abs(macro_f1(y, p, 5) - sk.f1_score(y, p, average="macro")) < 1e-9
    mine = report_dict(y, p, names)
    ref = sk.classification_report(y, p, labels=range(5), target_names=names, output_dict=True, zero_division=0)
    for k in names + ["macro avg", "weighted avg"]:
        for m in ("precision", "recall", "f1-score", "support"):
            assert abs(mine[k][m] - ref[k][m]) < 1e-9, (k, m)
    pd.DataFrame(mine).T.to_csv("/tmp/_r.csv")               # the exact call evaluate.py makes
