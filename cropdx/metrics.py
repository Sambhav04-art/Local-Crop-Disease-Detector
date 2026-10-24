"""Pure numpy/pandas replacements for the few scikit-learn functions we used
(so the project runs on locked-down Windows machines that block scipy's DLLs)."""
import numpy as np


def confusion(y, p, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    return cm


def _prf(cm):
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    return prec, rec, f1, true


def macro_f1(y, p, n):
    """Macro F1 over classes present in y or p (same convention as sklearn's default)."""
    cm = confusion(y, p, n)
    _, _, f1, _ = _prf(cm)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return float(f1[present].mean())


def report_dict(y, p, names):
    """Same structure as sklearn.classification_report(output_dict=True) with all classes listed."""
    n = len(names)
    cm = confusion(y, p, n)
    prec, rec, f1, sup = _prf(cm)
    out = {nm: {"precision": float(prec[i]), "recall": float(rec[i]), "f1-score": float(f1[i]), "support": int(sup[i])}
           for i, nm in enumerate(names)}
    out["accuracy"] = float(np.trace(cm) / max(1, cm.sum()))
    out["macro avg"] = {"precision": float(prec.mean()), "recall": float(rec.mean()), "f1-score": float(f1.mean()),
                        "support": int(sup.sum())}
    w = sup / max(1.0, sup.sum())
    out["weighted avg"] = {"precision": float((prec * w).sum()), "recall": float((rec * w).sum()),
                           "f1-score": float((f1 * w).sum()), "support": int(sup.sum())}
    return out


def grouped_stratified_split(groups, labels, fractions, seed):
    """Greedy stratified GROUP split: every group goes wholly into one split, class proportions
    are kept close to `fractions` (e.g. [0.70, 0.15, 0.15]). Returns an int array of split ids."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    fr = np.asarray(fractions, dtype=float)
    K = int(labels.max()) + 1
    total = np.bincount(labels, minlength=K).astype(float)
    members = {}
    for i, (g, c) in enumerate(zip(groups, labels)):
        members.setdefault(g, []).append(i)
    rng = np.random.default_rng(seed)
    order = list(members)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))          # stable: big groups first, random among equals
    assigned = np.zeros((len(fr), K))
    out = np.zeros(len(groups), dtype=int)
    for g in order:
        idx = members[g]
        cnt = np.bincount(labels[idx], minlength=K).astype(float)
        nz = np.nonzero(cnt)[0]
        score = [(cnt[nz] * (fr[s] - assigned[s, nz] / total[nz]) / fr[s]).sum() for s in range(len(fr))]
        s = int(np.argmax(score))
        assigned[s] += cnt
        out[idx] = s
    return out
