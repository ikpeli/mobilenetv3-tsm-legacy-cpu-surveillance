#!/usr/bin/env python3
"""
verify_paper_numbers.py

Recomputes every quantitative claim in the manuscript from the exported result
files in this archive and checks each against the value printed in the paper.
Run it from the archive root:

    python verify_paper_numbers.py --results results/

Exit status is 0 if every check passes, 1 otherwise.

Inputs expected in --results:
    test_clip_probs.csv                       per-window probabilities, test partition
    val_clip_probs.csv                        per-window probabilities, validation partition
    test_video_level_scores.csv               per-video aggregated scores, test
    alert_sweep_validation.csv                operating-point sweep on validation
    final_summary.json                        metrics emitted by the training notebook
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

# Operating point fixed on validation and applied unchanged to test (Section III-G).
LAMBDA, TAU, N_CONSEC = 0.30, 0.60, 3
TUNED_CLIP_THRESHOLD = 0.490234375
BOOTSTRAP_N = 10000
SEED = 42

results = []


def check(name, got, expected, tol=0.0005, unit=''):
    ok = expected is None or abs(got - expected) <= tol
    results.append(ok)
    exp = 'n/a' if expected is None else f'{expected:g}'
    flag = 'PASS' if ok else 'FAIL'
    print(f'  [{flag}] {name:<52} computed {got:.4g}{unit}   paper {exp}{unit}')


def raise_alert(probs, lam=LAMBDA, tau=TAU, n=N_CONSEC):
    """Equations (3) and (4): EMA smoothing plus N-consecutive confirmation."""
    s, run = None, 0
    for p in probs:
        s = p if s is None else lam * p + (1 - lam) * s
        run = run + 1 if s >= tau else 0
        if run >= n:
            return 1
    return 0


def event_table(clip_csv):
    df = pd.read_csv(clip_csv).sort_values(['path', 'window_idx'])
    rows = [(path, g.class_name.iloc[0], int(g.label.iloc[0]), raise_alert(g.prob.values))
            for path, g in df.groupby('path')]
    return pd.DataFrame(rows, columns=['path', 'class_name', 'y', 'alert'])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def bootstrap_ci(y, s, metric, groups=None, n=BOOTSTRAP_N):
    """Percentile CI. If groups is given, resample whole groups (videos), not rows."""
    rng = np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    if groups is None:
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                out.append(metric(y[i], s[i]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(min(n, 2000)):
            pick = rng.choice(uniq, len(uniq))
            i = np.concatenate([idx[g] for g in pick])
            if len(np.unique(y[i])) > 1:
                out.append(metric(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e


def main(res):
    j = os.path.join
    test = pd.read_csv(j(res, 'test_clip_probs.csv'))
    vid = pd.read_csv(j(res, 'test_video_level_scores.csv'))
    summary = json.load(open(j(res, 'final_summary.json')))

    print('\nTABLE 4  Detection performance')
    ap = average_precision_score(test.label, test.prob)
    rc = roc_auc_score(test.label, test.prob)
    lo, hi = bootstrap_ci(test.label, test.prob, average_precision_score, groups=test.path)
    check('Sliding window, PR-AUC (7,425 windows)', ap, 0.886)
    check('Sliding window, ROC-AUC', rc, 0.926)
    print(f'         bootstrap 95% CI on PR-AUC: [{lo:.3f}, {hi:.3f}]')
    check('Video level max, PR-AUC',
          average_precision_score(vid.true_label, vid.score_max), 0.896)
    check('Video level max, ROC-AUC',
          roc_auc_score(vid.true_label, vid.score_max), 0.902)
    check('Video level top-3 mean, PR-AUC',
          average_precision_score(vid.true_label, vid.score_topk_mean), 0.899)
    check('Video level top-3 mean, ROC-AUC',
          roc_auc_score(vid.true_label, vid.score_topk_mean), 0.907)
    # single-clip row comes from the notebook loader, not from the window exports
    sc = summary['test_metrics_single_clip_per_video'] if \
        'test_metrics_single_clip_per_video' in summary else \
        summary['test_metrics_default_threshold_clip_level']
    check('Single clip per video, PR-AUC (75 clips)', sc['pr_auc'], 0.938)
    check('Single clip per video, ROC-AUC', sc['roc_auc'], 0.922, tol=0.001)

    print('\nSECTION IV-C  Clip-level operating figures')
    pred = (test.prob >= TUNED_CLIP_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(test.label, pred).ravel()
    check('Sliding window recall (%)', 100 * tp / (tp + fn), 83.5, tol=0.1, unit='%')
    check('Sliding window precision (%)', 100 * tp / (tp + fp), 87.0, tol=0.1, unit='%')

    print('\nTABLE 5  Event-level metrics at the deployed operating point')
    for split, csv, exp in (('test', 'test_clip_probs.csv', (81.6, 88.6, 10.8)),
                            ('validation', 'val_clip_probs.csv', (81.1, 88.2, 10.5))):
        ev = event_table(j(res, csv))
        tn, fp, fn, tp = confusion_matrix(ev.y, ev.alert).ravel()
        r, lo_, hi_ = wilson(tp, tp + fn)
        check(f'{split}: event recall (%)', r, exp[0], tol=0.1, unit='%')
        check(f'{split}: event precision (%)', 100 * tp / (tp + fp), exp[1], tol=0.1, unit='%')
        check(f'{split}: false-alarm rate (%)', 100 * fp / (fp + tn), exp[2], tol=0.1, unit='%')
        print(f'         counts tp={tp} fn={fn} fp={fp} tn={tn};'
              f' recall Wilson 95% CI [{lo_:.1f}, {hi_:.1f}]')
        if split == 'test':
            for cls, e in (('Fighting', 100.0), ('Burglary', 73.3), ('Stealing', 80.0)):
                s = ev[ev.class_name == cls]
                check(f'test: {cls} recall (%)', 100 * s.alert.mean(), e, tol=0.1, unit='%')

    print('\nSECTION IV-D  Calibration')
    check('Sliding window ECE', ece(test.label, test.prob), 0.109, tol=0.001)
    check('Single clip ECE', summary.get('calibration_ece_clip_level', float('nan')),
          0.141, tol=0.001)

    print('\nTABLE 6  Validation operating-point sweep')
    sw = pd.read_csv(j(res, 'alert_sweep_validation.csv')).sort_values(
        ['event_f1', 'event_recall', 'false_alarm_rate'], ascending=[False, False, True])
    top = sw.iloc[0]
    check('Top-ranked sweep row: lambda', top.alpha, LAMBDA)
    check('Top-ranked sweep row: threshold', top.threshold, TAU)
    check('Top-ranked sweep row: N', top.N, N_CONSEC)

    n_fail = results.count(False)
    print(f'\n{len(results) - n_fail}/{len(results)} checks passed.')
    return 1 if n_fail else 0


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--results', default='results/')
    sys.exit(main(ap_.parse_args().results))
