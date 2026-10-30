import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, mutual_info_score

OUT = "results"
N1 = N3 = 600


def gen_type1(n, rng):
    a, b, e = rng.uniform(-1, 1, size=(3, n))
    return np.column_stack([a, b, 0.5 + a + b + e])


def gen_type3(n, rng):
    a, b, e = rng.normal(0.0, 0.5, size=(3, n))
    return np.column_stack([a, b + e, 0.5 + a + e])


def make_sample(seed):
    rng = np.random.default_rng(seed)
    X = np.vstack([gen_type1(N1, rng), gen_type3(N3, rng)])
    y = np.r_[np.zeros(N1, int), np.ones(N3, int)]
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def score(X, labels, y):
    k = len(np.unique(labels))
    mi = mutual_info_score(y, labels)
    if not 1 < k < len(X):
        return k, np.nan, mi
    return k, silhouette_score(X, labels), mi


def run(X, bw):
    return MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X)


def main():
    X, y = make_sample(0)
    print(f"выборка: {len(X)} точек, типы 1 и 3 по {N1}\n")

    print("=== 1. Зависимость от bandwidth (seed=0) ===")
    print(f"{'bw':>5} {'k':>3} {'silhouette':>11} {'mutual_info':>12}")
    rows = []
    for bw in np.arange(0.2, 1.45, 0.1):
        k, s, mi = score(X, run(X, bw), y)
        rows.append((bw, k, s, mi))
        print(f"{bw:5.2f} {k:3d} {s:11.3f} {mi:12.3f}")
    best = max((r for r in rows if not np.isnan(r[2])), key=lambda r: r[2])
    print(f"\nлучший bandwidth = {best[0]:.2f} (silhouette={best[2]:.3f})\n")

    print("=== 2. bandwidth из estimate_bandwidth ===")
    print(f"{'quantile':>9} {'bw':>6} {'k':>3} {'silhouette':>11} {'mutual_info':>12}")
    for q in (0.1, 0.2, 0.3, 0.4, 0.5):
        bw = estimate_bandwidth(X, quantile=q, n_samples=len(X), random_state=0)
        k, s, mi = score(X, run(X, bw), y)
        print(f"{q:9.1f} {bw:6.2f} {k:3d} {s:11.3f} {mi:12.3f}")

    print("\n=== 3. Устойчивость: bandwidth x seed генератора ===")
    grid = [0.5, 0.7, 0.9, 1.0, 1.1]
    print(f"{'bw':>5} | " + " ".join(f"seed{s}" for s in range(5)) + " | k=2  silhouette")
    table = {}
    for bw in grid:
        ks, ss = [], []
        for seed in range(5):
            Xs, ys = make_sample(seed)
            k, s, mi = score(Xs, run(Xs, bw), ys)
            ks.append(k); ss.append(s)
        table[bw] = (ks, ss)
        cells = " ".join(f"{k:5d}" for k in ks)
        frac = sum(k == 2 for k in ks) / len(ks)
        mean_s = np.nanmean(ss) if np.isfinite(ss).any() else np.nan
        print(f"{bw:5.2f} | {cells} | {frac:4.1f}  {mean_s:10.3f}")
    stable = [b for b, (ks, _) in table.items() if all(k == 2 for k in ks)]
    print(f"\nbandwidth с k=2 на всех seed: {stable}")

    print(f"для сравнения: silhouette истинных групп = {silhouette_score(X, y):.3f}, "
          f"mutual_info = {mutual_info_score(y, y):.3f} (максимум ln 2 для 2 классов)")

    l1, l2 = run(X, best[0]), run(X, best[0])
    print(f"повторный запуск с тем же bw дал идентичные метки: {np.array_equal(l1, l2)} "
          "(MeanShift детерминирован)")

    bw = best[0]
    fig = plt.figure(figsize=(11, 5))
    for i, (lab, title) in enumerate([(y, "истинные группы"), (run(X, bw), f"MeanShift, bw={bw:.2f}")], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=lab, cmap="tab10", s=18)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(f"{OUT}/points_clusters.png", dpi=130)


if __name__ == "__main__":
    main()
