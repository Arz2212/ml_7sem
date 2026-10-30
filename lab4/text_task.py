
# вариант 8: 10 статей об искусстве + 10 о спорте
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (silhouette_score, mutual_info_score,
                             pairwise_distances)
from sklearn.preprocessing import Normalizer
from texts_data import ART, SPORT, ART_SYNONYMS, SPORT_SYNONYMS

OUT = "results"
QUANTILES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


def normalize(text, mapping):
    """Замена синонимов общим словом внутри группы."""
    return re.sub(r"[a-z]+", lambda m: mapping.get(m.group(0), m.group(0)), text.lower())


def features(texts, n_comp=8):
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    Z = Normalizer().fit_transform(TruncatedSVD(n_comp, random_state=0).fit_transform(X))
    return Z, X, vec


def bandwidths(Z):
    d = pairwise_distances(Z)
    d = d[np.triu_indices(len(Z), 1)]
    return [float(np.quantile(d, q)) for q in QUANTILES]


def score(Z, lab, y):
    k = len(np.unique(lab))
    mi = mutual_info_score(y, lab)
    if not 1 < k < len(Z):
        return k, np.nan, mi
    return k, silhouette_score(Z, lab), mi


def top_terms(X, vec, lab, k, n=8):
    names = np.array(vec.get_feature_names_out())
    res = []
    for c in range(k):
        m = np.asarray(X[lab == c].mean(axis=0)).ravel()
        res.append(", ".join(names[m.argsort()[::-1][:n]]))
    return res


def run(texts, y, tag, verbose=True):
    Z, X, vec = features(texts)
    print(f"\n--- {tag}: TF-IDF {X.shape[0]}x{X.shape[1]} -> SVD-8 ---")
    print(f"{'bandwidth':>9} {'k':>3} {'silhouette':>11} {'mutual_info':>12} {'':>3}")

    rows, labels = [], {}
    for bw in bandwidths(Z):
        if bw <= 1e-6:
            continue
        lab = MeanShift(bandwidth=round(bw, 3), bin_seeding=True).fit_predict(Z)
        k, s, mi = score(Z, lab, y)
        rows.append((bw, k, s, mi))
        labels[bw] = lab
        print(f"{bw:9.3f} {k:3d} {s:11.3f} {mi:12.3f}")

    ok = [r for r in rows if 2 <= r[1] <= 6 and not np.isnan(r[2])]
    bw = max(ok, key=lambda r: r[2])[0] if ok else max(rows, key=lambda r: r[1])[0]
    ms = labels[bw]
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(Z)

    print(f"\nвыбран bandwidth = {bw:.3f}")
    print(f"{'метод':>10} {'k':>3} {'silhouette':>11} {'mutual_info':>12} {'':>3}")
    out = {}
    for name, lab in (("MeanShift", ms), ("KMeans", km)):
        k, s, mi = score(Z, lab, y)
        out[name] = dict(k=k, sil=s, mi=mi, lab=lab)
        print(f"{name:>10} {k:3d} {s:11.3f} {mi:12.3f}")

    for name, r in out.items():
        labs = r["lab"]
        print(f"\n{name}: размеры {np.bincount(labs).tolist()}, "
              f"искусство/спорт по кластерам "
              f"{[(int((y[labs == c] == 0).sum()), int((y[labs == c] == 1).sum())) for c in range(r['k'])]}")
        for c, t in enumerate(top_terms(X, vec, labs, r["k"])):
            print(f"  кластер {c}: {t}")

    P = PCA(2, random_state=0).fit_transform(Z)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (lab, title) in zip(axes, [(y, "истинные классы"),
                                       (ms, f"MeanShift, bw={bw:.2f}"),
                                       (km, "KMeans, k=2")]):
        ax.scatter(P[:, 0], P[:, 1], c=lab, cmap="coolwarm", s=60)
        for i in range(len(texts)):
            ax.annotate(str(i), (P[i, 0], P[i, 1]), fontsize=6)
        ax.set_title(title)
        ax.set_xlabel("SVD-1"); ax.set_ylabel("SVD-2")
    fig.tight_layout()
    fig.savefig(f"{OUT}/text_{tag}.png", dpi=130)
    return out


def main():
    y = np.r_[np.zeros(len(ART), int), np.ones(len(SPORT), int)]
    base = run(ART + SPORT, y, "base")
    improved = ([normalize(t, ART_SYNONYMS) for t in ART] +
                [normalize(t, SPORT_SYNONYMS) for t in SPORT])
    impr = run(improved, y, "improved")

    print("\n=== Сравнение (silhouette / mutual_info) ===")
    print(f"{'метод':>10} {'было':>18} {'стало':>18}")
    for name in ("MeanShift", "KMeans"):
        b, i = base[name], impr[name]
        print(f"{name:>10} {b['sil']:8.3f} / {b['mi']:6.3f} {i['sil']:12.3f} / {i['mi']:6.3f}")


if __name__ == "__main__":
    main()
