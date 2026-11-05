import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)


def act(name, z):
    if name == "step":
        return (z > 0.0).astype(float)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    if name == "relu":
        return np.maximum(0.0, z)
    return z


def dact(name, a, z):
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "relu":
        return (z > 0.0).astype(float)
    return np.zeros_like(z)


class MLP:
    def __init__(self, sizes, hidden="tanh", out="sigmoid", seed=0):
        self.sizes = list(sizes)
        self.hidden = hidden
        self.out = out
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for f_in, f_out in zip(sizes[:-1], sizes[1:]):
            s = np.sqrt(2.0 / (f_in + f_out))
            self.W.append(rng.normal(0.0, s, (f_in, f_out)))
            self.b.append(np.zeros(f_out))
        self.mu = None
        self.sd = None

    def copy(self):
        m = MLP(self.sizes, self.hidden, self.out)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        m.mu, m.sd = self.mu, self.sd
        return m

    def forward_all(self, X):
        a = np.asarray(X, float)
        if self.mu is not None:
            a = (a - self.mu) / self.sd
        zs, acts = [], [a]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            fn = self.out if i == len(self.W) - 1 else self.hidden
            z = a @ W + b
            a = act(fn, z)
            zs.append(z)
            acts.append(a)
        return zs, acts

    def hidden_outs(self, X):
        return self.forward_all(X)[1][1:-1]

    def predict(self, X):
        y = self.forward_all(X)[1][-1].ravel()
        return (y > 0.5).astype(int) if self.out != "step" else y.astype(int)

    def fit(self, X, y, epochs=600, lr=0.015, batch=64, seed=0):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = len(X)
        rng = np.random.default_rng(seed)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8

        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        losses, ep_times = [], []
        L = len(self.W)

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            idx = rng.permutation(n)
            tot_loss = 0.0
            for s in range(0, n, batch):
                xb = X[idx[s:s + batch]]
                yb = y[idx[s:s + batch]]
                zs, acts = self.forward_all(xb)
                p = np.clip(acts[-1], 1e-9, 1.0 - 1e-9)
                tot_loss -= float((yb * np.log(p) + (1.0 - yb) * np.log(1.0 - p)).sum())

                delta = (acts[-1] - yb) / len(xb)
                gW, gb = [None] * L, [None] * L
                for k in range(L - 1, -1, -1):
                    gW[k] = acts[k].T @ delta
                    gb[k] = delta.sum(axis=0)
                    if k > 0:
                        delta = (delta @ self.W[k].T) * dact(self.hidden, acts[k], zs[k - 1])

                grads = gW + gb
                for j, g in enumerate(grads):
                    m[j] = b1 * m[j] + (1.0 - b1) * g
                    v[j] = b2 * v[j] + (1.0 - b2) * (g * g)
                    mh = m[j] / (1.0 - b1 ** ep)
                    vh = v[j] / (1.0 - b2 ** ep)
                    params[j] -= lr * mh / (np.sqrt(vh) + eps)

            ep_times.append(time.perf_counter() - t0)
            losses.append(tot_loss / n)
        return losses, ep_times

    def zero_neurons(self, layer_idx, frac, seed=0):
        n_neur = self.W[layer_idx].shape[0]
        k = int(round(frac * n_neur))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n_neur, size=k, replace=False)
        self.W[layer_idx][chosen, :] = 0.0
        if layer_idx > 0:
            self.W[layer_idx - 1][:, chosen] = 0.0
            self.b[layer_idx - 1][chosen] = 0.0
        return chosen


def metrics(model, X, y):
    y = np.asarray(y).ravel()
    pr = model.predict(X)
    tp = int(((y == 1) & (pr == 1)).sum())
    fp = int(((y == 0) & (pr == 1)).sum())
    fn = int(((y == 1) & (pr == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    acc = float((pr == y).mean())
    return acc, prec, rec


def split_data(X, y, test_ratio=0.3, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(len(X) * (1.0 - test_ratio))
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


TRI = np.array([[-1.0, 0.5], [1.0, 0.5], [0.0, 1.6]])
BOX = (-2.0, 2.0, -1.0, 2.4)


def in_sample2d(X):
    x, y = X[:, 0], X[:, 1]
    in_hp = y > 0.0
    in_tri = (y > 0.5) & (1.1 * x - y + 1.6 > 0.0) & (-1.1 * x - y + 1.6 > 0.0)
    return (in_hp & ~in_tri).astype(int)


def make_2d(n_per_cls=1200, seed=0):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for cls in (1, 0):
        pts = []
        while sum(len(p) for p in pts) < n_per_cls:
            cand = rng.uniform([BOX[0], BOX[2]], [BOX[1], BOX[3]], size=(4096, 2))
            pts.append(cand[in_sample2d(cand) == cls])
        X_list.append(np.vstack(pts)[:n_per_cls])
        y_list.append(np.full(n_per_cls, cls, dtype=int))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_manual_net():
    net = MLP([2, 4, 2, 1], hidden="step", out="step")
    net.W[0] = np.array([[0.0, 0.0, -1.1, 1.1],
                         [1.0, -1.0, 1.0, 1.0]])
    net.b[0] = np.array([0.0, 0.5, -1.6, -1.6])
    net.W[1] = np.array([[1.0, 0.0],
                         [0.0, 1.0],
                         [0.0, 1.0],
                         [0.0, 1.0]])
    net.b[1] = np.array([-0.5, -0.5])
    net.W[2] = np.array([[1.0], [1.0]])
    net.b[2] = np.array([-1.5])
    return net


def run_task2():
    print("Задание 2 (выборка 8)")
    print("прямых: 4, выпуклых областей: кл1=3, кл2=2, сеть: 2-4-2-1")

    X, y = make_2d(1200, seed=0)
    Xtr, ytr, Xte, yte = split_data(X, y, 0.3, seed=1)

    xs = np.linspace(BOX[0], BOX[1], 300)
    ys = np.linspace(BOX[2], BOX[3], 300)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    manual = build_manual_net()
    acc_tr, pr_tr, rc_tr = metrics(manual, Xtr, ytr)
    acc_te, pr_te, rc_te = metrics(manual, Xte, yte)
    print(f"\nручные веса (step): train acc={acc_tr:.3f}, test acc={acc_te:.3f} (prec={pr_te:.3f}, rec={rc_te:.3f})")

    print("\nсмена активации на ручных весах:")
    for fn in ("step", "tanh", "sigmoid", "relu"):
        m = manual.copy()
        m.hidden = fn
        a, p, r = metrics(m, Xte, yte)
        print(f"  {fn:7s}: acc={a:.3f} prec={p:.3f} rec={r:.3f}")

    h1, h2 = manual.hidden_outs(grid)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes[0, 0].scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=4, c="royalblue", label="кл 1")
    axes[0, 0].scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=4, c="tomato", label="кл 2")
    axes[0, 0].set_title("Выборка")
    axes[0, 0].legend(fontsize=7, loc="lower right")

    for ax, z_map, ttl in [
        (axes[0, 1], manual.predict(grid), "Выход (ручные)"),
        (axes[0, 2], h2[:, 0], "Слой 2, u1"),
        (axes[0, 3], h2[:, 1], "Слой 2, u2"),
        (axes[1, 0], h1[:, 0], "Слой 1, z1"),
        (axes[1, 1], h1[:, 1], "Слой 1, z2"),
        (axes[1, 2], h1[:, 2], "Слой 1, z3"),
        (axes[1, 3], h1[:, 3], "Слой 1, z4"),
    ]:
        ax.contourf(gx, gy, z_map.reshape(gx.shape), levels=[-0.1, 0.5, 1.1],
                    colors=["#ffd6cc", "#cce0ff"])
        tri_closed = np.vstack([TRI, TRI[0]])
        ax.plot(tri_closed[:, 0], tri_closed[:, 1], "k--", lw=0.9)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        ax.set_title(ttl, fontsize=9)
    for ax in axes.ravel():
        ax.set_xlim(BOX[0], BOX[1]); ax.set_ylim(BOX[2], BOX[3])
    fig.tight_layout()
    fig.savefig("results/task2_manual.png", dpi=130)
    plt.close(fig)

    print("\nобучение 2-4-2-1:")
    trained_nets = {}
    for fn in ("tanh", "relu", "sigmoid"):
        net = MLP([2, 4, 2, 1], hidden=fn, out="sigmoid", seed=1)
        net.fit(Xtr, ytr, epochs=700, lr=0.02, batch=64, seed=1)
        atr, ptr, rtr = metrics(net, Xtr, ytr)
        ate, pte, rte = metrics(net, Xte, yte)
        trained_nets[fn] = net
        print(f"  {fn:7s}: train acc={atr:.3f} (p={ptr:.3f}, r={rtr:.3f}) | test acc={ate:.3f} (p={pte:.3f}, r={rte:.3f})")

    best_net = trained_nets["tanh"]
    th1, th2 = best_net.hidden_outs(grid)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes[0, 0].contourf(gx, gy, best_net.predict(grid).reshape(gx.shape),
                        levels=[-0.1, 0.5, 1.1], colors=["#ffd6cc", "#cce0ff"])
    axes[0, 0].scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=3, c="royalblue", alpha=0.5)
    axes[0, 0].scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=3, c="tomato", alpha=0.5)
    axes[0, 0].set_title("Выход (tanh)", fontsize=9)

    axes[0, 1].contourf(gx, gy, trained_nets["relu"].predict(grid).reshape(gx.shape),
                        levels=[-0.1, 0.5, 1.1], colors=["#ffd6cc", "#cce0ff"])
    axes[0, 1].set_title("Выход (relu)", fontsize=9)

    for j in range(2):
        axes[0, 2 + j].pcolormesh(gx, gy, th2[:, j].reshape(gx.shape),
                                  shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0, 2 + j].set_title(f"Слой 2, нейрон {j+1}", fontsize=9)

    for j in range(4):
        axes[1, j].pcolormesh(gx, gy, th1[:, j].reshape(gx.shape),
                              shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, j].set_title(f"Слой 1, нейрон {j+1}", fontsize=9)

    fig.tight_layout()
    fig.savefig("results/task2_trained.png", dpi=130)
    plt.close(fig)

    print("\nчисло нейронов (1-й слой):")
    configs = [
        ([2, 2, 2, 1], "2-2-2-1"),
        ([2, 3, 2, 1], "2-3-2-1"),
        ([2, 4, 2, 1], "2-4-2-1"),
        ([2, 5, 2, 1], "2-5-2-1"),
        ([2, 6, 2, 1], "2-6-2-1"),
    ]
    n_stats = []
    for cfg, label in configs:
        m = MLP(cfg, hidden="tanh", out="sigmoid", seed=3)
        m.fit(Xtr, ytr, epochs=700, lr=0.02, batch=64, seed=3)
        atr, ptr, rtr = metrics(m, Xtr, ytr)
        ate, pte, rte = metrics(m, Xte, yte)
        n_stats.append((label, ptr, rtr, pte, rte, ate))
        print(f"  {label}: train p={ptr:.3f} r={rtr:.3f} | test p={pte:.3f} r={rte:.3f} acc={ate:.3f}")

    print("\nобнуление нейронов (2-8-8-1, предпоследний слой):")
    wide = MLP([2, 8, 8, 1], hidden="tanh", out="sigmoid", seed=4)
    wide.fit(Xtr, ytr, epochs=700, lr=0.02, batch=64, seed=4)
    ablation_res = []
    for frac, name in [(0.0, "0%"), (0.25, "25%"), (0.50, "50%")]:
        ps, rs, acs = [], [], []
        for s in range(5):
            mc = wide.copy()
            if frac > 0:
                mc.zero_neurons(layer_idx=2, frac=frac, seed=s)
            a, p, r = metrics(mc, Xte, yte)
            acs.append(a); ps.append(p); rs.append(r)
        ablation_res.append((name, np.mean(ps), np.std(ps), np.mean(rs), np.std(rs), np.mean(acs)))
        print(f"  {name:4s}: test p={np.mean(ps):.3f} r={np.mean(rs):.3f} acc={np.mean(acs):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x_idx = np.arange(len(n_stats))
    ax1.plot(x_idx, [r[1] for r in n_stats], "o-", label="prec train")
    ax1.plot(x_idx, [r[2] for r in n_stats], "s-", label="rec train")
    ax1.plot(x_idx, [r[3] for r in n_stats], "o--", label="prec test")
    ax1.plot(x_idx, [r[4] for r in n_stats], "s--", label="rec test")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([r[0] for r in n_stats])
    ax1.set_ylim(0.5, 1.03)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Влияние числа нейронов")

    xb = np.arange(len(ablation_res))
    ax2.bar(xb - 0.18, [r[1] for r in ablation_res], 0.35, yerr=[r[2] for r in ablation_res],
            capsize=4, label="prec test")
    ax2.bar(xb + 0.18, [r[3] for r in ablation_res], 0.35, yerr=[r[4] for r in ablation_res],
            capsize=4, label="rec test")
    ax2.set_xticks(xb)
    ax2.set_xticklabels([r[0] for r in ablation_res])
    ax2.set_ylim(0.0, 1.08)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8)
    ax2.set_title("Обнуление весов (2-8-8-1)")

    fig.tight_layout()
    fig.savefig("results/task2_experiments.png", dpi=130)
    plt.close(fig)


def gen_ball(n, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.random(n) ** (1.0 / 3.0)
    return v * r[:, None]


def gen_cube(n, rng):
    return rng.uniform(-0.5, 0.5, size=(n, 3)) + np.array([0.5, 0.0, 0.0])


def gen_below_plane(n, rng):
    pts = []
    while sum(len(p) for p in pts) < n:
        c = rng.uniform(-2.2, 2.2, size=(4096, 3))
        pts.append(c[c[:, 0] - c[:, 1] + c[:, 2] < -3.0])
    return np.vstack(pts)[:n]


def make_3d(n_per_class=1500, seed=0):
    rng = np.random.default_rng(seed)
    half = n_per_class // 2
    c1 = np.vstack([gen_ball(half, rng), gen_below_plane(n_per_class - half, rng)])
    c2 = gen_cube(n_per_class, rng)
    X = np.vstack([c1, c2])
    y = np.r_[np.ones(len(c1), int), np.zeros(len(c2), int)]
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def run_task3():
    print("\nЗадание 3 (выборки 1, 3, 5: сеть 3-k-10-1)")
    X, y = make_3d(1500, seed=0)
    Xtr, ytr, Xte, yte = split_data(X, y, 0.3, seed=1)

    print("  k | ep_ms  sec | tr_p  tr_r  tr_a | te_p  te_r  te_a")
    rows = []
    for k in range(5, 55, 5):
        net = MLP([3, k, 10, 1], hidden="tanh", out="sigmoid", seed=10)
        t0 = time.perf_counter()
        _, ep_times = net.fit(Xtr, ytr, epochs=200, lr=0.01, batch=64, seed=10)
        tot_s = time.perf_counter() - t0
        ep_ms = float(np.mean(ep_times) * 1000.0)
        atr, ptr, rtr = metrics(net, Xtr, ytr)
        ate, pte, rte = metrics(net, Xte, yte)
        rows.append((k, ep_ms, tot_s, ptr, rtr, atr, pte, rte, ate))
        print(f"{k:3d} | {ep_ms:5.2f} {tot_s:4.2f} | {ptr:.3f} {rtr:.3f} {atr:.3f} | {pte:.3f} {rte:.3f} {ate:.3f}")

    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    p1 = Xtr[ytr == 1][::4]
    p2 = Xtr[ytr == 0][::4]
    ax3d.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=4, c="royalblue", label="кл 1 (1+5)")
    ax3d.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=4, c="tomato", label="кл 2 (3)")
    ax3d.set_title("Выборка 3D")
    ax3d.legend(fontsize=7)

    ks = [r[0] for r in rows]
    ax_t = fig.add_subplot(2, 2, 2)
    ax_t.plot(ks, [r[2] for r in rows], "o-", c="tab:orange", label="всего, с")
    ax_t2 = ax_t.twinx()
    ax_t2.plot(ks, [r[1] for r in rows], "s--", c="tab:blue", label="эпоха, мс")
    ax_t.set_xlabel("k")
    ax_t.set_ylabel("с", color="tab:orange")
    ax_t2.set_ylabel("мс", color="tab:blue")
    ax_t.grid(alpha=0.3)
    ax_t.set_title("Время обучения")

    ax_p = fig.add_subplot(2, 2, 3)
    ax_p.plot(ks, [r[3] for r in rows], "o-", label="train")
    ax_p.plot(ks, [r[6] for r in rows], "s--", label="test")
    ax_p.set_xlabel("k"); ax_p.set_ylabel("precision")
    ax_p.set_ylim(0.7, 1.03); ax_p.grid(alpha=0.3); ax_p.legend(fontsize=8)
    ax_p.set_title("Precision")

    ax_r = fig.add_subplot(2, 2, 4)
    ax_r.plot(ks, [r[4] for r in rows], "o-", label="train")
    ax_r.plot(ks, [r[7] for r in rows], "s--", label="test")
    ax_r.set_xlabel("k"); ax_r.set_ylabel("recall")
    ax_r.set_ylim(0.7, 1.03); ax_r.grid(alpha=0.3); ax_r.legend(fontsize=8)
    ax_r.set_title("Recall")

    fig.tight_layout()
    fig.savefig("results/task3_results.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_task2()
    run_task3()
