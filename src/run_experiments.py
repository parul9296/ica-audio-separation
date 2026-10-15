"""Run every experiment, export audio, plots and RESULTS.md.

    python src/run_experiments.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from audio import load, save, SR
from ica import ica, center, pca_whiten
from metrics import evaluate, align, kurtosis

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC, OUT, PLOTS = (os.path.join(ROOT, d) for d in ("sources", "output", "plots"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

SOURCES = {
    "speech_female": "speech_female.wav",
    "music_singing": "music_singing.wav",
    "ambient_dishes": "ambient_dishes.wav",
    "speech_male": "speech_male.wav",
}
OFFSET = {"ambient_dishes": 10 * SR}   # skip the first 10 s of the long dishes recording

# Our own mixing matrices (rows = microphones, cols = sources). Fixed so results are reproducible.
A3 = np.array([[1.0, 0.6, 0.4],
               [0.5, 1.0, 0.7],
               [0.3, 0.5, 1.0]])
A4x3 = np.array([[1.0, 0.6, 0.4],
                 [0.5, 1.0, 0.7],
                 [0.3, 0.5, 1.0],
                 [0.8, 0.2, 0.9]])
A4 = np.array([[1.0, 0.5, 0.3, 0.6],
               [0.4, 1.0, 0.6, 0.2],
               [0.3, 0.4, 1.0, 0.5],
               [0.7, 0.3, 0.2, 1.0]])


def get_sources(names, seconds=None):
    sigs = {n: load(os.path.join(SRC, SOURCES[n])) for n in names}
    sigs = {n: s[OFFSET.get(n, 0):] for n, s in sigs.items()}
    if seconds:  # longer demo: repeat short clips (with a 0.3 s pause) up to the target length
        T = int(seconds * SR)
        gap = np.zeros(int(0.3 * SR))
        sigs = {n: np.tile(np.concatenate([s, gap]), T // (len(s) + len(gap)) + 1)[:T] for n, s in sigs.items()}
    L = min(len(s) for s in sigs.values())
    S = np.array([sigs[n][:L] for n in names])
    return S / S.std(axis=1, keepdims=True)           # unit variance per source


def run(names, A, tag, noise_snr=None, seed=0, fun="logcosh", export=True, seconds=None):
    S = get_sources(names, seconds)
    X = A @ S
    if noise_snr is not None:
        rng = np.random.default_rng(seed)
        p = (X ** 2).mean(axis=1, keepdims=True)
        X = X + rng.standard_normal(X.shape) * np.sqrt(p / 10 ** (noise_snr / 10))
    r = ica(X, n_components=len(names), fun=fun, seed=seed)
    ica_scores, perm = evaluate(S, r["S"])
    # PCA baseline: whitened principal components (decorrelated, but not independent)
    pca_scores, _ = evaluate(S, r["Z"])
    if export:
        for i in range(X.shape[0]):
            save(os.path.join(OUT, f"{tag}_mixture_{i+1}.wav"), X[i])
        aligned, _, _ = align(S, r["S"])
        for i, n in enumerate(names):
            save(os.path.join(OUT, f"{tag}_separated_{n}.wav"), aligned[i])
    return {"S": S, "X": X, "r": r, "ica": ica_scores, "pca": pca_scores, "perm": perm}


def summarise(res, names):
    return {n: {"ica_corr": res["ica"][i]["corr"], "ica_si_sdr": res["ica"][i]["si_sdr"],
                "pca_corr": res["pca"][i]["corr"], "pca_si_sdr": res["pca"][i]["si_sdr"]}
            for i, n in enumerate(names)}


def main():
    results = {}
    N3 = ["speech_female", "music_singing", "ambient_dishes"]
    N4 = N3 + ["speech_male"]

    # ---- Experiment 1: 3 sources -> 3 mixtures -------------------------------------
    e1 = run(N3, A3, "exp1")
    results["exp1_3src_3mix"] = {"mixing_matrix": A3.tolist(), "iterations": e1["r"]["n_iter"],
                                 "scores": summarise(e1, N3)}

    # ---- Experiment 2: 3 sources -> 4 mixtures (more mics than sources) -----------
    e2 = run(N3, A4x3, "exp2")
    results["exp2_3src_4mix"] = {"mixing_matrix": A4x3.tolist(), "iterations": e2["r"]["n_iter"],
                                 "pca_eigenvalues": pca_whiten(center(e2["X"])[0])[2].tolist(),
                                 "scores": summarise(e2, N3)}

    # ---- Experiment 3: 4 sources (two speakers) -> 4 mixtures ----------------------
    e3 = run(N4, A4, "exp3")
    results["exp3_4src_4mix"] = {"mixing_matrix": A4.tolist(), "iterations": e3["r"]["n_iter"],
                                 "scores": summarise(e3, N4)}

    # ---- Long listening demo: same 3 sources looped to 30 s ------------------------
    eL = run(N3, A3, "long", seconds=30)
    results["long_demo_30s"] = {"iterations": eL["r"]["n_iter"], "scores": summarise(eL, N3)}

    # ---- Experiment 4: robustness to sensor noise (SNR sweep, 5 seeds each) --------
    snrs = [40, 30, 20, 15, 10, 5, 0]
    sweep = {}
    for snr in snrs:
        vals = [np.mean([s["si_sdr"] for s in run(N3, A3, "", snr, seed, export=False)["ica"]])
                for seed in range(5)]
        sweep[snr] = {"mean_si_sdr": float(np.mean(vals)), "std": float(np.std(vals))}
    results["exp4_snr_sweep"] = sweep

    # ---- Experiment 5: choice of non-linearity -------------------------------------
    nl = {}
    for fun in ["logcosh", "exp", "cube"]:
        r = run(N3, A3, "", fun=fun, export=False)
        nl[fun] = {"mean_si_sdr": float(np.mean([s["si_sdr"] for s in r["ica"]])),
                   "iterations": r["r"]["n_iter"]}
    results["exp5_nonlinearity"] = nl

    # ---- Non-Gaussianity of each signal type ---------------------------------------
    S4 = get_sources(N4)
    results["kurtosis"] = {
        **{n: kurtosis(S4[i]) for i, n in enumerate(N4)},
        "mixture_1 (exp1)": kurtosis(e1["X"][0]),
        "gaussian_noise": kurtosis(np.random.default_rng(0).standard_normal(S4.shape[1])),
    }

    with open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    make_plots(e1, N3, results)
    write_markdown(results)
    print(json.dumps(results, indent=2))


# --------------------------------------------------------------------------- plots ---
def make_plots(e1, names, results):
    S, X = e1["S"], e1["X"]
    aligned, _, _ = align(S, e1["r"]["S"])
    t = np.arange(S.shape[1]) / SR

    fig, ax = plt.subplots(3, 3, figsize=(15, 7), sharex=True)
    for i, n in enumerate(names):
        ax[0, i].plot(t, S[i], lw=.4, color="tab:blue"); ax[0, i].set_title(f"Original: {n}")
        ax[1, i].plot(t, X[i], lw=.4, color="tab:gray"); ax[1, i].set_title(f"Mixture {i+1}")
        ax[2, i].plot(t, aligned[i], lw=.4, color="tab:green")
        ax[2, i].set_title(f"ICA estimate (SI-SDR {e1['ica'][i]['si_sdr']:.1f} dB)")
        ax[2, i].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "waveforms.png"), dpi=110); plt.close(fig)

    # Joint distributions: sources -> mixtures -> whitened -> ICA (first two channels)
    idx = np.random.default_rng(0).choice(S.shape[1], 6000, replace=False)
    Xc, _ = center(X)
    Z = pca_whiten(Xc)[0]
    panels = [("Sources (independent)", S), ("Mixtures (correlated)", X),
              ("PCA-whitened (uncorrelated)", Z), ("ICA output (independent)", aligned)]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for a, (title, D) in zip(ax, panels):
        a.scatter(D[0, idx], D[1, idx], s=1, alpha=.3)
        a.set_title(title); a.set_aspect("equal", "datalim")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "scatter_stages.png"), dpi=110); plt.close(fig)

    # PCA vs ICA
    sc = results["exp1_3src_3mix"]["scores"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .2, [sc[n]["pca_si_sdr"] for n in names], .4, label="PCA (whitening only)")
    ax.bar(x + .2, [sc[n]["ica_si_sdr"] for n in names], .4, label="FastICA (ours)")
    ax.set_xticks(x, names); ax.set_ylabel("SI-SDR (dB, higher is better)")
    ax.axhline(0, color="k", lw=.6); ax.legend(); ax.set_title("PCA vs ICA separation quality")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "pca_vs_ica.png"), dpi=110); plt.close(fig)

    # SNR sweep
    sw = results["exp4_snr_sweep"]
    snrs = sorted(sw, key=float, reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar([float(s) for s in snrs], [sw[s]["mean_si_sdr"] for s in snrs],
                yerr=[sw[s]["std"] for s in snrs], marker="o", capsize=3)
    ax.invert_xaxis(); ax.set_xlabel("Sensor SNR of mixtures (dB)")
    ax.set_ylabel("Mean SI-SDR of estimates (dB)"); ax.set_title("Separation quality vs sensor noise")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "snr_sweep.png"), dpi=110); plt.close(fig)

    # Kurtosis + histograms (non-Gaussianity / CLT)
    k = results["kurtosis"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(list(k), list(k.values()), color="tab:purple")
    ax[0].set_ylabel("excess kurtosis (0 = Gaussian)"); ax[0].set_title("Non-Gaussianity")
    ax[0].tick_params(axis="x", rotation=30)
    bins = np.linspace(-5, 5, 120)
    ax[1].hist(S[0] / S[0].std(), bins, density=True, alpha=.6, label="source: speech")
    ax[1].hist(X[0] / X[0].std(), bins, density=True, alpha=.6, label="mixture 1")
    g = np.exp(-bins ** 2 / 2) / np.sqrt(2 * np.pi)
    ax[1].plot(bins, g, "k--", label="Gaussian")
    ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("Mixing makes signals more Gaussian (CLT)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "non_gaussianity.png"), dpi=110); plt.close(fig)

    # Convergence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(e1["r"]["history"], marker=".")
    ax.set_xlabel("iteration"); ax.set_ylabel("max |1 - |<w_new, w_old>||")
    ax.set_title("FastICA convergence (experiment 1)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "convergence.png"), dpi=110); plt.close(fig)


def write_markdown(res):
    L = ["# Results\n", "_Generated by `python src/run_experiments.py`._\n"]
    for key, title in [("exp1_3src_3mix", "Experiment 1 — 3 sources → 3 mixtures"),
                       ("exp2_3src_4mix", "Experiment 2 — 3 sources → 4 mixtures (overdetermined)"),
                       ("exp3_4src_4mix", "Experiment 3 — 4 sources (two speakers) → 4 mixtures")]:
        e = res[key]
        L.append(f"\n## {title}\n\nFastICA converged in **{e['iterations']}** iterations.\n")
        L.append("| Source | ICA \\|corr\\| | ICA SI-SDR (dB) | PCA \\|corr\\| | PCA SI-SDR (dB) |")
        L.append("|---|---|---|---|---|")
        for n, s in e["scores"].items():
            L.append(f"| {n} | {s['ica_corr']:.4f} | {s['ica_si_sdr']:.1f} | {s['pca_corr']:.3f} | {s['pca_si_sdr']:.1f} |")
        if "pca_eigenvalues" in e:
            ev = ", ".join(f"{v:.3g}" for v in e["pca_eigenvalues"])
            L.append(f"\nAll 4 covariance eigenvalues: {ev}. With 4 microphones and only 3 sources the 4th "
                     "eigenvalue is ~0 (no new information), so whitening keeps only the top 3 dimensions.")
    L.append("\n## Experiment 4 — sensor-noise sweep (mean over 5 seeds)\n")
    L.append("| Mixture SNR (dB) | Mean SI-SDR (dB) | Std |\n|---|---|---|")
    for snr, v in sorted(res["exp4_snr_sweep"].items(), key=lambda kv: -float(kv[0])):
        L.append(f"| {snr} | {v['mean_si_sdr']:.1f} | {v['std']:.2f} |")
    L.append("\n## Experiment 5 — non-linearity G\n\n| G | Mean SI-SDR (dB) | Iterations |\n|---|---|---|")
    for f, v in res["exp5_nonlinearity"].items():
        L.append(f"| {f} | {v['mean_si_sdr']:.1f} | {v['iterations']} |")
    L.append("\n## Excess kurtosis (0 = Gaussian)\n\n| Signal | Kurtosis |\n|---|---|")
    for n, v in res["kurtosis"].items():
        L.append(f"| {n} | {v:.2f} |")
    L.append("\n## Plots\n")
    for p in ["waveforms", "scatter_stages", "pca_vs_ica", "non_gaussianity", "snr_sweep", "convergence"]:
        L.append(f"![{p}](plots/{p}.png)\n")
    with open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
