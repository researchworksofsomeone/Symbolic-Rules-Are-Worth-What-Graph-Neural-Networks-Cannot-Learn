# Symbolic Rules Are Worth What Graph Neural Networks Cannot Learn

**A learnability theory of neuro-symbolic gains in GNNs, and out-of-fold discovery of the missing rule.**

This repository contains the full experimental notebook (`ACML.ipynb`) behind the paper *"Symbolic Rules Are Worth What Graph Neural Networks Cannot Learn: A Learnability Theory and Out-of-Fold Discovery of Missing Rules"* (under review; authors anonymised). Every number in the paper comes from real public datasets. No synthetic data is used for any reported result; synthetic graphs appear only as smoke tests and controls.

---

## Contents

1. [The idea in one paragraph](#the-idea-in-one-paragraph)
2. [Theory](#theory)
3. [The discovery method](#the-discovery-method)
4. [Main results](#main-results)
5. [Datasets and download links](#datasets-and-download-links)
6. [Notebook map](#notebook-map)
7. [How to run](#how-to-run)
8. [Reproducibility notes](#reproducibility-notes)
9. [Limitations](#limitations)
10. [Citation](#citation)

---

## The idea in one paragraph

Neuro-symbolic graph learning adds hand-designed knowledge, such as cycle counts, motif counts or probabilistic label constraints, to graph neural networks. The usual justification is expressive power: message passing is bounded by the 1-Weisfeiler–Leman (1-WL) test, so structure that 1-WL cannot see should be supplied symbolically. We argue that expressivity is the wrong yardstick. A rule is worth exactly the label information that the **trained** network has failed to learn and that the rule restores. We prove this with three exact identities and show it on real benchmarks. The theory then gives a procedure: read the missing rule out of the network's own **out-of-fold** errors.

---

## Theory

Notation. A graph $`G`$ has label $`Y`$. A symbolic rule is a function $`S=s(G)`$. A model family $`\mathcal F`$ has *computable information* $`\sigma_{\mathcal F}`$, the σ-algebra of everything its representations can compute. A trained network has readout input $`T`$ (here, the concatenated per-layer sum-pools). All losses are log-loss in nats.

### Theorem 1: absorption identity

For the family of all $`\sigma`$-measurable predictors, the largest log-loss reduction a rule can give is

```math
\Delta^\star(S)=I(Y;S\mid\sigma)=I(Y;S)-I(S;\sigma)+I(S;\sigma\mid Y),
```

with computable bounds

```math
\big[I(Y;S)-I(S;\sigma)\big]_+\ \le\ \Delta^\star(S)\ \le\ \min\{H(S\mid\sigma),\,H(Y\mid\sigma)\}.
```

The gap between the upper bound $`H(S\mid\sigma)`$ and the unclipped lower bound is exactly $`H(S\mid Y)`$.

**Corollary (expressible knowledge is worthless).** If $`S`$ is $`\sigma`$-measurable, it is worth zero. For a $`t`$-layer GIN this holds for every rule whose WL ambiguity $`\alpha_t(s)`$ is zero on the data, *including rules that 1-WL cannot compute on general graphs*.

### Theorem 2: gain-ledger identity (neural debt)

Real networks are not Bayes-optimal. Define the *neural debt* $`\varepsilon_{\mathcal F}=H_{\mathcal F}(Y\mid G)-H(Y\mid\sigma_{\mathcal F})\ge 0`$. Then, for every family and rule,

```math
\Delta_{\mathcal F}(S)=\underbrace{I(Y;S\mid\sigma_{\mathcal F})}_{\text{representational}}+\underbrace{\varepsilon_{\mathcal F}-\varepsilon_{\mathcal F\oplus S}}_{\text{debt repaid}},\qquad 0\le\Delta_{\mathcal F}(S)\le I(Y;S\mid\sigma_{\mathcal F})+\varepsilon_{\mathcal F}.
```

A rule the family can already compute can only repay debt. Every nat of debt that a better architecture repays itself, for example a better readout, removes one nat from what any such rule can add.

### Theorem 3: learnability decomposition

For one trained network with representation $`T`$, where $`\sigma(T)\subseteq\sigma_{\mathcal F}`$,

```math
I(Y;S\mid T)=I(Y;S\mid\sigma_{\mathcal F})+\underbrace{I(Y;\sigma_{\mathcal F}\mid T)-I(Y;\sigma_{\mathcal F}\mid T,S)}_{\text{information lost in training and restored by }S}.
```

Expressivity is therefore neither necessary nor sufficient for a rule to pay off. A rule beyond 1-WL that the network has decoded anyway is worth zero. A rule inside 1-WL that training missed can be worth a lot.

### Proposition 1: Fano certificate

If a probe $`\hat S=\varphi(T)`$ recovers each discrete rule column with error $`P_{e,c}`$, then

```math
I(Y;S\mid T)\le H(S\mid T)\le\sum_c\big[h_b(P_{e,c})+P_{e,c}\log(|\mathcal S_c|-1)\big].
```

A rule the trained network decodes exactly is worth zero. This can be checked before training any rule model.

### Proposition 2: local value of a block (the discovery score)

Let $`r`$ be the network's residual and $`\tilde C_B`$ the part of a block of candidate rules that $`T`$ does not predict. With an orthonormal basis $`Q_B`$ of $`\tilde C_B`$, the block energy $`E_B=\lVert Q_B^\top r\rVert_F^2/n`$ is:

- exact in the Gaussian case, where $`I = -\tfrac12\log(1-E_B/\mathrm{Var}\,r)`$;
- the second-order log-loss reduction for logistic models, about $`E_B/(2w)`$ when the curvature $`p(1-p)\approx w`$.

In multiclass, non-Gaussian and finite-sample settings it is a **ranking** surrogate. That is why significance only *nominates* rules and validation loss *decides* how many to keep.

### Proposition 3: why residuals must be out-of-fold

The in-sample residual of a network that fits its training graphs carries a memorisation term $`\omega`$. When the network interpolates, the evidence for every candidate collapses. Residuals from networks that never saw the graph estimate the true population residual covariance. This is the cross-fitting principle of double machine learning, applied to rule discovery.

Full proofs are in the paper's appendix. A short version is in the notebook cell *"Proofs (short, complete)"*.

---

## The discovery method

```
1. Split the training graphs into two halves; for 2 seeds, train the plain GNN on each half.
2. For every training graph, take the residual r_i of the networks that did NOT see it (out-of-fold),
   and the part of each candidate rule that a cross-fitted probe cannot predict from T.
3. For every block of candidates (every single column and every rule family):
      E_B = ||Q_B^T r||^2 / n
   Calibrate it against 1000 residual permutations with a moment-matched scaled chi-square
   (a = Var/2mu, nu = 2mu^2/Var), giving a surprise s_B on a common scale.
4. Pick the block with the largest surprise; family-wise p-value by Westfall-Young max-T.
   Project the residual off the picked span and repeat (group orthogonal matching pursuit).
5. Retrain the network with rule branches for the nested picks; keep the set with the best
   validation loss.
```

The candidate library is generated automatically: about 300 ring statistics for molecules, 26 graph statistics for social graphs, plus the classical rules as distractors. It runs on one GPU in about an hour per experiment.

---

## Main results

| Finding | Evidence |
|---|---|
| **Expressible knowledge is absorbed** | One round of 1-WL makes six beyond-1-WL motif rules unambiguous on 7 TU benchmarks (max 0.12%, 0% after two rounds). GNNs absorb a median **99%** of a belief-propagation layer's value on 11 node-classification graphs. Under random splits, classical rules add **−0.007 nats** on average over a noise-input control. |
| **Realised gains are debt repayment** | Where GIN still gains from 1-WL-computable rules, an MLP readout removes 80% (COLLAB) and 83% (NCI1) of the gain. |
| **Learnability beats expressivity** | On ZINC, bond counts invisible to GIN are decoded from atoms ($`R^2_T=0.86`$) and worth nothing. Poorly learned ring counts cut test MAE by **27–32%**. |
| **Discovery finds the hidden term** | On ZINC the first pick is $`\max(0,\ell_{\max}-6)`$, the ring penalty in the penalised-logP target, in **three independent runs** ($`p_{\text{adj}}=0.001`$). MAE is **0.1196** vs 0.1306 for expert cycle counts. |
| **Strong baselines** | Plain GINE reaches 0.7588 ROC-AUC on ogbg-molhiv, above the published OGB GIN (0.7558). Discovery adds +0.021 [+0.010, +0.031]. |
| **Held-out, frozen method** | On 5 unseen datasets (molbace, molbbbp, PROTEINS, IMDB-B, REDDIT-B; 10 seeds), the discovered set is never significantly worse than the plain network or the expert rule. On BBBP it significantly beats both: +0.035 ROC-AUC over plain and +0.016 over the expert. On the other four the point estimates lie between −0.012 and −0.002. |

---

## Datasets and download links

All datasets are public. The notebook downloads each one itself; no manual download is needed.

| Dataset(s) | Task | Source |
|---|---|---|
| MUTAG, PTC, PROTEINS, NCI1, IMDB-BINARY, IMDB-MULTI, COLLAB, REDDIT-BINARY | graph classification | TUDataset: https://chrsmrrs.github.io/datasets/ (the notebook uses the GIN authors' release: https://github.com/weihua916/powerful-gnns) |
| ZINC-12k (penalised logP) | graph regression | https://github.com/graphdeeplearning/benchmarking-gnns |
| ogbg-molhiv, ogbg-molbace, ogbg-molbbbp (scaffold splits) | molecular property prediction | Open Graph Benchmark: https://ogb.stanford.edu/docs/graphprop/ |
| Cora, CiteSeer, PubMed | node classification | https://github.com/kimiyoung/planetoid |
| Amazon-Photo, Amazon-Computers | node classification | https://github.com/shchur/gnn-benchmark |
| Actor | node classification | https://github.com/graphdml-uiuc-jlu/geom-gcn |
| Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions | node classification | https://github.com/yandex-research/heterophilous-graphs |

Optional precomputed results, caches and figures are mirrored on the Hugging Face Hub at `<HF_REPO>` (link to be added after review).

---

## Notebook map

`ACML.ipynb` was written for [marimo](https://marimo.io) / molab and exported to Jupyter format. The cells after Part II are **standalone**: each one has its own imports, downloads, caches and verdicts, and can be run on its own.

| Cells | What they do |
|---|---|
| 1–3 | Environment, resource report, configuration (`SMOKE=1` runs a tiny CPU smoke test) |
| 4–9 | Node-level data loaders (no torch_geometric), graph utilities, MLP/GCN/SAGE, the symbolic compatibility/belief-propagation layer, metrics, memory plan |
| 10–24 | Node-level experiments: resumable heavy stage, tables with confidence intervals, absorption (H5), information ceilings (H2, H6), figures, pre-registered verdicts |
| G1–G3 | TU graph data, GIN/DeepSets with symbolic readout features, per-fold units |
| G4–G6 | Graph-level absorption and the **exact 1-WL ambiguity analysis** |
| G7–G10 | Readout test (linear vs MLP) and the **gain ledger** with CIs and shuffled/size-only controls |
| Proofs | Short complete proofs of the theorems |
| L1–L2 | **Learnability law** on ZINC, NCI1 and PROTEINS: probes, $`R^2_T`$, Fano certificates, verdicts |
| D1–D6 | Discovery development: ring candidates on ZINC, permutation certification, molhiv with all OGB features, COLLAB |
| D7–D8 | **Final method**: out-of-fold block discovery with scaled-χ² calibration and Westfall–Young p-values (ZINC, molhiv, COLLAB) |
| D9 | **Held-out confirmation**: the frozen D7 method on molbace, molbbbp, PROTEINS, IMDB-B, REDDIT-B |
| T1 | Random vs distribution-shifted splits (scaffold, small-to-large graphs) with a noise-rule placebo |
| HF cells | Optional push, check, restore and save-everything helpers for a Hugging Face mirror |

Some cells appear twice: earlier versions are kept for provenance, and the later copy is the one used in the paper.

---

## How to run

**Requirements.** Python ≥ 3.10, PyTorch with CUDA, numpy, scipy, pandas, matplotlib, networkx, rdkit (for the molecular cells), ogb, and marimo (optional). `huggingface_hub` is needed only for the optional mirror cells.

```bash
pip install torch numpy scipy pandas matplotlib networkx rdkit ogb marimo
# optional mirror helpers (keep <1.0 if you also use transformers/datasets)
pip install "huggingface_hub>=0.34,<1.0"
```

**Run.**

```bash
jupyter lab ACML.ipynb        # or convert to marimo: marimo convert ACML.ipynb -o acml.py && marimo edit acml.py
```

- Each experiment cell (G4–G10, L1, D1–D9, T1) is sized to finish in about an hour on one 48–96 GB GPU and checkpoints after every unit, so it can be resumed.
- Set `SMOKE=1` to check the pipeline end to end on CPU in minutes.
- **Hugging Face cells are optional.** They read the token from the `HF_TOKEN` environment variable. The placeholder `<YOUR_HF_TOKEN>` must never be replaced with a real token in a committed file. Set the target repo in `HF_REPO`.

---

## Reproducibility notes

- All comparisons are paired by seed. Intervals are 95% percentile-bootstrap intervals over seeds (4000 resamples).
- OGB checkpoints are selected by validation ROC-AUC, following the OGB protocol. Others are selected by validation loss.
- Discovery uses 2 seeds × 2 folds of half-data networks, 1000 permutations and 3 rounds. The number of kept blocks is chosen on validation only.
- Development datasets (ZINC, molhiv, COLLAB) and held-out datasets (molbace, molbbbp, PROTEINS, IMDB-B, REDDIT-B) are kept strictly apart. The method was frozen before the held-out run.
- In OGB atom-type codes, atomic number minus one: 5 = C, 6 = N, 7 = O. Bond type 3 = aromatic.

---

## Limitations

- Large gains concentrate where a structural term is unlearned (ring-dependent molecular targets). On social and protein graphs, trained GINs leave little usable debt.
- Selection mainly buys parsimony and interpretability. The whole candidate pool sometimes matches it, but it can also hurt (molhiv −0.039).
- Significance nominates rules but does not measure their size.
- Value under distribution shift is not yet predicted by any in-distribution quantity.
- The Fano certificate is valid but loose. It certifies worthlessness; it does not estimate value.
- A discovered rule is a residual correlate of one trained network, **not a causal mechanism**. Treat named chemotypes as hypotheses for domain experts.

---

## Citation

The citation will be added after the review process. Until then, please cite this repository.

## License

Code is released under the MIT License. Datasets are subject to their original licenses (see the links above).
