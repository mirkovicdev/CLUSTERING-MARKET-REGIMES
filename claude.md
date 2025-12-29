# Clustering Market Regimes Using the Wasserstein Distance

## Paper Reference
**Authors**: B. Horvath, Z. Issa, A. Muguruza
**Date**: October 25, 2021
**arXiv**: 2110.11848v1

---

## 1. Introduction and Problem Statement

### 1.1 The Market Regime Clustering Problem (MRCP)

Financial time series exhibit **stylised facts** that are ubiquitous across asset classes:
- Return series are non-stationary in the strong sense
- Exhibit volatility clustering
- Periods of similar behaviour followed by distinct periods indicating significantly different underlying distributions

These periods are called **market regimes**. The task of finding an effective way of grouping different regimes is the **Market Regime Clustering Problem (MRCP)**.

### 1.2 Key Insight: Lifting to Probability Space

The main idea is to **lift the regime clustering problem from Euclidean space to the space of probability distributions** with finite p-th moment.

Given a sequence of return segments $(r_i)_{i \geq 0}$ where:
$$r_i = (r_i^1, \ldots, r_i^n) \text{ for } n \in \mathbb{N}$$

Any vector $r_i \in \mathbb{R}^n$ can be associated to an **empirical measure**:
$$\delta_{r_i} = \frac{1}{n} \sum_{j=1}^{n} \delta_{r_i^j} \text{ for } i \geq 0$$

with $n$ atoms. Thus, the problem of classifying market regimes is equivalent to assigning labels to probability measures $\mu \in \mathcal{P}_p(\mathbb{R})$, where $\mathcal{P}_p(\mathbb{R})$ is the set of probability measures on $\mathbb{R}$ with finite p-th moment.

---

## 2. Mathematical Framework

### 2.1 Data Streams

**Definition 1.1 (Set of data streams)**. Let $\mathcal{X}$ be a non-empty set. The set of streams of data $\mathcal{S}$ over $\mathcal{X}$ is given by:
$$\mathcal{S}(\mathcal{X}) = \{x = (x_1, \ldots, x_n) : x_i \in \mathcal{X}, n \in \mathbb{N}\}$$

In this paper, $\mathcal{X} = \mathbb{R}$ and we fix $N \in \mathbb{N}$.

### 2.2 Log Returns

Given $S \in \mathcal{S}(\mathbb{R})$ (a price path), define the vector of **log-returns** $r^S$ associated to $S$ by:
$$r_i^S = \log(s_{i+1}) - \log(s_i) \quad \text{for } 0 \leq i \leq N-1$$

### 2.3 Stream Lift (Sliding Window)

**Definition 1.2 (Stream lift)**. For $x \in \mathcal{S}(\mathbb{R})$ and $h_1, h_2 \in \mathbb{N}$ with $h_1 > h_2$, define a lift $\ell := \ell_{h_1,h_2}$ from $\mathcal{S}(\mathbb{R})$ to $\mathcal{S}(\mathcal{S}(\mathbb{R}))$ via:
$$\ell^i(x) = (x_{1+h_2(i-1)}, \ldots, x_{1+h_1+h_2(i-1)}) \quad \text{for } i = 1, \ldots, M$$

where:
- $M := \lfloor \frac{N}{h_1-h_2} \rfloor$ is the maximum number of partitions with length $h_1$
- $h_1$ = **window length** (number of returns per segment)
- $h_2$ = **sliding window offset parameter** (overlap between consecutive windows)

### 2.4 Empirical Measure

**Definition 1.3 (Empirical measure)**. Let $x \in \mathcal{S}(\mathbb{R})$ such that $x = (x_1, \ldots, x_N)$ for $N \in \mathbb{N}$. Let $Q^j : \mathcal{S}(\mathbb{R}) \to \mathbb{R}$ be the function which extracts the j-th order statistic of $x$, for $j = 1, \ldots, N$. Then, the cumulative distribution function of the empirical measure $\mu \in \mathcal{P}_p(\mathbb{R})$ associated to $x$ is defined as:
$$\mu^x((-\infty, x]) = \frac{1}{N} \sum_{i=1}^{N} \chi_{\{Q^i(x) \leq x\}}(x)$$

where $\chi : \mathbb{R} \to [0,1]$ is the indicator function.

Thus, we can associate to each segment of data $r_i \in \ell(r^S)$ the empirical measure $\mu_i$ for $i = 1, \ldots, M$. This gives us a family of measures:
$$\mathcal{K} = \{(\mu_1, \ldots, \mu_M) : \mu_i \in \mathcal{P}_p(\mathbb{R}) \text{ for } i = 1, \ldots, M\}$$

**This family $\mathcal{K}$ is the subject of our clustering algorithm.**

---

## 3. The Wasserstein Distance

### 3.1 Definition

**Definition 2.1 (p-Wasserstein distance)**. Suppose $(X, d)$ is a separable Radon space. The p-th Wasserstein distance between measures $\mu, \nu \in \mathcal{P}_p(X)$ is defined by:
$$\mathcal{W}_p^p(\mu, \nu) := \min_{\mathbb{P} \in \Pi(\mu,\nu)} \left\{ \int_{X \times X} d(x,y)^p \, \mathbb{P}(dx, dy) \right\}$$

where:
$$\Pi(\mu, \nu) := \{\mathbb{P} \in \mathcal{P}(X \times X) : \mathbb{P}(A \times X) = \mu(A), \mathbb{P}(X \times B) = \nu(B)\}$$

is the set of **transport plans** between $\mu$ and $\nu$.

### 3.2 Closed-Form Solution for 1D Distributions

**Proposition 2.5**. Suppose $\mu, \nu \in \mathcal{P}_p(\mathbb{R}^d)$ and let $d = 1$. Moreover, suppose that $\mu, \nu$ are absolutely continuous with respect to the Lebesgue measure on $\mathbb{R}$. Then, the p-Wasserstein distance is given by:
$$\mathcal{W}_p(\mu, \nu) = \left( \int_0^1 |F_\mu^{-1}(z) - F_\nu^{-1}(z)|^p \, dz \right)^{1/p}$$

where the **quantile function** $F_\mu^{-1} : [0,1) \to \mathbb{R}$ is defined as:
$$F_\mu^{-1}(z) = \inf\{x : F_\mu(x) > z\}$$

### 3.3 Wasserstein Distance for Empirical Measures

For empirical measures $\mu, \nu$ with equal numbers of atoms $N \in \mathbb{N}$:
$$\mu((-\infty, x]) = \frac{1}{N} \sum_{i=1}^{N} \chi_{\alpha_i \leq x}(x), \quad \nu((-\infty, x]) = \frac{1}{N} \sum_{i=1}^{N} \chi_{\beta_i \leq x}(x)$$

where $(\alpha_i)_{1 \leq i \leq N}$ and $(\beta_i)_{1 \leq i \leq N}$ are **increasing sequences** (sorted atoms).

The quantile function becomes:
$$F_\mu^{-1}(z) = \alpha_i \quad \text{for all } z \in \left[\frac{i-1}{N}, \frac{i}{N}\right), \quad i = 1, \ldots, N$$

**The Wasserstein distance between empirical measures is:**
$$\mathcal{W}_p(\mu, \nu)^p = \frac{1}{N} \sum_{i=1}^{N} |\alpha_i - \beta_i|^p$$

**Complexity**: $\mathcal{O}(N \log N)$ if atoms need sorting, $\mathcal{O}(N)$ if already sorted.

---

## 4. The Wasserstein Barycenter

### 4.1 Definition

**Definition 2.3 (Wasserstein barycenter)**. Suppose $(X, d)$ is a separable Radon space and let $\mathcal{K} = \{\mu_i\}_{i \geq 1} \subset \mathcal{P}(X)$ be a family of Radon measures. Define the p-Wasserstein barycenter $\bar{\mu}$ of $\mathcal{K}$ to be:
$$\bar{\mu} = \arg\min_{\nu \in \mathcal{P}(X)} \sum_{\mu_i \in \mathcal{K}} \mathcal{W}_p(\mu_i, \nu)$$

### 4.2 Barycenter for Empirical Measures (p=1)

**Proposition 2.6**. Suppose that $\{\mu_i\}_{1 \leq i \leq M}$ are a family of empirical probability measures, each with $N$ atoms $(\alpha_j^i)_{1 \leq j \leq N} \subset \mathbb{R}^N$. Let:
$$a_j = \text{Median}(\alpha_j^1, \ldots, \alpha_j^M) \quad \text{for } j = 1, \ldots, N$$

Then, the cumulative distribution function of the Wasserstein barycenter $\bar{\mu} \in \mathcal{P}_1(\mathbb{R})$ over $\{\mu_i\}_{1 \leq i \leq M}$ with respect to the 1-Wasserstein distance is given by:
$$\bar{\mu}((-\infty, x]) = \frac{1}{N} \sum_{i=1}^{N} \chi_{a_i \leq x}(x)$$

**Note**: For $p > 1$, use **Mean** instead of **Median**.

---

## 5. The WK-Means Algorithm

### 5.1 Algorithm Definition

**Definition 2.7 (WK-means algorithm)**. Let $\mathcal{K} \subset \mathcal{P}_p(\mathbb{R})$ be a family of measures with finite p-th moment. We refer to the k-means clustering algorithm on $(\mathcal{P}_p(\mathbb{R}), \mathcal{W}_p)$, with aggregation method given by the Wasserstein barycenter and loss function given by:
$$l(\bar{\mu}^{n-1}, \bar{\mu}^n) = \sum_{i=1}^{k} \mathcal{W}_p(\bar{\mu}_i^{n-1}, \bar{\mu}_i^n)$$

as the **Wasserstein k-means algorithm**, or **WK-means**.

### 5.2 Pseudocode

```
Algorithm 1: WK-means algorithm
Result: k centroids

1. Calculate ℓ(r^S) given S (sliding window decomposition)
2. Define family of empirical distributions K = {μ_j}_{1≤j≤M}
3. Initialize centroids μ̄_i, i = 1,...,k by sampling k times from K
4. while loss_function > tolerance do
       foreach μ_j do
           assign closest centroid wrt W_p to cluster C_l, l = 1,...,k
       end
       update centroid i as the Wasserstein barycenter relative to C_l
       calculate loss_function
   end
```

### 5.3 Nearest Neighbor Assignment

At each step $n \in \mathbb{N}$, calculate the nearest neighbours:
$$\mathcal{C}_l^n := \left\{ \mu_i \in \mathcal{K} : \arg\min_{j=1,\ldots,k} \mathcal{W}_p(\mu_i, \bar{\mu}_j^{n-1}) = l \right\}$$

associated to each $\bar{\mu}_j^{n-1}$ for $j = 1, \ldots, k$.

---

## 6. Validation: Maximum Mean Discrepancy (MMD)

### 6.1 Two-Sample Test Problem

**Problem 1.6 (Two-sample test)**. Let $(X, d)$ be a metric space. Suppose $X$ and $Y$ are independent random variables on $X$. If $X_{\#}\mathbb{P} = \mu$ and $Y_{\#}\mathbb{P} = \nu$, when can we determine if $\mu \neq \nu$?
$$H_0 : \mu = \nu \quad \text{against} \quad H_1 : \mu \neq \nu$$

### 6.2 MMD Definition

**Definition 1.7 (Maximum mean discrepancy)**. Let $\mathcal{F}$ be a class of functions $f : X \to \mathbb{R}$ and let $\mu, \nu$ be defined as above. Then, the MMD between $\mu$ and $\nu$ is defined as:
$$\text{MMD}[\mathcal{F}, \mu, \nu] := \sup_{f \in \mathcal{F}} \left( \mathbb{E}_\mu[f(x)] - \mathbb{E}_\nu[f(y)] \right)$$

### 6.3 Biased Empirical Estimate

Given samples $x = (x_1, \ldots, x_n)$ and $y = (y_1, \ldots, y_m)$:
$$\text{MMD}_b[\mathcal{F}, x, y] = \sup_{f \in \mathcal{F}} \left[ \frac{1}{n} \sum_{i=1}^{n} f(x_i) - \frac{1}{m} \sum_{j=1}^{m} f(y_j) \right]$$

### 6.4 Gaussian Kernel

The Gaussian kernel:
$$\kappa_G : \mathbb{R}^d \times \mathbb{R}^d \to [0, +\infty), \quad \kappa_G(x, y) = \exp\left( -\frac{\|x - y\|_{\mathbb{R}^d}^2}{2\sigma^2} \right)$$

is characteristic to the set of Borel measures on $X$ and makes the MMD a metric on $\mathcal{P}(X)$.

### 6.5 Kernel MMD Formula

Using the kernel, MMD can be computed as:
$$\text{MMD}^2[\mathcal{F}, \mu, \nu] = \mathbb{E}_{x,x' \sim \mu}[\kappa(x,x')] - 2\mathbb{E}_{x \sim \mu, y \sim \nu}[\kappa(x,y)] + \mathbb{E}_{y,y' \sim \nu}[\kappa(y,y')]$$

Biased empirical estimate:
$$\text{MMD}_b[\mathcal{F}, x, y] = \left[ \frac{1}{n^2} \sum_{i,j=1}^{n} \kappa(x_i, x_j) - \frac{2}{mn} \sum_{i,j=1}^{m,n} \kappa(x_i, y_j) + \frac{1}{m^2} \sum_{i,j=1}^{m} \kappa(y_i, y_j) \right]^{1/2}$$

### 6.6 Self-Similarity Score

**Definition 1.9 (Within-cluster self-similarity/homogeneity)**. For $n, m \in \mathbb{N}$, the self-similarity score associated to $X$ is:
$$\text{Sim}(X) = \text{Median}\left( (\text{MMD}_b^2[\mathcal{F}, x_i, y_i])_{1 \leq i \leq n} \right)$$

where $x_i = (x_1^i, \ldots, x_m^i)$ and $y_i = (y_1^i, \ldots, y_m^i)$ are samples drawn pairwise from $X$ for $i = 1, \ldots, n$.

---

## 7. Benchmark Algorithms

### 7.1 Moment k-means (MK-means)

**Definition 3.1 (Moment k-means)**. For $p \geq 1$, associate to each $\mu_i \in \mathcal{K}$ the $\mathbb{R}^p$-vector $\varphi^p(\mu_i)$ for $i = 1, \ldots, M$, where:
$$\varphi^p(\mu) = \left( \frac{1}{n!} \int_\mathbb{R} x^n \, \mu(dx) \right)_{1 \leq n \leq p}$$

is the truncated unstandardised p-th moment map.

**Important**: Each slice $\{\varphi^p(\mu_i)_j\}_{1 \leq i \leq M}$ must be **standardized** component-wise:
$$\mathbb{E}[\{x_j^i\}_{1 \leq i \leq N}] = 0 \quad \text{and} \quad \text{Var}(\{x_j^i\}_{1 \leq i \leq N}) = 1 \quad \text{for } j = 1, \ldots, d$$

### 7.2 Hidden Markov Model (HMM)

Classical approach assuming:
1. The latent state variable specifying the current regime is Markovian
2. The likelihood of observing a return given the latent state is parametric (often Gaussian)

---

## 8. Experimental Setup

### 8.1 Real Data Parameters
- **Data**: One-hourly log-returns of SPY index from 2005-01-03 to 2020-12-31
- **Hyperparameters**: $(h_1, h_2) = (35, 28)$
  - Partitions time-series into ~weeks
  - Adjacent partitions within one day of each other
- **Number of clusters**: $k = 2$ (bull/bear regimes)

### 8.2 Visualization

**Mean-Variance Scatter Plot**: Project each distribution $\mu \in \mathcal{K}$ onto $\mathbb{R}^2$ via:
$$f_p : \mathcal{P}_p(\mathbb{R}) \to \mathbb{R}^2, \quad \mu \mapsto \left( \sqrt{\text{Var}(\mu)}, \mathbb{E}[\mu] \right)$$

**Historical Coloring Plot**: Color time-series segments according to cluster membership.

### 8.3 Cluster Validation Indexes

**Davies-Bouldin Index** (lower is better):
$$DB\left( \{(\bar{x}_l, \mathcal{C}_l)\}_{l=1}^k \right) = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{d_i + d_j}{d(\bar{x}_i, \bar{x}_j)}$$

where $d_l = \frac{1}{|\mathcal{C}_l|} \sum_{x \in \mathcal{C}_l} d(x, \bar{x}_l)$.

**Dunn Index** (higher is better):
$$D\left( \{(\bar{x}_l, \mathcal{C}_l)\}_{l=1}^k \right) = \frac{\min_{1 \leq i,j \leq k} \underline{d}_{ij}}{\max_{1 \leq l \leq k} \bar{d}_l}$$

**Silhouette Coefficient** (between -1 and 1, higher is better):
$$S(i) = \frac{b_i - a_i}{\max(a_i, b_i)}$$

---

## 9. Synthetic Data Experiments

### 9.1 Accuracy Metrics

**Definition 3.7**. For a given vector of log-returns $r^S \in \mathcal{S}(\mathbb{R})$ and cluster assignments $\mathcal{C} = \{\mathcal{C}_l\}_{l=1}^k$:

**Regime-off accuracy score (ROFS)**:
$$\text{ROFS}(r^S, \mathcal{C}) = \frac{\sum_{r_i^S \in N} \bar{Y}_1^i}{\sum_{r_i^S \in N} \sum_{k=1,2} \bar{Y}_k^i}$$

**Regime-on accuracy score (RONS)**:
$$\text{RONS}(r^S, \mathcal{C}) = \frac{\sum_{r_i^S \in R} \bar{Y}_2^i}{\sum_{r_i^S \in R} \sum_{k=1,2} \bar{Y}_k^i}$$

**Total accuracy (TA)**:
$$\text{TA}(r^S, \mathcal{C}) = \frac{\sum_{r_i^S \in N} \bar{Y}_1^i + \sum_{r_i^S \in R} \bar{Y}_2^i}{\sum_{i=1}^{N-1} \sum_{k=1,2} \bar{Y}_k^i}$$

where $R$ is the set of regime change intervals and $N = \Delta \setminus R$ is the normal regime.

### 9.2 Geometric Brownian Motion

For gBm with parameters $\theta = (\mu, \sigma)$:
$$\ln S_t \sim \text{Normal}\left( (\mu - \sigma^2/2)t, \sigma^2 t \right) \quad \text{for all } t \geq 0$$

**Test parameters**:
- $\theta_{\text{bull}} = (0.02, 0.2)$
- $\theta_{\text{bear}} = (-0.02, 0.3)$
- $T = 20$ years, $r = 10$ regime changes
- $l_i = 0.5 \times 252 \times 7$ (half year)

### 9.3 Merton Jump Diffusion

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t + S_{t-} \, dJ_t \quad \text{for } t \geq 0$$

where:
$$J_t = \sum_{j=1}^{N_t} V_j - 1$$

with $N_t \sim \text{Po}(\lambda t)$ (Poisson) and $\ln(1 + V_j) \sim \text{Normal}(\gamma, \delta^2)$.

**Moments of log-returns**:
$$\mathbb{E}[R_t^M] = (\mu - \sigma^2/2) + \lambda \gamma \, dt$$
$$\text{Var}(R_t^M) = (\sigma^2 + \lambda(\delta^2 + \gamma^2)) \, dt$$

**Test parameters**:
- $\theta_{\text{bull}} = (0.05, 0.2, 5, 0.02, 0.0125)$
- $\theta_{\text{bear}} = (-0.05, 0.4, 10, -0.04, 0.1)$

---

## 10. Results Summary

### 10.1 Real Data (SPY)
- WK-means correctly identifies: GFC 2008, Eurozone crisis 2010, S&P downgrade 2011, Chinese crash 2015/16, COVID crash 2020
- MK-means only identifies major events (GFC, COVID)
- WK-means clusters are more self-similar (lower within-cluster MMD)

### 10.2 Synthetic Data - gBm

| Algorithm | Total | Regime-on | Regime-off | Runtime |
|-----------|-------|-----------|------------|---------|
| Wasserstein | 90.60% ± 5.81% | **87.24%** ± 4.11% | 91.72% ± 6.46% | 0.87s |
| Moment | 93.23% ± 0.41% | 74.83% ± 1.57% | **99.38%** ± 0.1% | 1.06s |
| HMM | 58.16% ± 7.11% | 41.51% ± 7.43% | 63.72% ± 11.94% | 0.58s |

### 10.3 Synthetic Data - Merton Jump Diffusion

| Algorithm | Total | Regime-on | Regime-off | Runtime |
|-----------|-------|-----------|------------|---------|
| Wasserstein | **91.28%** ± 4.08% | **86.87%** ± 3.1% | 92.76% ± 4.43% | 1.11s |
| Moment | 66.64% ± 3.42% | 27.25% ± 8.73% | 79.79% ± 7.40% | 1.71s |
| HMM | 75.05% ± 0.01% | 0.66% ± 0.04% | 99.87% ± 0.01% | 0.66s |

**Key Finding**: WK-means significantly outperforms alternatives for non-Gaussian distributions.

---

## 11. Hyperparameter Selection

### 11.1 Window Length $h_1$
- **Too large**: Regime changes may not be captured, detection is lagged
- **Too small**: Spurious classifications dominated by noise
- **Recommendation**: Depends on application; reasonable choices give robust results

### 11.2 Overlap Parameter $h_2$
- Larger $h_2$ (relative to $h_1$) increases the number of samples
- Useful in low-data environments
- In general, does not drastically affect centroids if $h_1$ is suitable

---

## 12. Implementation Notes

### 12.1 Key Steps

1. **Load price data** and compute log-returns
2. **Apply sliding window** with parameters $(h_1, h_2)$ to create segments
3. **Create empirical distributions** from each segment (sort atoms)
4. **Initialize centroids** by random sampling from distributions
5. **Iterate**:
   - Compute Wasserstein distance from each distribution to each centroid
   - Assign each distribution to nearest centroid
   - Update centroids using Wasserstein barycenter
   - Check convergence
6. **Visualize** results in mean-variance space and as historical coloring

### 12.2 Computational Complexity

- Wasserstein distance between empirical measures: $\mathcal{O}(N \log N)$
- Wasserstein barycenter (p=1): $\mathcal{O}(MN)$ where $M$ is cluster size
- Total per iteration: $\mathcal{O}(Mk \cdot N \log N)$

### 12.3 Libraries to Use

- **NumPy/SciPy**: Core numerical computations
- **POT (Python Optimal Transport)**: Wasserstein distance and barycenters
- **scikit-learn**: k-means initialization, HMM benchmarks
- **matplotlib/seaborn**: Visualization

---

## 13. Plots to Reproduce

1. **Mean-Variance Scatter Plot** (Figure 1)
   - x-axis: Standard deviation $\sqrt{\text{Var}(\mu)}$
   - y-axis: Mean $\mathbb{E}[\mu]$
   - Color by cluster membership
   - Mark centroids with crosses

2. **Historical Cluster Coloring** (Figure 2)
   - Plot price path over time
   - Color segments by cluster membership
   - Highlight known crisis periods

3. **MMD Histograms** (Figures 3, 4)
   - Between-cluster MMD distribution
   - Within-cluster MMD distribution
   - Compare WK-means vs MK-means

4. **Centroid Approximation** (Figures 8, 13)
   - Distribution of segment means with theoretical values
   - Distribution of segment variances with theoretical values
   - Dashed lines for centroid estimates

5. **Synthetic Path Examples** (Figures 5, 7, 9, 12)
   - gBm and Merton paths with regime changes highlighted
   - Historical coloring showing algorithm performance

6. **Hyperparameter Sensitivity** (Figures 14, 15)
   - Effect of $h_2$ on clustering
   - Effect of $h_1$ on accuracy scores

---

## 14. Key Equations Summary

| Concept | Formula |
|---------|---------|
| Log-returns | $r_i = \log(S_{i+1}) - \log(S_i)$ |
| Empirical measure CDF | $\mu^x((-\infty, x]) = \frac{1}{N} \sum_{i=1}^{N} \chi_{\{Q^i(x) \leq x\}}(x)$ |
| Wasserstein distance (empirical) | $\mathcal{W}_p(\mu, \nu)^p = \frac{1}{N} \sum_{i=1}^{N} \|\alpha_i - \beta_i\|^p$ |
| Wasserstein barycenter (p=1) | $a_j = \text{Median}(\alpha_j^1, \ldots, \alpha_j^M)$ |
| Loss function | $l(\bar{\mu}^{n-1}, \bar{\mu}^n) = \sum_{i=1}^{k} \mathcal{W}_p(\bar{\mu}_i^{n-1}, \bar{\mu}_i^n)$ |
| MMD (kernel form) | $\text{MMD}^2 = \mathbb{E}[\kappa(x,x')] - 2\mathbb{E}[\kappa(x,y)] + \mathbb{E}[\kappa(y,y')]$ |
| Gaussian kernel | $\kappa_G(x,y) = \exp(-\|x-y\|^2 / 2\sigma^2)$ |

---

## References

- [AGS05] Ambrosio, Gigli, Savare. *Gradient Flows: In Metric Spaces and in the Space of Probability Measures*.
- [GBR+12] Gretton et al. *A kernel two-sample test*. JMLR 2012.
- [KNS+19] Kolouri et al. *Generalized sliced wasserstein distances*. arXiv 2019.
