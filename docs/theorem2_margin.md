**Theorem 2 (Margin preservation and risk upper bound of PLFC).**  
Let $K < 0$ be a fixed curvature and let a sample $x$ be mapped by the penultimate layer to a vector of real-valued scores
$$
u(x) = (u_1(x), \dots, u_m(x)) \in \mathbb{R}^m,
$$
where $u_c(x)$ is the score of the true class $c$. Define the *pre-logit margin*
$$
\Delta(x) := u_c(x) - \max_{j \neq c} u_j(x).
$$

Consider two output-layer designs that both use signed geodesic distances from $x$ to a family of coordinate hyperplanes as logits:

1. **PLFC head (intrinsic design).**  
   The spatial coordinate of the Lorentz point is chosen as
   $$
   y_{s,k}^{\mathrm{PLFC}}(x)
   = \frac{1}{\sqrt{-K}} \sinh\big(\sqrt{-K}\,u_k(x)\big),
   $$
   and the signed Lorentzian geodesic distance from $y(x)$ to the $k$-th coordinate hyperplane is
   $$
   d_k^{\mathrm{PLFC}}(x)
   = \frac{1}{\sqrt{-K}} \operatorname{asinh}\big(\sqrt{-K}\,y_{s,k}^{\mathrm{PLFC}}(x)\big).
   $$

2. **LFC head (extrinsic / linear design).**  
   The spatial coordinate is taken directly as
   $$
   y_{s,k}^{\mathrm{LFC}}(x) = u_k(x),
   $$
   and the signed Lorentzian geodesic distance to the same hyperplane is
   $$
   d_k^{\mathrm{LFC}}(x)
   = \frac{1}{\sqrt{-K}} \operatorname{asinh}\big(\sqrt{-K}\,u_k(x)\big).
   $$

Define the *distance-based margins*
$$
\Delta^{\mathrm{PLFC}}(x) := d_c^{\mathrm{PLFC}}(x) - \max_{j \neq c} d_j^{\mathrm{PLFC}}(x),
$$
$$
\Delta^{\mathrm{LFC}}(x) := d_c^{\mathrm{LFC}}(x) - \max_{j \neq c} d_j^{\mathrm{LFC}}(x).
$$

Then the following properties hold for every sample $x$ and every $K < 0$:

1. (**Exact margin preservation of PLFC**)
   $$
   \Delta^{\mathrm{PLFC}}(x) = \Delta(x).
   $$

2. (**Margin contraction of LFC**)
   $$
   \Delta^{\mathrm{LFC}}(x) \le \Delta(x) = \Delta^{\mathrm{PLFC}}(x),
   $$
   with strict inequality for generic inputs with nonzero margin.

3. (**Cross-entropy upper bounds**)  
   Let the softmax cross-entropy loss of a single example be
   $$
   \ell(x) :=
   -\log\frac{\exp(d_c(x))}{\sum_{k=1}^{m} \exp(d_k(x))}
   = \log\Big(1 + \sum_{j \neq c} \exp\big(d_j(x) - d_c(x)\big)\Big),
   $$
   where $d_k$ denotes the distance-based logit used by a chosen head.
   Then
   $$
   \ell_{\mathrm{PLFC}}(x)
   \le \log\big(1 + (m-1)\,e^{-\Delta(x)}\big)
   \le (m-1)\,e^{-\Delta(x)},
   $$
   and
   $$
   \ell_{\mathrm{LFC}}(x)
   \le \log\big(1 + (m-1)\,e^{-\Delta^{\mathrm{LFC}}(x)}\big)
   \le (m-1)\,e^{-\Delta^{\mathrm{LFC}}(x)}.
   $$
   Since $\Delta^{\mathrm{PLFC}}(x) = \Delta(x) \ge \Delta^{\mathrm{LFC}}(x)$, the PLFC head admits a strictly tighter (smaller) exponential upper bound on the loss than the LFC head, given the same pre-logits $u(x)$.

---

*Proof.*

Define the scalar function
$$
h(t) := \frac{1}{\sqrt{-K}} \operatorname{asinh}\big(\sqrt{-K}\,t\big), \qquad K < 0.
$$
By construction, for any Lorentz point whose $k$-th spatial coordinate is $y_{s,k}$, the signed geodesic distance to the $k$-th coordinate hyperplane is exactly $h(y_{s,k})$.

**Step 1: PLFC yields identity distances.**

For the PLFC head, we have
$$
y_{s,k}^{\mathrm{PLFC}}(x)
= \frac{1}{\sqrt{-K}} \sinh\big(\sqrt{-K}\,u_k(x)\big).
$$
Plugging this into the distance formula,
$$
d_k^{\mathrm{PLFC}}(x)
= h\big(y_{s,k}^{\mathrm{PLFC}}(x)\big)
= \frac{1}{\sqrt{-K}} \operatorname{asinh}\Big(\sqrt{-K} \cdot \frac{1}{\sqrt{-K}} \sinh\big(\sqrt{-K}\,u_k(x)\big)\Big)
= \frac{1}{\sqrt{-K}} \operatorname{asinh}\big(\sinh(\sqrt{-K}\,u_k(x))\big).
$$
Using the standard identity
$$
\operatorname{asinh}(\sinh z) = z \quad \text{for all } z \in \mathbb{R},
$$
we obtain
$$
d_k^{\mathrm{PLFC}}(x) = u_k(x).
$$
Therefore, for the PLFC head,
$$
\Delta^{\mathrm{PLFC}}(x)
= d_c^{\mathrm{PLFC}}(x) - \max_{j \neq c} d_j^{\mathrm{PLFC}}(x)
= u_c(x) - \max_{j \neq c} u_j(x)
= \Delta(x),
$$
which proves Claim (1).

---

**Step 2: LFC contracts the margin.**

For the LFC head, by definition
$$
y_{s,k}^{\mathrm{LFC}}(x) = u_k(x)
\quad \Rightarrow \quad
d_k^{\mathrm{LFC}}(x) = h\big(u_k(x)\big).
$$

First, we compute the derivative of $h$:
$$
h'(t)
= \frac{d}{dt}\left[\frac{1}{\sqrt{-K}} \operatorname{asinh}\big(\sqrt{-K}\,t\big)\right]
= \frac{1}{\sqrt{-K}} \cdot \frac{\sqrt{-K}}{\sqrt{1 + (-K)t^2}}
= \frac{1}{\sqrt{1 + (-K)t^2}}.
$$
Because $(-K) > 0$, the denominator satisfies $\sqrt{1 + (-K)t^2} \ge 1$, hence
$$
0 < h'(t) \le 1 \quad \text{for all } t \in \mathbb{R}.
$$
Thus $h$ is strictly increasing and 1-Lipschitz. In particular, for any $a > b$,
$$
h(a) - h(b)
= \int_{b}^{a} h'(t)\,dt
\le \int_{b}^{a} 1\,dt
= a - b.
$$

Now consider the margin of the LFC head. For any $j \neq c$,
$$
h\big(u_c(x)\big) - h\big(u_j(x)\big)
\le u_c(x) - u_j(x).
$$
Taking the maximum over all $j \neq c$ on both sides (noting that the left-hand side becomes $\ge \Delta^{\mathrm{LFC}}(x)$, while the right-hand side becomes $\Delta(x)$), we obtain
$$
\Delta^{\mathrm{LFC}}(x)
= h\big(u_c(x)\big) - \max_{j \neq c} h\big(u_j(x)\big)
\le u_c(x) - \max_{j \neq c} u_j(x)
= \Delta(x).
$$
Combining this with Step 1 yields
$$
\Delta^{\mathrm{LFC}}(x)
\le \Delta(x) = \Delta^{\mathrm{PLFC}}(x),
$$
which proves Claim (2). Strict inequality holds generically whenever the margins are nonzero and at least one competing logit is sufficiently large in magnitude, because then $h'(t) < 1$ on a nontrivial interval.

---

**Step 3: Cross-entropy upper bounds.**

Let the distance-based logits $\{d_k(x)\}_{k=1}^{m}$ be either $\{d_k^{\mathrm{PLFC}}(x)\}$ or $\{d_k^{\mathrm{LFC}}(x)\}$. The softmax cross-entropy loss of example $x$ with true class $c$ is
$$
\ell(x)
= -\log\frac{\exp(d_c(x))}{\sum_{k=1}^{m} \exp(d_k(x))}
= \log\left(1 + \sum_{j \neq c} \exp\big(d_j(x) - d_c(x)\big)\right).
$$
Define the distance-based margin
$$
\Delta_d(x) := d_c(x) - \max_{j \neq c} d_j(x).
$$
Then, for each $j \neq c$,
$$
d_j(x) - d_c(x) \le -\Delta_d(x),
$$
so
$$
\sum_{j \neq c} \exp\big(d_j(x) - d_c(x)\big)
\le \sum_{j \neq c} \exp\big(-\Delta_d(x)\big)
= (m-1)\,\exp\big(-\Delta_d(x)\big).
$$
Therefore,
$$
\ell(x)
\le \log\Big(1 + (m-1)\,\exp\big(-\Delta_d(x)\big)\Big)
\le (m-1)\,\exp\big(-\Delta_d(x)\big),
$$
where the second inequality uses $\log(1+z) \le z$ for all $z \ge 0$.

Applying this to the PLFC head, where $\Delta_d(x) = \Delta^{\mathrm{PLFC}}(x) = \Delta(x)$, we obtain
$$
\ell_{\mathrm{PLFC}}(x)
\le \log\big(1 + (m-1)\,e^{-\Delta(x)}\big)
\le (m-1)\,e^{-\Delta(x)}.
$$
For the LFC head, we similarly get
$$
\ell_{\mathrm{LFC}}(x)
\le \log\big(1 + (m-1)\,e^{-\Delta^{\mathrm{LFC}}(x)}\big)
\le (m-1)\,e^{-\Delta^{\mathrm{LFC}}(x)}.
$$

Combining with Claim (2), $\Delta^{\mathrm{PLFC}}(x) = \Delta(x) \ge \Delta^{\mathrm{LFC}}(x)$, shows that for a fixed pre-logit vector $u(x)$, the PLFC head admits a strictly tighter exponential upper bound on the cross-entropy loss than the LFC head.

This proves Claim (3) and completes the proof of the theorem. $\square$
