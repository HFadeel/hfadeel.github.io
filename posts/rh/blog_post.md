# Preference RL Is Hard: Whose Preference?

Reinforcement Learning from Human Feedback is the dominate paradigm for aligning large language models to human preferences and values. At its core lies a simple idea: learn what humans prefer by having them compare outputs, then train a model to maximize those preferences. But beneath this simple idea is a fundamental problem—**whose preferences are we actually capturing?**

The standard approach assumes all humans share a single underlying reward function. This assumption is mathematically convenient but empirically false. When we aggregate preferences from diverse annotators, we don't get a representative model—we get a compromised one that may satisfy no one.

---

## The Bradley-Terry-Luce Model

Most RLHF systems use the **Bradley-Terry-Luce (BTL)** model to learn reward functions from pairwise comparisons. Given two responses $a_w$(winner) and $a_l$ (loser) to a prompt x, BTL models the probability that an annotator prefers $a_w$ as:

$P(a_w \succ a_l | x) = \sigma(r^*(x, a_w) - r^*(x, a_l)) = \frac{e^{r^*(x, a_w)}}{e^{r^*(x, a_w)} + e^{r^*(x, a_l)}}$

where $\sigma$ is the logistic function and $r^*$ is a latent reward function. The model is trained via maximum likelihood estimation on a dataset of human-labeled preferences.

BTL it's simple, differentiable, and handles noisy labels gracefully through its probabilistic formulation. But it makes one critical assumption—**all annotators share the same reward function $r^*$.** This is where things break down.

Anyone who worked with annotators before know that this is a false assumption. For example, In my AV experience where we had to annotators label simulation and real driving sense as safe vs not safe or will this result in a collision or not, people vary widely in what they consider safe or not even with very strict guideline and training.

---

## What's Wrong with BTL? Three Core Problems

### 1. The Averaging Problem

When human preferences are genuinely diverse, BTL doesn't capture that diversity—it averages over it. Consider a scenario where half your annotators prefer detailed, technical responses while the other half prefer concise, accessible ones. BTL will learn a reward function that produces... medium-length, moderately technical responses that neither group actually wants.

This isn't just theoretically concerning. Recent work from MiCRo (Shen et al., EMNLP 2025) proves that when preferences follow a mixture distribution of diverse subgroups, **a single BTL model incurs an irreducible error**. You cannot fit a unimodal BTL to multimodal preference data without systematic bias.

More formally, if the true preference distribution is a mixture:

$P(a_w \succ a_l | x) = \sum_{k=1}^{K} P(z=k|x) \cdot P(a_w \succ a_l | x, z=k)$

where z indexes latent subpopulations, then standard BTL training will converge to something like the mean of these subpopulations.

### 2. Annotation Bias and Majority Dominance

The averaging problem becomes particularly pernicious when combined with imbalanced annotation populations. If 80% of your annotators come from one demographic or hold one set of values, the resulting reward model will largely ignore the remaining 20%.

The VPL paper (Poddar et al., NeurIPS 2024) illustrates this with an important example: imagine a college admissions chatbot where wealthy annotators have weak preferences about financial aid information, while low-income annotators strongly need it. Standard RLHF will learn to deprioritize financial aid discussions, actively harming the minority group.

### 3. The Need for Pluralistic Alignment

The deeper issue is philosophical: should AI systems have a single set of values, or should they adapt to diverse user needs? Many preference dimensions—like verbosity, formality, directness, and humor—have no objectively correct answer. Different users want different things.

Current RLHF approaches enforce a prescriptive set of values curated by a small set of AI researchers, typically lacking diversity. This creates models that work well for people similar to the annotators but fail to serve broader populations.

---

## Solutions: Moving Beyond Single-Reward RLHF

Before diving into current ideas, it’s important to keep in mind RLHF is an always going to be an open ended problem because it’s impossible to measure and quantify human preferences. But we will keep chipping away at it just like recommendation systems.

### Solution 1: Variational Preference Learning (VPL)

Instead of assuming a single reward function, VPL models preferences as conditioned on a latent user variable z. Given a few preference examples from a user, we infer their latent representation and predict personalized rewards.

We can think of VPL as learning a "preference space" where each user occupies a different location (kinda like VAE). When a new user provides a few preference examples, VPL figures out where they sit in this space and uses that position to predict what they'll like. So, rather than averaging all preferences into one reward function, VPL maintains the full diversity by conditioning on user identity. The averaging problem disappears because we're no longer trying to fit one function to everyone—we're fitting a family of functions indexed by the latent z.

VPL replaces the standard BTL likelihood with a latent-conditional version:

$P(a_w \succ a_l | x, z) = \sigma(r_\phi(x, a_w, z) - r_\phi(x, a_l, z))$

The latent z is inferred via variational inference. Given a set of N preference examples from a user, an encoder $q_\psi$ produces a distribution over z:

$z \sim q_\psi(z | \{(s_A^i, s_B^i, y^i)\}_{i=1}^N)$

VPL optimizes an evidence lower bound (ELBO).

VPL enables **active learning**—using the uncertainty in $q_\psi$ to select maximally informative preference queries. Experiments show this achieves the same personalization with half the queries compared to random selection.

**Results**: On pluralistic language datasets, VPL outperforms standard BTL by 10-25% in reward prediction accuracy when users have divergent preferences.

**Limitations**:

- **Cold start problem**: VPL needs preference examples from a user at inference time to infer their latent z. For brand new users with no interaction history, you're back to guessing.
- **Latent capacity**: The latent space must be expressive enough to capture meaningful preference variation, but not so large that it overfits. Finding the right dimensionality requires tuning.
- **Assumes continuous preference space**: VPL models users as points in a continuous latent space, which may not capture discrete preference "types" as naturally as mixture models.
- **Encoder quality depends on data diversity**: If training data lacks diverse annotators, the encoder won't learn to distinguish preference types it hasn't seen.

---

### Solution 2: MiCRo (Mixture Modeling + Context-aware Routing)

Instead of one reward function trying to please everyone, MiCRo learns K different "expert" reward functions—one might specialize in users who like detailed responses, another for users who prefer brevity, etc. When a query comes in, a router looks at the context (the prompt, user metadata, or a few calibration questions) and decides which expert(s) to consult. It's like having a panel of judges with different tastes, and choosing which judges' opinions matter for each case.

**Stage 1 – Mixture Modeling**: Instead of a single reward function, learn K reward functions $\{r_{\phi_k}\}_{k=1}^K$
with context-dependent mixture weights:

$P(a_w \succ a_l | x) = \sum_{k=1}^{K} f_{\psi,k}(x) \cdot \sigma(r_{\phi_k}(x, a_w) - r_{\phi_k}(x, a_l))$

where $f_\psi: \mathcal{X} \rightarrow \Delta_K$ is a router network outputting mixture weights.

**Stage 2 – Context-aware Routing**: Given additional contextual information (user instructions, metadata, interaction history), fine-tune the router. This enables efficient adaptation with minimal supervision—just 50 labeled examples per attribute in experiments.

**Key Theoretical Result**: MiCRo proves that for mixture preference distributions, the cross-entropy loss of any single BTL model has a lower bound proportional to the variance across subpopulations:

$\mathcal{L}_{CE}(r) \geq 2\rho K \cdot \mathbb{E}_x\left[\text{Var}_z[\{s_k^*\}_{k=1}^K]\right] + H(x, \pi, P(z|x))$

This lower bound grows with preference diversity—you literally cannot do better with a single model.

**Results**: On HelpSteer2 and RPR benchmarks, MiCRo's mixture heads specialize to different preference dimensions (helpfulness vs. verbosity vs. coherence) and consistently outperform single-reward baselines by 6-40% depending on the attribute.

**Limitations**:

- **Choosing K is hard**: How many mixture components do you need? Too few and you're back to averaging; too many and components become redundant or overfit. The paper shows performance is stable as K increases, but finding the right K for a new domain requires experimentation.
- **Heads may not align with interpretable concepts**: The learned mixture components specialize automatically, but there's no guarantee they'll correspond to human-interpretable preference dimensions. You might get one head for "helpfulness + verbosity" rather than separate heads.
- **Router requires context signal**: The context-aware routing only works if the prompt or metadata actually contains signal about which preference mode is relevant. For ambiguous queries, routing may be unreliable.
- **Training complexity**: Learning a mixture model with a router jointly is more complex than standard BTL, requiring careful balancing of the mixture likelihood and regularization terms.

---

### Solution 3: General Preference Representations (preference embeddings)

Move beyond scalar rewards entirely. Learn preference *representations* (embeddings) that can capture richer structures including intransitive and cyclic preferences.

BTL forces all preferences through a single-number bottleneck—every response gets one reward score, and we pick the higher one. But human preferences aren't always reducible to a single number. You might prefer response A over B because A is more helpful, B over C because B is more concise, but C over A because C is more creative. These cyclic preferences can't be represented by any scalar reward function.

General preference representations instead learn to embed responses in a high-dimensional space where preference relationships are encoded geometrically. Think of it like word embeddings (word2vec) but for preferences—similar preference patterns cluster together, and the geometry captures relationships that scalars can't. By removing the scalar reward bottleneck, these methods don't force multimodal preferences to collapse into a single number. The embedding space can naturally represent that "user type A prefers this direction" while "user type B prefers that direction" without averaging them.

The general approach is to learn embedding functions that map prompt-response pairs to a learned preference space where geometric relationships encode preference structure. This allows representing preferences that don't reduce to a single number.

**Limitations**:

- **Harder to use for RL**: Standard RL algorithms expect scalar rewards. If your preference model outputs embeddings, you need additional machinery to convert that to a training signal.
- **Less interpretable**: A scalar reward is easy to understand ("this response scores 0.8"). An embedding is much harder to reason about.
- **Intransitivity may be noise, not signal**: While human preferences can be genuinely intransitive, sometimes cyclic preferences in annotation data are just noise. Models that capture intransitivity may be fitting annotation errors.
- **Less mature**: This line of work is newer than BTL-based approaches, with fewer established training recipes and less empirical validation at scale.