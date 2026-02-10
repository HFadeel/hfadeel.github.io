Massive amounts of effort and money have gone into scaling (e.g. compute, data, parameters), and some architectural and efficiency tweaks (e.g., Mixture-of-Experts, attention variants, better optimizers, better parallelism). All of which are valuable. But they also raise an uncomfortable question:



> If our deep neural networks are already “universal approximators,” Why does it still feel like we’re brute-forcing representation—sometimes learning \*good performance\* without learning \*good structure? why do LLMs need all the web text to learn how to write basic sentences? Why it takes millions (if not billions) of images to generate a five-fingered hand image?\*

> 



This essay argues that the bottleneck is \*\*representational efficiency,\*\* more than anything else, more than raw expressivity. The headline “universal approximation” is correct but incomplete: it does not tell you how many parameters, how many regions, how much depth, or how much data is needed to represent the structure of interest \*in a way that training actually finds\*.



---



\# Universality is the wrong headline



A model class being a \*\*universal approximator\*\* means: for any (reasonable) target function $f^\*$ and error tolerance $\\epsilon$, there exists parameters $\\theta$ such that $f\_\\theta$ approximates $f^\*$ within $\\epsilon$.



But universality says nothing about:



\- \*\*How many parameters\*\* you need to reach $\\epsilon$.

\- \*\*How stable\*\* that approximation is off-distribution.

\- Whether the representation is \*\*modular / reusable\*\* vs “spaghetti.”

\- Whether \*\*SGD can find\*\* the efficient representation.



So the right question is not “can it represent it?”, but:



> \*\*What is the approximation rate and the learnability of that representation under realistic training?\*\*

> 



---



\# Why most deep nets naturally become “partition + affine templates”



Consider a standard MLP:



$h^{(0)} = x,\\quad

h^{(\\ell+1)} = \\sigma\\!\\left(W^{(\\ell)} h^{(\\ell)} + b^{(\\ell)}\\right),

\\quad

f(x) = W^{(L)}h^{(L)}+b^{(L)}$.



With ReLU, $\\sigma(z)=\\max(z,0)$, define a gate vector:



$g^{(\\ell)}(x) = \\mathbf{1}\\{W^{(\\ell)}h^{(\\ell)}(x)+b^{(\\ell)} > 0\\}$



Conditioned on a \*fixed\* gating pattern across layers, each ReLU is either identity (“on”) or zero (“off”). The entire network collapses to an affine function:



$f(x) = A\_{g}x + c\_{g}$.



So the input space is partitioned into regions (polytopes) where the gating pattern is constant, and on each region the network is affine. This is the intuitive core formalized in \*A Spline Theory of Deep Networks\* (\[Balestriero \& Baraniuk, 2018](https://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf)), which expresses such networks using \*\*max-affine spline operators (MASOs)\*\* and shows how these models implement a “template matching” view: choose an affine template based on region membership.



Note: this also applies to other similar activation functions e.g. Swish, SiLU, ELU.



---



\# What “piecewise” really means for representation



Piecewise-linear nets represent functions by partitioning input space into linear regions. Depth can blow up the number of regions (in principle), which is why they’re expressive and universal, but not necessarily efficient. For global algebraic structure: multiplicative relations, periodic functions, or high-degree polynomial interactions may require \*\*many regions\*\* to approximate well, even if the underlying structure is simple in a different coordinate system or basis. Even with so many regions the approximation might be brittle or weak.



---



\## What’s Limiting Our Deep Learning Representation



\### A) Element-wise isolation limits interaction order per layer



Most common activations (ReLU, SiLU/Swish, GELU) are coordinate-wise:



$\\sigma(z)\_i = \\phi(z\_i)$



This creates a distinctive computational grammar:



$\\text{linear mix} \\rightarrow \\text{element-wise nonlinearity} \\rightarrow \\text{linear mix} \\rightarrow \\cdots$



Cross-feature interactions are created indirectly through repeated mixing and nonlinearity. High-order interactions (e.g., products $x\_i$$x\_j$, compositional invariants, periodic structure) are possible, but the representation can be \*\*depth-hungry\*\* or \*\*width-hungry\*\*, and the learned solution can be sensitive to training details.



\### B) Basis alignment: the “wrong” basis makes simple structure expensive



Piecewise-affine bases are naturally aligned to piecewise-affine targets. But if the target is smooth with curvature, or periodic, or governed by multiplicative invariants, approximating it with affine patches can be representationally costly.



The core idea here is \*\*basis mismatch\*\*: the model can approximate the function, but it is forced to do so using a basis that does not compactly encode the structure.



\### C) Performance does not imply clean internal organization



\*Questioning Representational Optimism in Deep Learning\* argues that strong task performance can coexist with internally disorganized mechanisms—what the paper calls \*\*fractured entangled representations (FER)\*\*—contrasting SGD-trained networks with solutions found via open-ended evolutionary procedures on a simple image-generation task. (\[Baker et al., 2025](https://arxiv.org/pdf/2505.11581?utm\_source=chatgpt.com))



Even without adopting every claim, the conclusion is a useful corrective: internal representations can be redundant, scattered, and entangled while still yielding good outputs. That affects:



\- adaptation,

\- compositional generalization,

\- robustness under distribution shift,

\- interference during fine-tuning.



So “representation” is not only about capacity; it’s also about \*how solutions are organized in parameter space and discovered by training\*.



---



\# Ideas for different primitive for more efficient representation



Let me preface by saying: I don’t have the solution and the right solutions should take into account our hardware (i.e. GPU limitations - FLOPs, Memory bandwidth, and Special Functions). But let’s think together.



A pragmatic way to talk about representational power is:



> A representation is good when the target function has low \*description length\* in the model’s function class.

> 



“Description length” can be approximated by:



\- number of regions for piecewise-affine models,

\- polynomial degree / interaction order for algebraic models,

\- number of basis functions or experts used,

\- effective rank / sparsity of interactions.



From this angle, we have three design levers for investigation



\## Lever 1 — Interaction algebra: what operations are primitive?



\### \*\*High-order Interactions via matrix Exponentiation\*\*



IMExp makes high-order interactions inside one layer via $M^k$. (\[Singh et al., 2020](https://arxiv.org/pdf/2008.03936?utm\_source=chatgpt.com)). Here how it work:



1\. Compute a feature vector from the input:

&nbsp;   

&nbsp;   $\\phi(x) = Ux$.

&nbsp;   

2\. Assemble a square matrix as an affine function of features:

&nbsp;   

&nbsp;   $M(x) = B + \\sum\_{a=1}^{d} \\phi\_a(x)\\, T\_a,$

&nbsp;   

&nbsp;   where each $T\_a \\in \\mathbb{R}^{n\\times n}$ is a learned “template” matrix, and B is a learned bias matrix.

&nbsp;   

3\. Apply the matrix exponential:

&nbsp;   

&nbsp;   $\\exp(M) = \\sum\_{k=0}^{\\infty}\\frac{1}{k!}M^k$.

&nbsp;   

4\. Read out a scalar/vector prediction:

&nbsp;   

&nbsp;   $p(x)=V+\\langle S,\\exp(M(x))\\rangle$.

&nbsp;   



This allow features to interact together.  The series expansion contains \*\*all powers\*\*:



$\\exp(M)=I + M + \\frac{1}{2}M^2 + \\frac{1}{6}M^3 + \\cdots$



Matrix multiplication couples dimensions, so $M^2$, $M^3$, $\\ldots$ generate a combinatorial family of cross-feature interactions. In effect, the M-layer can create polynomial-like interaction structure inside one block, rather than requiring many layers to build those interactions compositionally.



The paper shows how choosing structured templates can encode specific monomials and can represent periodic functions via exponentials of rotation-like generators, yielding extrapolation behavior in certain synthetic tasks.



Matrix exponentiation is scientifically more expensive that element-wise activations and introduces numerical stability issues. But as a representation lens, it cleanly demonstrates a design principle:



> Some structures become cheap when the primitive matches the target algebra (multiplication/periodicity), and expensive when the primitive is misaligned.

> 



\### Cheap explicit multiplicative interactions



Instead of forcing multiplication to emerge indirectly, we can add a small budget of explicit bilinear terms:



$\\text{Block}(x)=W\_2\\,\\sigma(W\_1 x)\\;+\\;\\sum\_{k=1}^{K} v\_k \\,\\big(a\_k^\\top x\\big)\\big(b\_k^\\top x\\big)$.



\- K controls interaction capacity.

\- This is a low-rank quadratic augment: it captures key crosses without a full $O(d^2)$ quadratic layer.

\- It targets the “interaction order per layer” constraint directly.



\## Lever 2 — Adaptive basis: allocate different local bases to different regions



Instead of one global basis everywhere (one specific activation function), allow the model to choose/mix bases depending on input. A general formulation:



$h(x) = \\sum\_{m=1}^{N} \\alpha\_m(x)\\,\\phi\_m(Wx),

\\quad

\\alpha(x)=\\mathrm{softmax}\\!\\left(\\frac{Gx}{\\tau}\\right)$.



Where $\\phi\_m$ could include:



\- piecewise or smooth piecewise (ReLU/SiLU/GELU).

\- periodic or semi-periodic (sin⁡e, Fourier features, unbounded sine snake $x + 1/a \\sin^2(ax)$).

\- polynomial/spline bases.

\- multiplicative gates.



This can be seen as a continuum between:



\- mixture-of-activations.

\- mixture-of-subspaces.

\- mixture-of-experts.



The goal is to increase representational efficiency without paying full compute everywhere. But again there are optimization challenges and considerations, such as:



\- The loss landscape of a piecewise and smooth piecewise (e.g. Relu, SiLu) is generally convex-ish (or at least easy to navigate). The loss landscape of a Sine network is filled with infinite local minima. It is essentially a fractal surface making optimization challenging.

\- Sines are bounded between -1 and 1. This can lead to vanishing gradients in very deep networks compared to ReLU

\- Selection collapse



\## Lever 3 — Organization: encourage factoring and reuse during training



If internal solutions tend toward redundancy and entanglement, then “better representation” may require explicitly optimizing for it. The spline/MASO perspective suggests regularizers over templates and partitions, and the FER hypothesis suggests that organization is not automatic under SGD. (\[Balestriero \& Baraniuk, 2018](https://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf?utm\_source=chatgpt.com); \[Baker et al., 2025](https://arxiv.org/pdf/2505.11581?utm\_source=chatgpt.com))



---



\## Closing: the open problem



Scaling will keep producing wins. But “capability per FLOP” increasingly depends on whether the representational primitives match the structure of the world.



The core design questions are simple to ask and difficult to solve:



\- What interaction algebra should be cheap?

\- What bases should be available, and how should they be selected?

\- What training objectives encourage reuse and factorization instead of redundant entanglement?

\- Which improvements provide real gains under modern constraints (stability, hardware efficiency, scaling behavior)?



If the next decade repeats the last—more scale plus incremental tweaks—progress will continue. But a step-change in \*\*representational efficiency\*\* likely requires upgrading the primitives themselves: not just more pieces, but better building blocks.

