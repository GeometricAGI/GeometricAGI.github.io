---
layout: post
title: "Growing Languages for Self-Improving Agents"
description: "How agents can grow programming languages, reusable abstractions, and formal models to improve their search for solutions, illustrated by Tessella."
date: 2026-10-23
author:
  - name: "Geometric Team"
    linkedin: "https://www.linkedin.com/company/112782074/"
---

# Growing Languages for Self-Improving Agents

By the [Geometric Team](https://www.linkedin.com/company/112782074/)

<blockquote style="max-width: 595px; margin: 28px 0 32px; padding: 0; border: 0;">
  <p style="margin: 0; font-family: 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif; font-size: 17px; font-style: italic; line-height: 1.7;">“The limits of my language mean the limits of my world.”</p>
  <p style="margin: 10px 0 0; font-size: 13.6px; line-height: 1.7;">Ludwig Wittgenstein, <a href="https://www.wittgensteinproject.org/w/index.php/Tractatus_Logico-Philosophicus_%28English%29"><em>Tractatus Logico-Philosophicus</em>, 5.6</a></p>
</blockquote>

Reasoning about a complex system involves choosing what to represent and how to represent it. These choices influence the questions we can formulate and the explanations or solutions we can investigate. For an agent solving a sequence of problems, these abstractions may influence both the cost of subsequent searches and the complexity of the solutions it can discover.

**We propose an approach to agent self-improvement in which agents develop and refine their languages as they construct, evaluate, and improve solutions.** The research question is whether developing reusable representations and models of a problem helps agents reason about alternatives and improve performance on subsequent tasks, after accounting for the cost of that development.

We use **Tessella**, a GPU programming language grown through automated agent iterations, to illustrate this approach. Its development combines typed kernel operations, formal models, and hardware experiments.

## Learning representations for search

[DreamCoder (Ellis et al., PLDI 2021)](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf) studies this relationship through a wake–sleep cycle. Its wake phase searches for programs in a typed functional language. Abstraction sleep extracts reusable components from solutions, while dream sleep trains a search policy using solved tasks and sampled programs. Since learned components can refer to earlier components, successive iterations can produce a hierarchy of abstractions.

![Solving tasks supplies programs for learning abstractions, while the growing vocabulary supplies examples for learning to search.](/assets/growing-languages/dreamcoder-cycle.svg)

*Figure 1. DreamCoder's wake–sleep cycle, after Ellis et al. Learned components change the vocabulary available to subsequent program searches.*

Advances in LLMs invite us to revisit DreamCoder's foundational idea with agents that can build and revise the programming systems they use. [LILO](https://proceedings.iclr.cc/paper_files/paper/2024/file/819cebb05f993840e8a52d7564c5c282-Paper-Conference.pdf) combines LLM-guided synthesis with refactoring and documentation to grow a reusable program library. At a larger scale, Anthropic reports [agents building a C compiler](https://www.anthropic.com/engineering/building-c-compiler) capable of compiling the Linux kernel, although its generated code remains inefficient. These results motivate asking whether agents can revise a language's operations, analyses, and lowerings alongside the programs expressed through it.

A solved task can reveal a construction worth reusing. Turning it into an abstraction changes the vocabulary available to later searches. [ReGAL](https://proceedings.mlr.press/v235/stengel-eskin24a.html) studies this transition by refactoring existing programs into reusable functions and testing their effect on later program prediction. [TheoryCoder-2](https://arxiv.org/abs/2602.00929) learns abstractions from experience and incorporates them into hierarchical planning. These results motivate a broader question: can agents extend reusable abstractions into the language and models they use to construct and assess solutions? The value should be measured by which solutions become discoverable and the cumulative effort needed to find them.

## Composition and IR

Reusable constructions become useful for search when agents can combine them under clear rules. A vocabulary with explicit composition rules and meanings gives agents and compilers a structure they can inspect and transform. This is the sense in which we use language here. Treating the vocabulary as a language also lets us apply the theory and methods of compilers: intermediate representations, semantics-preserving rewrites, progressive lowering, and verified compilation give agents established ways to transform programs and to justify each step.

While Hazy Research argues that agents may let us [retire CUDA DSLs](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions), we see a new role for abstractions: helping agents search, reuse results, and tackle more complex tasks. Their argument concerns layers built around human implementation work and preserves the need for correctness oracles.

Following [Harper's definitions](https://www.cs.cmu.edu/~rwh/pfpl/), we distinguish three aspects of a language: its **syntax** defines the available expressions, its **static semantics** determines which of them are admissible, and its **behavioral semantics** gives their meaning. In Tessella, a single declaration per operation provides all three. Its syntax consists of operations on typed scalars and statically shaped tensors, written as traced Python and recorded in an intermediate representation; the PTX and SASS instruction sets are modeled as languages of their own. Its static semantics is enforced by types and verifiers generated from those declarations: operand types must match without implicit conversion, and each instruction form must be available on the selected target. Its behavioral semantics is given in Lean: each operation has a formal meaning, with preconditions and algebraic laws where they apply.

An **intermediate representation (IR)** is the structured program form that a compiler's analyses and transformations inspect and rewrite. It is where a language's vocabulary becomes concrete: each operation is an IR operation with typed operands and results, its composition rules are checked by the IR's verifiers, and its meaning is what the IR's rewrites must preserve. [MLIR](https://mlir.llvm.org/docs/LangRef/) illustrates an extensible IR: new operations and types can be added, with [interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/) exposing properties compiler passes need. Growing the vocabulary therefore means extending the IR: when an agent introduces an operation, it must define those properties so compiler passes can use it correctly.

A small Tessella kernel shows what the IR records. Thread `i` handles element `i + 64` of a row-major matrix with eight columns, and needs that element's row and column. The values are 32-bit unsigned integers, so addition wraps modulo 2³²:

```python
j = i + 64
row = j // 8
col = j % 8
```

In the IR, these expressions form a **computation graph**: each operation is a node, and each value flows along an edge to the operations that use it. For this pure arithmetic subgraph, the edges record which results depend on which, and nothing more: any evaluation order that respects them gives the same values, and the graph does not say where values are kept. A whole kernel needs more. Its IR must also represent control flow and the order of observable effects such as memory accesses; Tessella's IR has structured conditionals and loops, and its validation preserves the order of memory events. Returning to the example, a different subgraph computes the same row and column:

```python
row = (i + 64) >> 3
col = i & 7
```

Both graphs agree for every 32-bit pattern of `i`, wraparound included: at `i = 2³² − 1`, both compute `j = 63` and therefore row 7 and column 7. That `row` and `col` locate a valid matrix element is a separate claim, which also requires `i + 64` not to wrap and to stay below the number of elements. The second graph replaces division, which the GPU implements as a multi-instruction sequence, with a shift and a mask, and its column no longer depends on the addition.

A **rewrite** turns one graph into another that computes the same result. Tessella's optimization passes include this rewrite. Some rewrites hold unconditionally: `x // 8` equals `x >> 3` for every unsigned `x`. Others hold only under a side condition: `(x + c) % 8` equals `x & 7` only when 8 divides `c`. Each rewrite pattern names the Lean lemmas that justify it, and Lean can check, for each compiled kernel, that the optimized graph behaves like the traced one.

A **lowering** translates the graph into a more concrete language, committing to choices the higher level leaves open. Tessella lowers each operation to PTX, NVIDIA's virtual instruction set: the shift becomes `shr.u32`, the mask `and.b32`, and the addition `add.u32`. An operation can have several registered lowerings, selected by target and priority. A multiply-add, for example, lowers either to PTX's single `mad.lo` instruction or to a multiplication followed by an addition. Each lowering is tested on the GPU against its operation's Lean meaning, and the PTX of each compiled kernel is proved to behave like the optimized graph. These proofs cover a single thread executing in isolation.

![Two computation graphs for an element's row and column, related by a proved rewrite, and their lowering to PTX instructions.](/assets/growing-languages/rewrites-and-lowering.svg)

*Figure 2. Two graphs for the same result. A rewrite replaces division and remainder by a shift and a mask, under the lemmas that justify it. A lowering then maps each operation to PTX, where one operation may have several implementations.*

## Optimization as graph search

The two graphs in Figure 2 are two points in a larger space of programs, and two kinds of graph are involved. Within a candidate, the computation graph connects operations through the values they exchange. Across candidates, each vertex is a complete program, and each edge is a transformation between programs: a rewrite, or a choice among lowerings. Replacing division with a shift and a mask is one such edge; choosing `mad.lo` over a separate multiplication and addition is another. Other operations add further edges: a masked load can be lowered as a predicated load or as a guarded load followed by a select. Edges also exist below PTX: NVIDIA's `ptxas` assembler schedules machine instructions on its own, whereas Tessella's direct route to machine code brings that scheduling into the search. Optimization searches this second graph for a program that satisfies the contract at lower cost.

This framing makes optimization a form of planning. A planning agent uses a model of its environment to predict the effects of actions before taking them: [MuZero](https://www.nature.com/articles/s41586-020-03051-4) learns such a model to plan in games, and [world models](https://worldmodels.github.io/) let an agent learn behavior inside its own simulation of an environment. Here the language plays the part of the world model. Its operations and types define the states an agent can represent, its rewrites and lowerings define the available actions, and its semantics predicts how each action affects the program's meaning before any kernel runs. What the semantics does not predict, such as speed, memory traffic, and register pressure, must come from cost models or hardware experiments. The language also sets the granularity of planning. A rewrite backed by a lemma, such as the fact that division by a power of two is a shift, is a single action; without it, an agent must rediscover the equivalence through many lower-level edits and tests. A richer vocabulary turns longer chains of reasoning into single steps.

The language therefore shapes this graph in two distinct ways. Its operations and composition rules fix which computations are representable at all, and justified transformations specify admissible transitions between them. Within a finite budget, what also matters is which computations are discoverable. Adding a function definable from existing primitives enables no new computation, yet it can shorten programs and search paths enough to bring solutions within reach; DreamCoder's learned library grows in exactly this way. Much of the language growth we consider is of this second kind.

The graph becomes a model of the optimization problem when its vertices include representation and resource choices, its edges state admissibility conditions, and its objective measures the cost of executable candidates. A good path may pass through costlier intermediate programs.

Lean serves this search in two ways. A proved lemma guarantees a whole family of edges, so a search that follows only proved edges preserves meaning however far it goes; each of Tessella's rewrite patterns names the lemmas that justify it. For floating-point code, many useful edges are not exact: reordering a reduction or approximating an exponential changes results. There, the proved property is a numerical contract (an input domain, an error tolerance, and the treatment of infinities and NaNs) rather than equality. An unresolved proof obligation, in turn, tells the agent what remains to be established for a candidate. It does not show that the candidate is wrong: a true statement can stay unproved for lack of a lemma, of sufficient automation, or of time. A counterexample does show a violation, and becomes precise feedback, much as solvers for [syntax-guided synthesis](https://www.cis.upenn.edu/~alur/SyGuS13.pdf) feed counterexamples from failed checks back into their search over a grammar of candidates. Proofs do not measure cost, so correctness and cost are settled separately. [Equality saturation](https://blog.sigplan.org/2021/04/06/equality-saturation-with-egg/) makes the same split: it applies only valid rewrites to build a space of equivalent programs, then extracts the cheapest under a cost model. In our case, experiments supply the cost.

## Models and proof certificates

An IR makes candidate transformations explicit, but does not establish that they preserve behavior. Formalization addresses this gap. A defined vocabulary of typed operations and composition rules gives the model concrete objects to describe: their meanings, valid compositions, and the conditions under which transformations preserve the intended computation. This extends the program-library learning line: abstractions can guide later search, while a formal model also lets agents state proof obligations and produce certificates that people and a trusted checker can examine.

Consider the rewrites of Figure 2. The remainder rewrite is tempting to generalize: `(x + c) % 8` becomes `x & 7` whenever the addition looks irrelevant. That is correct only when 8 divides `c`. A proof distinguishes the two cases:

```lean
import Std.Tactic.BVDecide

-- The rewrite applied in Figure 2: valid for every 32-bit x, wraparound included.
theorem rem_aligned (x : BitVec 32) : (x + 64) % 8 = x &&& 7 := by
  bv_decide

-- The generalization is wrong: with c = 1 and x = 0, the two sides differ.
theorem rem_unaligned : ((0 : BitVec 32) + 1) % 8 ≠ 0 &&& 7 := by
  decide
```

These are exact statements about 32-bit machine arithmetic, with no numerical tolerance. Tessella's own lemma covers every width, every power-of-two modulus, and every `c` that modulus divides, and the rewrite pattern may fire only when that side condition holds. [Lean's kernel](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/) checks such a proof certificate independently of the generating model, giving agents and human reviewers evidence for the stated claim. Integer arithmetic is the easy case. Floating-point claims require explicit models for rounding, approximate exponentiation, and accumulation order, which Tessella provides at the PTX level and has yet to bring to its kernel language.

This adds a correctness check beyond benchmark scores. Agents can exploit gaps in an evaluator, producing reward-hacking shortcuts that score well without meeting the intended requirement. A proof rules out shortcuts that violate the property it establishes under the model's assumptions, but cannot guarantee requirements left out of the specification. The guarantee applies to the modeled construction. Connecting it to executable code also requires evidence that compilation and execution preserve the modeled behavior, following the principle of [proof-carrying code](https://www.cs.cmu.edu/~fox/pcc.html).

Tessella's existing PTX validation addresses another obligation: whether emitted instructions preserve the kernel representation's modeled behavior within isolated-thread execution. It does not establish that the kernel computes what its author intended, namely that `row` and `col` locate the right element. Algorithm correctness, compiler correctness, concurrent execution, and agreement with hardware require evidence appropriate to each claim.

For agentic systems, separating candidate generation from independent checking of properties offers several advantages:

- **Independent verification.** A proof certificate can be checked without relying on the generating model's confidence or explanation.
- **Feedback for revision.** Type errors and counterexamples identify specific violations, and unresolved proof obligations identify what the agent still needs to establish.
- **Coverage beyond test cases.** A proof can cover every input satisfying its assumptions, including numerical error bounds when rounding and floating-point behavior are modeled explicitly.
- **Stable correctness requirements.** Checking candidates against a fixed specification can reject improvements in benchmark score that violate that specification, addressing one source of [reward hacking](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/).

## Choosing abstraction levels

A computation can be represented at several levels of abstraction. Take attention, the central operation of transformer models: a softmax turns each query's scores against all keys into probabilities, which then weight an average of the values. As a tensor graph, this is three operations. A tile program processes the keys block by block, as in [FlashAttention](https://arxiv.org/abs/2205.14135), carrying a running maximum, sum, and output (m, l, u) so that normalization still covers every key. Distributed tiles assign each score to a lane. Finally, the work is expressed as instructions: first in PTX, NVIDIA's virtual instruction set, then in SASS, the machine code it is assembled into.

![Progressive lowering of attention from tensor expressions through a tiled loop and distributed fragments to target instructions, with different optimization decisions at each level.](/assets/growing-languages/abstraction-levels.svg)

*Figure 3. Progressive lowering of the same attention computation. Each representation exposes a different set of design choices. Lowering adds implementation commitments that must satisfy the higher-level contract.*

Each representation exposes a distinct set of optimization variables. At the tensor level, these include fusion and the order of operations. Tiling determines block sizes and data reuse, while thread distribution determines ownership, communication, and synchronization. Instruction scheduling introduces trade-offs between transfer overlap and register use. A decomposition constrains which transformations remain available at lower levels. An agent-directed language should preserve these dependencies so search can evaluate implementations across levels.

A lowering also requires a justification that the implementation satisfies the higher-level contract, as exemplified by [verified compilation](https://xavierleroy.org/publi/compcert-CACM.pdf). At the higher levels, these justifications carry the most information. Lowering the tensor graph to a tile program must show that the running state (m, l, u) still yields the softmax over all keys. Over the reals this invariant is exact; a floating-point implementation also needs an error analysis that bounds how far rounding moves the result from it; distributing a tile must show that the lanes sharing a row combine their partial results into that row's statistics. Proved once for an abstraction, together with its numerical contract, such a justification applies to every kernel that uses the abstraction and meets its preconditions, which is what makes a higher-level vocabulary worth growing.

## Language growth and recursive improvement

Extending the vocabulary can affect both candidate construction and traversal of the search graph. A learned component may reduce the decisions needed to express a candidate. A reusable transformation sequence may shorten the search path between designs. Both are potential outcomes of learning from previous investigations.

This gives a concrete counterpart to DreamCoder's wake–sleep cycle. **Optimization mode** builds and evaluates attention implementations using the current language. **Research mode** investigates their results and revises the means of construction. An agent might introduce a type for mergeable row statistics, prove its update invariant, and add a lowering that preserves its structure until the lane distribution is chosen. The next optimization phase can use that abstraction without reconstructing the complete argument.

![An inner search explores attention implementations, while the research phase retains abstractions, invariants, and search guidance for subsequent optimization.](/assets/growing-languages/optimization-and-research.svg)

*Figure 4. Optimization explores attention implementations that meet a fixed numerical contract; the reordered reduction and the approximate exponential are not exactly equivalent to the originals. Research develops representations and justified transformations, and evaluation measures their gain on later tasks before they return to the search.*

The same principle applies to a producer–consumer pipeline. A buffer's identity is insufficient when it is reused: a consumer must wait for the correct transfer, and a producer must wait for the previous use to finish. A reusable cursor can expose that state while leaving buffer count and scheduling to the agent. Its invariants constrain protocol changes. Research may also revise the investigation procedure itself, such as experiment selection or search allocation. Whether those revisions improve later investigations is the recursive-improvement question.

## An agentic loop for language design

Language design itself can be part of the agent's research loop. During optimization, agents use the current language to construct and evaluate candidate solutions. During research, they analyze these results and revise operations, types, models, or search methods. Proof obligations constrain correctness, experiments measure behavior and cost, and subsequent tasks test whether revisions improve search. For Tessella, candidate changes include tile operations, pipeline abstractions, and their models. Each revision changes which solutions agents can express and which paths they can explore in the optimization graph.

Evaluation must distinguish three claims. A proof establishes a property within a model, an experiment measures implementation behavior and cost, and subsequent tasks assess the utility of retained abstractions and lemmas. In Tessella, an instruction experiment compares observed outputs with its formal meaning, while a kernel benchmark compares end-to-end behavior and cost. Tests should exercise partial tiles in normalization, supported matrix layouts, and repeated buffer reuse. A faster kernel alone does not justify a new abstraction: the relevant comparison is whether that abstraction helps agents find better solutions on later tasks, including its development cost.

![An agentic loop in which agents use the current language to optimize, inspect results, revise the language, and evaluate changes on later tasks.](/assets/growing-languages/language-design-loop.svg)

*Figure 5. An agentic loop for language design. Agents revise the language in light of optimization results, assess changes with proof obligations and experiments, and test their value on later tasks.*

Experiments should isolate vocabulary learning, search-policy adaptation, and proof reuse, using direct generation and ordinary code reuse as baselines. Solution quality should be reported against cumulative time, tokens, and experimental cost, including language development, failed attempts, proof construction, checking, and maintenance. Repeated runs and uncertainty estimates are needed to distinguish persistent gains from variation between searches.

## Conclusion

We have argued that the language an agent works in shapes what it can find. Treating optimization as search over programs makes the language a model of the problem: its abstractions define what an agent can express and its proofs which steps are safe, while experiments decide which results are better. Growing that language then becomes part of the agent's own research.

Tessella offers one setting for studying language growth. In a future post, we will present experimental findings from running this agentic system, quantifying its token cost and studying the abstractions that different agents generate: which constructs they introduce, how those constructs differ across agents, and which ones prove their value on later tasks.

We would also like to explore how agents might develop languages and formal models for other complex systems, such as compiler pipelines, distributed services, and scientific discovery. The question is how those evolving representations change what agents can discover and build.
