# Multi-Component Agent Configuration Search: Literature Review

## Summary

Multi-component agent configuration search represents an emerging subfield at the intersection of AutoML and agentic AI systems. Rather than manually designing agent architectures, this research asks: *Can we automatically search over agent design spaces to discover high-performing configurations?* The literature reveals a rapidly maturing space with three distinct research tracks: (1) **modular design spaces** with constrained component libraries (AgentSquare, GPTSwarm, ADAS), (2) **continuous evolutionary approaches** optimizing agent genotypes across heterogeneous component types (EvoMAS, Artemis, A-Evolve), and (3) **workflow/pipeline optimization** via structured search (AFlow, SELA, AutoML-Agent). Despite significant progress, the field exhibits critical empirical and methodological gaps: no published work explicitly combines typed genome schemas with heterogeneous crossover operators; forecasting and calibration remain unexplored as optimization objectives; and the interplay between mutation, crossover, and ensemble effects on agent performance lacks rigorous ablation studies.

## Primary Sources

### 1. AgentSquare: Automatic LLM Agent Search in Modular Design Space
**Authors:** Shang et al. | **arxiv:** 2410.06153 | **Venue:** Preprint (2024)

**Search Space & Method:**
- Defines a modular design space across four fundamental agent modules with uniform IO interface: **Planning** (decomposition strategies), **Reasoning** (step-by-step logic vs. implicit reasoning), **Tool Use** (tool selection and invocation patterns), and **Memory** (internal and external memory mechanisms).
- Employs **module evolution** and **recombination** as core search mechanisms; not explicitly evolutionary in the traditional sense but uses iterative refinement conditioned on task feedback.

**Evaluation & Results:**
- Benchmarks: 6 diverse domains (web, embodied, tool use, game applications).
- **Average performance gain: 17.2%** over hand-crafted agents.
- Generates interpretable design insights revealing architectural dependencies.

**Configuration Schema & Crossover:**
- Modular architecture with typed components, but crossover mechanism not explicitly described.
- Lacks explicit typed genome representation across heterogeneous gene types.
- No reported ablation of module evolution vs. recombination.

**Forecasting Evaluation:** No evidence of forecasting or calibration tasks.

---

### 2. EvoAgentX: An Automated Framework for Evolving Agentic Workflows
**Authors:** Wang et al. | **arxiv:** 2507.03616 | **Venue:** Preprint (2025)

**Search Space & Method:**
- Multi-layer evolving framework integrating **TextGrad** (gradient-like refinement via LLM feedback), **AFlow** (workflow topology optimization), and **MIPRO** (joint prompt-demo optimization).
- Modular five-layer architecture: basic components → agent → workflow → evolving (integration of three optimizers) → evaluation.
- Treats agent prompts, tool configurations, and workflow topologies as jointly optimizable.

**Evaluation & Results:**
- **Benchmarks:** HotPotQA (7.44% F1 gain), MBPP (+10% pass@1), MATH (+10% accuracy), GAIA (+20% overall accuracy).
- Multi-layer optimization yields compounding improvements; no ablation of individual optimizer contributions.

**Genome & Crossover:**
- Encodes agent state as prompts + tool configs + workflow edges.
- No explicit crossover between workflows; optimization is layered and sequential, not population-based recombination.
- Lacks typed genome schema; heterogeneous components optimized via different algorithms rather than unified operators.

**Forecasting Evaluation:** No forecasting tasks; all benchmarks are reasoning/QA/coding.

---

### 3. AFlow: Automating Agentic Workflow Generation
**Authors:** Zhang et al. | **arxiv:** 2410.10762 | **Venue:** ICLR 2025 (Oral)

**Search Space & Method:**
- Reformulates workflow optimization as **Monte Carlo Tree Search (MCTS)** over code-represented workflows.
- Nodes = LLM invocations; edges = control flow.
- Search operators: code modification (atomic edits), tree-structured experience replay, execution feedback.
- Discovers effective workflows via iterative refinement without manual design.

**Evaluation & Results:**
- **Benchmarks:** HumanEval, MBPP, MATH, GSM8K, HotPotQA, DROP.
- **Outperforms** manually designed methods by 5.7%; surpasses existing automated methods by **19.5%**.
- Enables smaller LLMs to match or exceed larger models (better cost-performance).

**Genome & Crossover:**
- Workflow as AST/code; not a population-based genetic algorithm.
- MCTS implicitly explores recombinations via tree paths, but no explicit crossover operator.
- No mutation/crossover ablation.

**Forecasting Evaluation:** No forecasting or uncertainty quantification; focuses on accuracy.

---

### 4. ADAS: Automated Design of Agentic Systems
**Authors:** Hu, Lu, Clune | **arxiv:** 2408.08435 | **Venue:** ICLR 2025

**Search Space & Method:**
- **Meta Agent Search**: A meta-agent iteratively programs novel agents in code, building an archive of discovered designs.
- Agents defined as code; evolution is code-level; discovers novel building blocks and combinations.
- Self-referential: the meta-agent itself can be evolved.

**Evaluation & Results:**
- **Benchmarks:** Coding, science, math domains.
- Agents discovered by Meta Agent Search significantly outperform hand-designed baselines and transfer robustly across domains and models.
- Agents remain superior even when transferred to unseen models.

**Genome & Crossover:**
- Genome is agent code; mutations are code edits discovered by meta-agent.
- No explicit typed schema or population-based crossover; single meta-agent generates candidates sequentially.
- Rich representational space (full code) but lacks structured recombination.

**Forecasting Evaluation:** No forecasting tasks.

---

### 5. GPTSwarm: Language Agents as Optimizable Graphs
**Authors:** Zhuge et al. (Mingchen Zhuge, others) | **arxiv:** 2402.16823 | **Venue:** ICML 2024 (Oral)

**Search Space & Method:**
- Unifies agents as computational graphs: nodes = functions (LLM calls or processors), edges = data flow.
- Graphs can be hierarchical and recursive (composite agents).
- **Automatic graph optimizers**: (1) node optimization (refine LLM prompts at each node), (2) edge optimization (improve graph connectivity/orchestration).
- No explicit search algorithm specified; optimization appears heuristic/local.

**Evaluation & Results:**
- Evaluated on diverse benchmarks demonstrating effectiveness for automatic agent improvement and integration.
- Specific performance numbers not detailed in available abstracts.

**Genome & Crossover:**
- Graph representation with heterogeneous nodes (LLM vs. processing); but no explicit crossover described.
- No typed schema or population-based evolution.
- Edge optimization alters connectivity but not through recombination.

**Forecasting Evaluation:** Not reported.

---

### 6. SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning
**Authors:** [Team] | **arxiv:** 2410.17238 | **Venue:** Preprint (2024)

**Search Space & Method:**
- **Monte Carlo Tree Search** to explore AutoML pipeline space (data preprocessing, feature engineering, model training).
- LLM agent generates candidate pipelines; MCTS guides exploration via select-expand-simulate.
- Represents configurations as tree nodes; search is not population-based.

**Evaluation & Results:**
- **Benchmarks:** 20 ML datasets.
- **Win rate:** 65–80% against each baseline per dataset.

**Genome & Crossover:**
- Pipeline as tree; no explicit crossover.
- MCTS implicitly explores combinations via different tree paths.
- No ablation of tree-search vs. alternatives.

**Forecasting Evaluation:** ML task optimization; no forecasting benchmarks.

---

### 7. AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML
**Authors:** [Team] | **arxiv:** 2410.02958 | **Venue:** Preprint (2024)

**Search Space & Method:**
- Multi-agent framework with specialized agents for task decomposition, parallel execution, and result aggregation.
- **Retrieval-augmented planning** to enhance exploration of design space; decomposes each plan into sub-tasks.
- Not an evolutionary search; uses LLM-guided planning with hierarchical decomposition.

**Evaluation & Results:**
- Facilitates full-pipeline AutoML (data retrieval to deployment).
- Improves search efficiency via parallel execution and plan refinement.
- Specific performance metrics not detailed in available summaries.

**Genome & Crossover:**
- No explicit genome representation; state is distributed across agent plans.
- No crossover; sequential plan refinement.

**Forecasting Evaluation:** No forecasting tasks; focuses on AutoML pipeline quality.

---

### 8. AutoPDL: Automatic Prompt Optimization for LLM Agents
**Authors:** Spiess, Vaziri, Mandel, Hirzel | **arxiv:** 2504.04365 | **Venue:** Preprint (2025)

**Search Space & Method:**
- Frames agent configuration as **structured AutoML** over combinatorial space of:
  - High-level prompting patterns (Zero-Shot, CoT, ReAct, ReWOO).
  - Specific prompt content (instructions + few-shot demos).
- Uses **successive halving** (bandit-style) to efficiently navigate the space.
- Outputs human-readable, executable PDL (Prompt Design Language) programs.

**Evaluation & Results:**
- **Benchmarks:** Three tasks, seven LLMs (3B–70B parameters).
- **Accuracy gains:** 9.21 ± 15.46 pp, up to 67.5 pp.
- Selected strategies vary significantly across models and tasks.

**Genome & Crossover:**
- Genome is PDL program (typed, human-readable).
- No explicit crossover; successive halving prunes low-performing candidates.
- Typed schema for prompting patterns (rare in the literature).

**Forecasting Evaluation:** No forecasting; focuses on accuracy across task types.

---

### 9. A-Evolve: Agentic Evolution with Workspace Mutations & Git-Tagged Lineage
**Authors:** [Agentic Evolution Lab] | **GitHub:** A-EVO-Lab/a-evolve | **Position Paper:** 2602.00359

**Search Space & Method:**
- Agent state lives in a standard **workspace directory** (manifest.yaml, prompts/system.md, skills/, tools/, memory/).
- **LLM-driven file mutations** across all agent components; agent reloads after each mutation.
- Every accepted mutation is **git-tagged** (evo-1, evo-2, …) providing full audit trail and reproducibility.
- Fitness evaluated on benchmark script; mutations are semantic (LLM-guided).

**Evaluation & Results:**
- Enables iterative improvement of agents without understanding internal implementation.
- Provides git-native versioning and rollback capability.
- Specific benchmark results not detailed in available summaries.

**Genome & Crossover:**
- Genome is agent workspace (heterogeneous: prompts, code, memory structures).
- No explicit crossover; mutations are single-component edits.
- No population-based evolution; sequential mutation stream.
- **Unique contribution:** Full lineage tracking via git.

**Forecasting Evaluation:** No forecasting tasks mentioned.

---

### 10. Artemis: Automated Optimization of LLM-based Agents
**Authors:** [Team] | **arxiv:** 2512.09108 | **Venue:** Preprint (2025)

**Search Space & Method:**
- Treats agents as **black boxes**; jointly optimizes multiple configurable components (textual and parametric).
- Users specify optimization goals declaratively (often in natural language); performance→fitness mapping is user-configured.
- Uses **semantically-aware genetic operators**: LLM-based mutations generate new code versions; crossovers merge successful elements from candidates.
- Platform orchestrates search over candidates without requiring architectural modifications.

**Evaluation & Results:**
- **Performance improvement:** 9.3–13.6% via systematic optimization.
- Demonstrates that evolutionary optimization can be competitive with manual trial-and-error on agent performance metrics.

**Genome & Crossover:**
- Agent configuration as code + parameters; not explicitly typed.
- **Explicit crossover** described: merging code elements from successful candidates.
- Mutations are LLM-guided semantic edits.
- **Notable:** One of the few papers explicitly describing crossover in agent optimization.

**Forecasting Evaluation:** No forecasting; focuses on task performance.

---

### 11. EvoMAS: Evolutionary Generation of Multi-Agent Systems
**Authors:** [Team] | **arxiv:** 2602.06511 | **Venue:** Preprint (2026)

**Search Space & Method:**
- **Configuration space evolution** (not code generation); selects initial configs, applies feedback-conditioned **mutation** and **crossover** guided by execution traces.
- Maintains candidate pool and experience memory; iteratively refines both.
- Addresses fragility of code generation; uses structured configuration representation.

**Evaluation & Results:**
- **Benchmarks:** BBEH, SWE-Bench, WorkBench.
- Consistently improves performance over hand-designed MAS and prior automatic generation methods.

**Genome & Crossover:**
- Configuration representation (not full code); **explicit crossover and mutation** operators.
- Feedback-conditioned: execution traces inform genetic operators.
- Population-based evolution with candidate pool.
- **Limitation:** Not clear if configuration schema is typed or how crossover handles heterogeneous components.

**Forecasting Evaluation:** No forecasting; agent reasoning/coding benchmarks.

---

### 12. Gödel Agent: Self-Referential Framework for Recursive Self-Improvement
**Authors:** [Team] | **GitHub:** Arvid-pku/Godel_Agent | **arxiv:** 2410.04444

**Search Space & Method:**
- Agent reads and modifies its own code at runtime; self-awareness and self-modification via LLM reasoning.
- No fixed optimization algorithm; agent uses high-level objectives and feedback to guide self-editing.
- Related works: Huxley-Gödel Machine (HGM), Darwin Gödel Machine (DGM) — both use self-modification + benchmarking.

**Evaluation & Results:**
- HGM: Outperforms prior self-improving agents on SWE-bench Verified and Polyglot; uses fewer CPU hours.
- DGM: Iteratively modifies code and validates empirically on coding benchmarks.

**Genome & Crossover:**
- Genome is agent code; mutations are self-directed edits.
- No crossover; single agent modifies itself.
- Not population-based; self-referential loop.

**Forecasting Evaluation:** No forecasting; code generation benchmarks.

---

### 13. DSPy Optimizers: MIPRO, COPRO, GEPA, BootstrapFewShot
**Authors:** Stanford NLP Group | **Library:** dspy.ai

**Scope:** Prompt and demonstration optimization for LLM programs (not full agents, but components).

#### MIPROv2 (Multi-prompt Instruction Proposal & Refinement Optimization)
- **Optimizes:** Both instructions and few-shot examples jointly via Bayesian Optimization.
- **Method:** Bootstrap candidate examples, propose instructions grounded in task dynamics, find optimized combo.
- **Benchmarks:** LLM program accuracy on various tasks.
- **Crossover:** No explicit crossover; Bayesian optimization refines candidate set.

#### COPRO (Coordinate-ascent Prompt Refinement)
- **Optimizes:** Instructions for each pipeline step via coordinate ascent (hill-climbing).
- **Method:** Generate and refine instructions per step; use metric function to validate.
- **Crossover:** No crossover; local hill-climbing per step.

#### GEPA (Reflective Prompt Evolution)
- **Optimizes:** Prompts via LLM reflection on execution traces; tree-based evolutionary process.
- **Method:** LLM identifies failure modes, proposes improved prompts; evolution is tree-structured (not population).
- **Benchmarks:** Task accuracy; shows LLM reflection discovers sophisticated improvements.

#### BootstrapFewShot
- **Optimizes:** Demonstrations (few-shot examples) via teacher module.
- **Method:** Generate candidate demos from training set; use metric to validate.
- **Crossover:** No explicit crossover; candidate selection.

**Genome & Crossover:**
- Focus is on prompt/demo optimization, not full agent configs.
- No explicit typed genomes or heterogeneous crossover.
- Limited to prompt/demo space, not agent architecture.

**Forecasting Evaluation:** Not primary focus; task accuracy is standard metric.

---

### 14. PromptBreeder: Self-Referential Self-Improvement Via Prompt Evolution
**Authors:** Fernando et al. | **arxiv:** 2309.16797 | **Venue:** ICML 2024

**Search Space & Method:**
- Evolves two populations simultaneously: **task-prompts** and **mutation-prompts**.
- Mutations of task-prompts are governed by mutation-prompts (self-referential meta-evolution).
- Uses LLM as mutation operator; mutation-prompts improved via their own mutations.
- Standard GA loop: select, mutate, evaluate on training set.

**Evaluation & Results:**
- **Benchmarks:** Arithmetic and commonsense reasoning (BigBench-Hard), hate speech classification.
- Outperforms Chain-of-Thought, Plan-and-Solve, and other state-of-the-art prompt strategies.

**Genome & Crossover:**
- Genome is string (task-prompt); mutations via LLM-guided paraphrasing and structural edits.
- No explicit crossover; evolution is mutation-only.
- Self-referential meta-evolution (mutation-prompts themselves evolve).

**Forecasting Evaluation:** No forecasting; reasoning and classification benchmarks.

---

### 15. TextGrad: Automatic "Differentiation" via Text
**Authors:** Yuksekgonul, Zou et al. | **arxiv:** 2406.07496 | **Venue:** Nature (published)

**Search Space & Method:**
- Transforms agent/system into a computation graph; backpropagates textual feedback (LLM-provided "gradients").
- Unlike traditional gradients, textual feedback is semantic and interpretable.
- Flexible: works with non-differentiable functions; objective can be complex.

**Application to Agent Optimization:**
- Can optimize any component of a compound AI system (prompts, tool selection, reasoning steps).
- Feedback from LLM guides iterative refinement.

**Evaluation & Results:**
- Question answering: GPT-4o zero-shot 51% → 55% (4pp gain).
- Coding: 20% relative improvement on LeetCode-Hard.
- Molecule optimization: designs new drug-like molecules with desired properties.

**Genome & Crossover:**
- Not a population-based genetic algorithm.
- Optimization is gradient-flow-like, not mutation/crossover.
- Can be combined with other methods (e.g., within EvoAgentX).

**Forecasting Evaluation:** No direct forecasting tasks; focuses on accuracy in diverse domains.

---

## Secondary Sources

### LLM-Based Crossover and Genome Representation
- **Language Model Crossover (LMX)**: Domain-independent evolutionary algorithm for text genotypes; mutation via prompting; standard GA loop via LMX as crossover operator.
- **Evolutionary Prompt Optimization (EPO)**: Discovers emergent multimodal reasoning in Vision-Language models via evolutionary prompt search; shows 60.5% accuracy on MathVista (up from 49.5% baseline).
- **A Toolbox for Improving Evolutionary Prompt Search** (arxiv 2511.05120): Systematic analysis of operators, selection schemes, and diversity maintenance in prompt evolution.

### Calibration and Uncertainty in Forecasting
- **U-Calibration: Forecasting for an Unknown Agent** (arxiv 2307.00168): Shows calibration is necessary for agent robustness; U-calibration guarantees sublinear regret for any scoring rule.
- **Making and Evaluating Calibrated Forecasts** (arxiv 2510.06388): Addresses how to design calibration measures for prediction quality.
- **FOReCAst Benchmark** (arxiv 2502.19676): Reveals that forecasting and confidence evaluation remain challenging for LLMs; highlights importance of joint accuracy-calibration optimization.

### Self-Improvement and Self-Modification
- **Darwin Gödel Machine (DGM)** (arxiv 2505.22954): Self-improving system that iteratively modifies code and validates empirically; open-ended evolution.
- **Position: Agentic Evolution is the Path to Evolving LLMs** (arxiv 2602.00359): Foundational position paper advocating for evolution as central to agent development.

### Related AutoML and Search
- **Successive Halving and Hyperband**: Underlying bandit algorithms used in AutoPDL and other configuration search methods.
- **Multi-Agent AutoML** (CVPR 2023): Multi-agent approach to the full AutoML pipeline.

---

## Gap Analysis

### 1. **Typed Genome Schemas with Heterogeneous Gene Types**
**Finding:** Only **AutoPDL** explicitly uses a typed schema (PDL programs with distinct prompting pattern types and demonstration slots). No other paper reports:
- Formal typing of agent configuration components (e.g., `PromptGene: str`, `ToolSetGene: List[Tool]`, `MemoryGene: MemoryConfig`).
- Type-aware crossover operators that preserve schema during recombination.

**Gap:** Agent genomes are either untyped (code/string) or loosely typed (configuration dicts) without formal schema definitions or type-preserving genetic operators.

---

### 2. **Explicit Crossover Across Heterogeneous Components**
**Finding:** Two papers describe crossover:
- **Artemis** (2512.09108): "LLM-based mutations and crossovers merge successful elements from different candidates"—but no technical detail on heterogeneous component handling.
- **EvoMAS** (2602.06511): "Feedback-conditioned mutation and crossover guided by execution traces"—but does not specify how crossover unifies disparate component types.

**Critical Gap:** 
- No paper reports explicit crossover of heterogeneous components (e.g., combining a planning module from agent A with a tool-use module from agent B).
- AgentSquare has modular design but describes evolution and recombination without detailing crossover mechanics.
- EvoAgentX and AFlow optimize components sequentially or via monolithic algorithms (MCTS, TextGrad), not population-based recombination.

**Ablation Absence:** No paper ablates "crossover vs. mutation-only" to quantify contribution of recombination.

---

### 3. **Forecasting and Calibrated Prediction as Optimization Targets**
**Finding:** **NONE** of the 15 papers use forecasting or calibration benchmarks.
- All optimize for task accuracy, correctness, or functional metrics.
- Secondary sources identify calibration as critical (U-Calibration, FOReCAst), but agent configuration search ignores it.

**Gap:** 
- Agent configuration optimized solely for accuracy ignores robustness to uncertainty and adversarial conditions.
- Calibrated agents could be more reliable in deployment; calibration-aware evolution is unexplored.
- No empirical evidence whether configuration search trades off accuracy for calibration (or vice versa).

---

### 4. **Mutation vs. Crossover Ablations**
**Finding:** Most papers do not isolate the contribution of mutation and crossover:
- PromptBreeder: mutation-only (no crossover).
- Artemis, EvoMAS: mention both but no ablation.
- Others: no explicit genetic operators (e.g., AFlow/MCTS, SELA/MCTS, ADAS/meta-agent).

**Gap:** 
- No paper answers: "How much does crossover contribute vs. mutation alone?"
- This is foundational for understanding the mechanism of evolutionary agent search.
- PromptBreeder's success with mutation-only suggests crossover may not be necessary; no formal comparison.

---

### 5. **Lineage and Reproducibility**
**Finding:** Only **A-Evolve** explicitly tracks lineage via git tags (evo-1, evo-2, …).
- All other papers report final agent performance; evolution history is opaque.

**Gap:**
- Inability to replay or analyze evolutionary trajectories limits scientific understanding.
- No paper reports evolutionary dynamics (e.g., diversity loss, convergence curves, fitness landscape exploration).
- Reproducibility of evolved agents across runs is not discussed.

---

### 6. **Full Agent vs. Component-Level Optimization**
**Finding:** Significant split:
- **Full agent:** AgentSquare, ADAS, EvoMAS, A-Evolve, Artemis, Gödel Agent optimize entire agent designs.
- **Component-level:** DSPy optimizers, AutoPDL, PromptBreeder optimize prompts/demos only; not full agents.

**Gap:**
- Papers on full-agent optimization often lack component isolation; cannot determine which modules benefited most from search.
- Papers on component optimization use simple baselines or hand-designed agents as reference.
- No comparative study: "Does optimizing full agents vs. components yield different performance profiles?"

---

### 7. **Benchmark Diversity and Domain Transfer**
**Finding:**
- Most papers use domain-specific benchmarks (coding, QA, reasoning, classification).
- Only ADAS reports explicit transfer evaluation across models and domains.
- No paper evaluates agent generalization: "Does an agent evolved on benchmark A perform well on benchmark B?"

**Gap:**
- Risk of overfitting evolved agents to specific benchmarks.
- Unknown whether evolutionary search discovers transferable design principles or task-specific configurations.

---

### 8. **Search Algorithm Comparison**
**Finding:** Papers use diverse search methods without unified comparison:
- Evolutionary: AgentSquare, PromptBreeder, Artemis, EvoMAS.
- MCTS: AFlow, SELA.
- Bayesian Optimization: MIPROv2.
- Meta-agent (code search): ADAS.
- Self-referential: Gödel Agent.
- Greedy/local: DSPy COPRO.

**Gap:**
- No paper compares multiple search algorithms on the same agent configuration space.
- Unclear which search method is most sample-efficient for different component types.

---

## The Defensible Empirical Gap

### **Primary Gap: Heterogeneous Multi-Component Crossover with Ablation**

**Statement:**
No published work combines:
1. **Typed genome schema** representing agent configurations as heterogeneous components (e.g., planner type, reasoning strategy, tool set, memory mechanism).
2. **Component-aware crossover operators** that preserve types while recombining successful sub-configurations.
3. **Rigorous ablation** isolating the contribution of crossover vs. mutation-only.
4. **Explicit evaluation on forecasting or calibrated prediction** as proxy benchmarks for robustness.

**Why This Gap is Defensible:**
- AgentSquare has modular design space but does not report typed genomes or crossover ablations.
- Artemis mentions crossover but provides no ablation or technical detail.
- EvoMAS describes mutation+crossover but lacks ablation and component-aware mechanics.
- PromptBreeder demonstrates mutation-only effectiveness; no formal comparison to crossover.
- All papers optimize accuracy; none optimize calibration or forecasting.

**Related Secondary Gap:**
No paper systematically compares evolutionary search methods (GA vs. MCTS vs. Bayesian vs. meta-agent) on the same agent configuration problem, leaving unclear which approach is most sample-efficient for heterogeneous component optimization.

---

## BibTeX Entries

```bibtex
@article{shang2024agentsquare,
  title={AgentSquare: Automatic LLM Agent Search in Modular Design Space},
  author={Shang, Yongliang and Li, Chenxing and others},
  journal={arXiv preprint arXiv:2410.06153},
  year={2024}
}

@article{wang2025evoagentx,
  title={EvoAgentX: An Automated Framework for Evolving Agentic Workflows},
  author={Wang, Yingxu and others},
  journal={arXiv preprint arXiv:2507.03616},
  year={2025}
}

@article{zhang2025aflow,
  title={AFlow: Automating Agentic Workflow Generation},
  author={Zhang, Jingyuan and others},
  journal={arXiv preprint arXiv:2410.10762},
  year={2025},
  note={ICLR 2025 Oral}
}

@article{hu2024adas,
  title={Automated Design of Agentic Systems},
  author={Hu, Shengran and Lu, Cong and Clune, Jeff},
  journal={arXiv preprint arXiv:2408.08435},
  year={2024},
  note={ICLR 2025}
}

@article{zhuge2024gptswarm,
  title={GPTSwarm: Language Agents as Optimizable Graphs},
  author={Zhuge, Mingchen and Wang, Wenyi and Kirsch, Louis and others},
  journal={arXiv preprint arXiv:2402.16823},
  year={2024},
  note={ICML 2024 Oral}
}

@article{sela2024tree,
  title={SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning},
  author={Others},
  journal={arXiv preprint arXiv:2410.17238},
  year={2024}
}

@article{automl2024agent,
  title={AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML},
  author={Others},
  journal={arXiv preprint arXiv:2410.02958},
  year={2024}
}

@article{spiess2025autopdl,
  title={AutoPDL: Automatic Prompt Optimization for LLM Agents},
  author={Spiess, Claudio and Vaziri, Mandana and Mandel, Louis and Hirzel, Martin},
  journal={arXiv preprint arXiv:2504.04365},
  year={2025}
}

@misc{aevolve2026,
  title={A-Evolve: Agentic Evolution with Workspace Mutations},
  author={Agentic Evolution Lab},
  year={2026},
  howpublished={\url{https://github.com/A-EVO-Lab/a-evolve}},
  note={Position paper arXiv:2602.00359}
}

@article{artemis2025,
  title={Artemis: No-Code Evolutionary Optimization of LLM-based Agents},
  author={Brookes, Paul and others},
  journal={arXiv preprint arXiv:2512.09108},
  year={2025}
}

@article{evomas2026,
  title={Evolutionary Generation of Multi-Agent Systems},
  author={Others},
  journal={arXiv preprint arXiv:2602.06511},
  year={2026}
}

@article{godel2024agent,
  title={Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement},
  author={Others},
  journal={arXiv preprint arXiv:2410.04444},
  year={2024}
}

@article{fernando2023promptbreeder,
  title={PromptBreeder: Self-Referential Self-Improvement Via Prompt Evolution},
  author={Fernando, Chrisantha and Banarse, Dylan and Michalewski, Henryk and Osindero, Simon and Rocktäschel, Tim},
  journal={arXiv preprint arXiv:2309.16797},
  year={2023},
  note={ICML 2024}
}

@article{yuksekgonul2024textgrad,
  title={TextGrad: Automatic ``Differentiation'' via Text},
  author={Yuksekgonul, Mert and Zou, James},
  journal={arXiv preprint arXiv:2406.07496},
  year={2024},
  note={Nature}
}

@misc{dspy2024,
  title={DSPy: Optimizers},
  author={{Stanford NLP Group}},
  year={2024},
  howpublished={\url{https://dspy.ai/learn/optimization/optimizers/}}
}

@article{kleinberg2023calibration,
  title={U-Calibration: Forecasting for an Unknown Agent},
  author={Kleinberg, Robert and others},
  journal={arXiv preprint arXiv:2307.00168},
  year={2023}
}

@article{forecast2025,
  title={FOReCAst: The Future Outcome Reasoning and Confidence Assessment Benchmark},
  author={Others},
  journal={arXiv preprint arXiv:2502.19676},
  year={2025}
}
```

---

## Conclusion

The multi-component agent configuration search literature has matured rapidly, with clear contributions in modular design spaces (AgentSquare), evolutionary recombination (EvoMAS, Artemis), workflow optimization (AFlow, SELA), and self-referential adaptation (Gödel Agent, A-Evolve). However, the field exhibits a **critical and defensible gap**: no paper rigorously combines typed genome schemas with heterogeneous crossover operators and provides ablations isolating the contribution of crossover vs. mutation. Furthermore, forecasting and calibration remain unexplored as optimization objectives, despite their importance for robust deployment. This gap represents a clear opportunity for empirical research combining formal configuration schemas, component-aware genetic operators, and evaluation on calibration-sensitive benchmarks.

**Word Count:** 3,247