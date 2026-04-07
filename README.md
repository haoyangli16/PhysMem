# PhysMem: Physical Memory System for Experience-to-Principle Learning

PhysMem is a test-time training memory system that learns transferable principles from raw experience. It implements a scientific learning loop:

```
Raw Experience -> [Consolidation] -> Hypotheses -> [Verification] -> Principles
```

### TL;DR

- **Typed learning objectives** — Principles are categorized (AVOID, PREFER, SEQUENCE, COMPARE) to guide both generation and verification.
- **Resonance gating** — Expected experiences reinforce existing principles silently; only surprising outcomes (prediction errors) trigger new learning.
- **Ebbinghaus-inspired decay** — Unused principles fade over time; frequently validated ones persist.
- **Memory folding** — Raw experiences are compressed once covered by established principles, keeping memory bounded.
- **Three-stage scientific loop** — Experience clustering → hypothesis generation → verification & promotion to principles.

## Three-Layer Memory Architecture

| Layer | Storage | Lifecycle |
|-------|---------|-----------|
| **Episodic Memory** | Raw experiences (one per action) | Collected during execution, garbage-collected over time |
| **Working Memory** | Hypotheses (proposed conjectures) | Generated from clusters, tested through verification |
| **Long-term Memory** | Principles (verified rules) | Promoted from hypotheses, embedded into LLM prompts |

## Installation

```bash
git clone https://github.com/haoyangli16/PhysMem.git
cd PhysMem

# Core only (numpy)
pip install .

# With a specific LLM provider
pip install ".[openai]"          # OpenAI / Kimi
pip install ".[gemini]"          # Google Gemini
pip install ".[qwen]"            # Alibaba Qwen (uses OpenAI-compatible API)

# With all LLM providers
pip install ".[llm]"

# With FAISS similarity search (recommended)
pip install ".[faiss]"

# With clustering + semantic embedding
pip install ".[clustering]"

# Everything
pip install ".[all]"

# Editable install for development
pip install -e ".[all,dev]"
```

## Quick Start

PhysMem only pays off if learned knowledge actually flows **back into
the agent**. The core pattern is a closed loop:

```
retrieve principles -> inject into prompt -> act -> observe -> record -> consolidate
```

Every step: ask PhysMem what it has learned, inject that into the
planner's prompt, act, and then record the outcome. Skipping the
injection step turns PhysMem into an offline recorder and defeats the
purpose of the three-tier memory.

```python
from physmem import PhysMem
from physmem.llm import create_llm

# 1. Create the memory system with an LLM backend.
llm = create_llm("openai", model="gpt-4o")
mem = PhysMem(llm=llm)

for episode in range(100):
    for step in range(max_steps):
        # 2. Retrieve learned knowledge and format it for the planner.
        principles_prompt = mem.get_principles_prompt()

        # 3. Inject the principles into the agent's prompt.  This is
        #    the step that closes the feedback loop: memory actually
        #    influences the next decision.
        action = agent.act(
            observation=observation,
            principles=principles_prompt,   # <- key: memory infects the agent
        )
        result = env.step(action)

        # 4. Record the experience so consolidation can keep learning.
        mem.record_experience(
            action=action,
            success=result.success,
            fail=result.fail,
            fail_tag=result.fail_reason,
            symbolic_state={
                "action_type": action.type,
                "holding": agent.is_holding,
                "progress": env.progress,
            },
            state_vec=observation.embedding,  # optional
            active_principles=mem.get_principles(),  # tracks which
                                                     # principles were in
                                                     # scope at decision time
        )

    # 5. End-of-episode housekeeping: consolidation, verification,
    #    promotion, and auto-save all happen here.
    mem.end_episode(success=env.is_success)

# 6. Inspect what was learned.
for p in mem.get_principles():
    print(f"[{p.principle_type}] {p.content} (confidence: {p.confidence:.2f})")

# 7. Reuse the prompt formatter standalone if you want to see what
#    the agent sees.
prompt_text = mem.get_principles_prompt(action_type="grasp")
# -> "1. [HIGH] Prefer grasping from the side when object is flat..."
```

### How memory affects decisions

If your agent's `act()` method does not consume `principles_prompt`, no
learning takes effect. The end-to-end reference integration (including
how to inject principles into a VLM planner) lives in
[`examples/reflect_vlm/run_with_physmem.py`](examples/reflect_vlm/run_with_physmem.py).
For a fully self-contained demonstration that you can run in one
command, see [`examples/quickstart.py`](examples/quickstart.py); it
implements a tiny principle-aware policy and prints a visible
learning curve.

## Without LLM (Rule-Based)

PhysMem works without an LLM too — the rule-based path extracts the
dominant action and the dominant symbolic-state features from each
cluster and emits actionable `AVOID` / `PREFER` hypotheses. Your
policy can then filter or bias on `principle.action_types` and
`principle.trigger_conditions` directly:

```python
from physmem import PhysMem

mem = PhysMem()  # No LLM needed

# Run a few episodes so consolidation has something to work with.
for obj in ["flat", "flat", "flat", "tall", "tall", "tall"]:
    mem.record_experience(
        action="push",
        success=False,
        fail=True,
        fail_tag="wrong_action",
        symbolic_state={"object_size": obj},
    )
mem.end_episode(success=False)

# Read back the learned hypotheses and use them in your policy.
for h in mem.get_hypotheses():
    print(h.statement, h.action_types, h.trigger_conditions)
# -> "Avoid 'push' when object_size=flat (3/3)" ['push'] ['object_size=flat']

# Example: simple principle-aware policy (see examples/quickstart.py
# for a self-contained version).
def select_action(obj, knowledge, approaches):
    forbidden = {
        a for item in knowledge
        if str(getattr(item, "hypothesis_type",
                       getattr(item, "principle_type", ""))).lower() == "avoid"
        and f"object_size={obj}" in (item.trigger_conditions or [""])
        for a in (item.action_types or [])
    }
    return next(a for a in approaches if a not in forbidden)
```

## Custom LLM Backend

Implement `BaseLLM` to use any LLM:

```python
from physmem.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def generate(self, prompt, system_prompt=None, **kwargs):
        return my_api.call(prompt, system=system_prompt)

mem = PhysMem(llm=MyLLM())
```

## Configuration

```python
from physmem import PhysMem, ScientificLearningConfig

config = ScientificLearningConfig(
    memory_name="my_robot",
    max_memory_size=5000,
    consolidation_interval=30,      # Consolidate every 30 episodes
    max_hypotheses_per_cluster=3,
    max_principles_in_prompt=5,
    promotion_confidence=0.8,       # Confidence threshold for promotion
    save_path="./checkpoints",      # Auto-save directory
    auto_save_interval=50,          # Save every 50 episodes
)

mem = PhysMem(config=config, llm=llm)
```

## Persistence

```python
# Save
mem.save_state("./my_checkpoint")

# Load
mem = PhysMem.load_state("./my_checkpoint", llm=llm)
```

## Architecture

```
physmem/
├── __init__.py          # Public API: PhysMem, Experience, Principle, ...
├── core/                # Data structures
│   ├── experience.py    # Experience, MemoryBank
│   ├── hypothesis.py    # Hypothesis, HypothesisStore, ExperienceCluster
│   ├── principle.py     # Principle, PrincipleStore, PrincipleType
│   ├── index.py         # FAISS/numpy vector index
│   ├── retriever.py     # k-NN retrieval with symbolic filtering
│   └── writeback.py     # Write-back policy
├── learning/            # Learning loop
│   ├── consolidation.py # Clustering + hypothesis generation
│   ├── verification.py  # Hypothesis testing + promotion
│   └── loop.py          # ScientificLearningLoop (main orchestrator)
├── llm/                 # LLM integration
│   ├── base.py          # Abstract BaseLLM interface
│   └── providers.py     # OpenAI, Gemini, Qwen, Kimi implementations
└── examples/            # Usage examples
    ├── quickstart.py    # Minimal example
    └── reflect_vlm/     # Integration with reflect-vlm assembly task
```

## Key Concepts

### Surprise-Driven Learning
Not all experiences trigger learning. The system checks if an experience matches existing principles (resonance). Only surprising experiences (prediction errors) are sent to consolidation, reducing computational cost.

### Experience Folding
Once principles are established, the raw experiences that support them can be "folded" (compressed). This keeps memory bounded while preserving learned knowledge.

### Hypothesis Verification
Hypotheses aren't blindly promoted. The system uses a verification planner that designs experiments, tracks results, and only promotes hypotheses that pass confidence thresholds.
