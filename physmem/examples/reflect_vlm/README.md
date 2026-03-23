# Reflect-VLM Integration Example

This example shows how to integrate PhysMem with the Reflect-VLM assembly task
(MS-HAB puzzle assembly with VLM-guided robot manipulation).

## Overview

In this integration, PhysMem provides:
1. **Episodic memory** for storing action outcomes
2. **Hypothesis generation** from experience clusters
3. **Principle learning** that gets injected into VLM prompts
4. **Surprise-driven learning** to focus on unexpected outcomes

The VLM agent (from `thirdparty/reflect-vlm`) provides:
1. **Action generation** using vision-language models
2. **Symbolic state extraction** from the assembly environment
3. **Oracle actions** for learning from corrections

## Architecture

```
                     PhysMem System
                  ┌─────────────────────┐
                  │  Episodic Memory     │
                  │  ┌───────────────┐   │
Env Observation ──┤  │ Experience    │   │
                  │  │ Bank          │   │
                  │  └───────┬───────┘   │
                  │          │           │
                  │  ┌───────▼───────┐   │
                  │  │ Consolidation │   │
                  │  │ Engine        │   │
                  │  └───────┬───────┘   │
                  │          │           │
                  │  ┌───────▼───────┐   │
                  │  │ Hypotheses    │   │
                  │  └───────┬───────┘   │
                  │          │           │
                  │  ┌───────▼───────┐   │      ┌──────────────┐
                  │  │ Principles    │───┼──────▶│ VLM Prompt   │
                  │  └───────────────┘   │      │ (pre-hoc     │
                  └─────────────────────┘      │  guidance)    │
                                                └──────────────┘
```

## Usage

```python
# See run_with_physmem.py for the full integration example.
# The key integration points are:

# 1. Create PhysMem with an LLM for hypothesis generation
from physmem import PhysMem
from physmem.llm import create_llm

llm = create_llm("qwen", model="qwen-plus-latest")
mem = PhysMem(llm=llm)

# 2. Extract symbolic state from your environment
symbolic_state = extract_symbolic_state(env_info)

# 3. Get principles for VLM prompt
principles_text = mem.get_principles_prompt(action_type="insert")

# 4. Record experience after action
mem.record_experience(
    action=action,
    success=success,
    fail=fail,
    fail_tag=fail_tag,
    symbolic_state=symbolic_state,
    oracle_action=oracle_action,
    active_principles=mem.get_principles(),
)

# 5. End episode
mem.end_episode(success=episode_success)
```

## Requirements

- PhysMem core: `pip install numpy`
- LLM support: `pip install openai` (for Qwen/OpenAI)
- Reflect-VLM: See `thirdparty/reflect-vlm/` for environment setup
