"""
PhysMem Quick Start Example.

Demonstrates the full feedback loop:
    observe -> retrieve principles -> decide -> act -> record -> learn

The "agent" here is a tiny policy that consults learned principles
before selecting an action. At the start it explores randomly; as
PhysMem consolidates experiences into principles, those principles
actively filter and bias the policy toward better actions. You should
see the rolling success rate climb over episodes.

This is intentionally LLM-free: it exercises the rule-based hypothesis
path so you can run the demo with nothing beyond ``numpy``.
"""

import random
from collections import deque
from typing import List, Optional, Sequence, Union

import numpy as np

from physmem import PhysMem, Principle, ScientificLearningConfig
from physmem.core.hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Simulated environment
# ---------------------------------------------------------------------------

def simulate_grasp(object_size: str, approach: str) -> dict:
    """Simulate a grasp attempt. Returns success/fail with reason.

    Ground-truth rules the agent should eventually learn:
        * push always fails
        * side_grasp works well on flat objects, poorly on tall ones
        * top_grasp works well on tall objects, poorly on flat ones
    """
    if approach == "push":
        return {"success": False, "fail": True, "fail_tag": "wrong_action"}
    if object_size == "flat" and approach == "side_grasp":
        ok = random.random() < 0.9
        return {"success": ok, "fail": not ok, "fail_tag": None if ok else "slip"}
    if object_size == "flat" and approach == "top_grasp":
        ok = random.random() < 0.3
        return {"success": ok, "fail": not ok, "fail_tag": None if ok else "slip"}
    if object_size == "tall" and approach == "top_grasp":
        ok = random.random() < 0.85
        return {"success": ok, "fail": not ok, "fail_tag": None if ok else "unstable"}
    if object_size == "tall" and approach == "side_grasp":
        ok = random.random() < 0.4
        return {"success": ok, "fail": not ok, "fail_tag": None if ok else "unstable"}
    return {"success": random.random() < 0.5, "fail": False, "fail_tag": None}


# ---------------------------------------------------------------------------
# Principle-aware policy
# ---------------------------------------------------------------------------

KnowledgeItem = Union[Principle, Hypothesis]


def _item_matches_object(item: KnowledgeItem, object_size: str) -> bool:
    """Does this principle/hypothesis apply to the current object?

    Rules:
        * If the item has no ``trigger_conditions``, it is globally
          applicable -- return True.
        * If the item has trigger_conditions but none of them mention
          ``object_size``, the rule is not state-restricted by object;
          return True (it still applies, just not gated on object).
        * Otherwise, return True iff one of the conditions matches the
          current object explicitly. Falls back to a content substring
          search for items whose triggers were never extracted.
    """
    conds = getattr(item, "trigger_conditions", None) or []
    if not conds:
        return True

    has_object_constraint = any(
        cond.strip().startswith("object_size=") for cond in conds
    )
    if not has_object_constraint:
        return True

    key = f"object_size={object_size}"
    if any(key in cond for cond in conds):
        return True

    content = getattr(item, "content", None) or getattr(item, "statement", "") or ""
    return object_size in content.lower()


def _item_weight(item: KnowledgeItem) -> float:
    """Rank signal: principles outrank hypotheses, high-importance first."""
    if isinstance(item, Principle):
        return 10.0 + float(item.importance_score)
    return float(getattr(item, "confidence", 0.0))


def select_action(
    object_size: str,
    knowledge: Sequence[KnowledgeItem],
    approaches: List[str],
    explore_prob: float = 0.15,
) -> str:
    """Pick an action using principles + active hypotheses.

    The three-tier memory in the paper injects both long-term principles
    and working-memory hypotheses into the planner's context (Sec. III-D).
    This policy mirrors that pattern, with a small twist: because rule-
    based hypotheses can be noisy and contradictory, we aggregate them as
    a signed majority vote per action rather than trusting any single
    item outright.

    For each action:
        score(a) = sum_over_items(sign * weight * trigger_match)
    where ``sign`` is +1 for PREFER and -1 for AVOID, ``weight`` favours
    principles over hypotheses and high-importance over low-importance,
    and ``trigger_match`` is 1 iff the item applies to the current state.
    The agent then ε-greedy picks the highest-scoring action.
    """
    scores = {a: 0.0 for a in approaches}

    for item in knowledge:
        ptype = str(getattr(item, "principle_type", None)
                    or getattr(item, "hypothesis_type", "")).lower()
        if ptype not in ("avoid", "prefer"):
            continue
        action_types = getattr(item, "action_types", None) or []
        if not action_types:
            continue
        if not _item_matches_object(item, object_size):
            # Skip state-qualified items that don't match; keep globally-
            # applicable items that carry no trigger conditions.
            if getattr(item, "trigger_conditions", None):
                continue
        weight = _item_weight(item)
        sign = 1.0 if ptype == "prefer" else -1.0
        for a in action_types:
            if a in scores:
                scores[a] += sign * weight

    # ε-greedy exploration.
    if random.random() < explore_prob:
        return random.choice(approaches)

    best_score = max(scores.values())
    best_actions = [a for a, s in scores.items() if s == best_score]
    return random.choice(best_actions)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(0)
    np.random.seed(0)

    n_episodes = 200

    config = ScientificLearningConfig(
        memory_name="grasp_learning",
        max_memory_size=1000,
        # Require enough samples per cluster so rule-based hypotheses
        # are not driven by 2-sample noise.
        consolidation_interval=20,
        min_cluster_size=6,
        min_experiences_for_consolidation=12,
        min_experiences_for_hypothesis=6,
        max_principles_in_prompt=20,
        run_consolidation_async=False,   # sync for deterministic demo
        save_path="./physmem_quickstart_output",
    )
    mem = PhysMem(config=config)

    objects = ["flat", "tall"]
    approaches = ["side_grasp", "top_grasp", "push"]

    print(f"Running {n_episodes} episodes of grasp simulation...\n")
    print(
        f"{'episode':>8} {'window_succ%':>13} {'exps':>5} "
        f"{'hyps':>5} {'prins':>6}"
    )

    window: deque = deque(maxlen=20)

    for episode in range(n_episodes):
        obj = random.choice(objects)

        # 1. Retrieve what we've learned so far (both tiers).
        principles = mem.get_principles()
        hypotheses = [
            h for h in mem.get_hypotheses()
            if str(getattr(h, "status", "")).lower() in ("proposed", "verified")
        ]

        # 2. Let the policy consult all active knowledge before deciding.
        approach = select_action(obj, list(principles) + list(hypotheses), approaches)

        # 3. Execute in the environment.
        result = simulate_grasp(obj, approach)

        # 4. Record the experience so consolidation can keep learning.
        #    Note: action is the *approach* only; the object is part of
        #    the symbolic state so cluster triggers and policy filters
        #    share a common vocabulary.
        mem.record_experience(
            action=approach,
            success=result["success"],
            fail=result.get("fail", False),
            fail_tag=result.get("fail_tag"),
            symbolic_state={"object_size": obj, "approach": approach},
            # No state_vec: let consolidation build a deterministic
            # embedding from the symbolic state so clusters separate
            # cleanly by (object_size, approach).
            active_principles=principles if principles else None,
        )

        stats = mem.end_episode(success=result["success"])
        window.append(1 if result["success"] else 0)

        if (episode + 1) % 10 == 0:
            rolling = 100.0 * sum(window) / len(window)
            print(
                f"{episode + 1:>8d} {rolling:>12.1f}% "
                f"{stats['total_experiences']:>5d} "
                f"{stats['total_hypotheses']:>5d} "
                f"{stats['total_principles']:>6d}"
            )

    # -------------------------------------------------------------------
    # Inspect what the system learned.
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LEARNED HYPOTHESES:")
    print("=" * 60)
    for h in mem.get_hypotheses():
        print(f"  [{h.status}] {h.statement}")
        print(
            f"    confidence={h.confidence:.2f}, type={h.hypothesis_type}, "
            f"action_types={h.action_types}"
        )

    print("\n" + "=" * 60)
    print("LEARNED PRINCIPLES:")
    print("=" * 60)
    for p in mem.get_principles(top_k=20):
        print(f"  [{p.principle_type}] {p.content}")
        print(
            f"    importance={p.importance_score:.1f}, "
            f"confidence={p.confidence:.2f}, "
            f"action_types={p.action_types}, "
            f"triggers={p.trigger_conditions}"
        )

    print("\n" + "=" * 60)
    print("STATISTICS:")
    print("=" * 60)
    stats = mem.get_stats()
    print(f"  Episodes: {stats['episode_count']}")
    print(f"  Experiences: {stats['memory']['total']}")
    print(f"  Hypotheses: {stats['hypotheses']['total']}")
    print(f"  Principles: {stats['principles']['total']}")

    mem.save_state()
    print(f"\nState saved to {config.save_path}")
    mem.shutdown()


if __name__ == "__main__":
    main()
