"""
PhysMem Quick Start Example.

Demonstrates the core API without any environment dependencies.
Simulates a simple pick-and-place task where the agent learns
principles about when grasping succeeds or fails.
"""

import random
import numpy as np

from physmem import PhysMem, ScientificLearningConfig


def simulate_grasp(object_size: str, approach: str) -> dict:
    """Simulate a grasp attempt. Returns success/fail with reason."""
    # Simple rules the agent should learn:
    # 1. Side grasp works better for flat objects
    # 2. Top grasp works better for tall objects
    # 3. Pushing always fails
    if approach == "push":
        return {"success": False, "fail": True, "fail_tag": "wrong_action"}
    elif object_size == "flat" and approach == "side_grasp":
        return {"success": random.random() < 0.9, "fail": False, "fail_tag": None}
    elif object_size == "flat" and approach == "top_grasp":
        return {"success": random.random() < 0.3, "fail": True, "fail_tag": "slip"}
    elif object_size == "tall" and approach == "top_grasp":
        return {"success": random.random() < 0.85, "fail": False, "fail_tag": None}
    elif object_size == "tall" and approach == "side_grasp":
        return {"success": random.random() < 0.4, "fail": True, "fail_tag": "unstable"}
    else:
        return {"success": random.random() < 0.5, "fail": False, "fail_tag": None}


def main():
    # Create PhysMem without LLM (uses rule-based hypothesis generation)
    config = ScientificLearningConfig(
        memory_name="grasp_learning",
        max_memory_size=500,
        consolidation_interval=5,       # Consolidate frequently for demo
        min_cluster_size=2,
        min_experiences_for_consolidation=2,
        run_consolidation_async=False,  # Sync for demo
        save_path="./physmem_quickstart_output",
    )
    mem = PhysMem(config=config)

    objects = ["flat", "tall"]
    approaches = ["side_grasp", "top_grasp", "push"]

    print("Running 100 episodes of grasp simulation...\n")

    for episode in range(100):
        obj = random.choice(objects)
        approach = random.choice(approaches)

        # Get current principles (if any)
        principles = mem.get_principles()

        # Simulate the grasp
        result = simulate_grasp(obj, approach)

        # Record experience
        mem.record_experience(
            action=f"{approach}_{obj}",
            success=result["success"],
            fail=result.get("fail", False),
            fail_tag=result.get("fail_tag"),
            symbolic_state={
                "object_size": obj,
                "approach": approach,
            },
            state_vec=np.random.randn(32).astype(np.float32),  # Dummy embedding
            active_principles=principles if principles else None,
        )

        # End episode
        stats = mem.end_episode(success=result["success"])

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}: "
                  f"experiences={stats['total_experiences']}, "
                  f"hypotheses={stats['total_hypotheses']}, "
                  f"principles={stats['total_principles']}")

    # Print learned knowledge
    print("\n" + "=" * 60)
    print("LEARNED HYPOTHESES:")
    print("=" * 60)
    for h in mem.get_hypotheses():
        print(f"  [{h.status}] {h.statement}")
        print(f"    confidence={h.confidence:.2f}, type={h.hypothesis_type}")

    print("\n" + "=" * 60)
    print("LEARNED PRINCIPLES:")
    print("=" * 60)
    for p in mem.get_principles(top_k=20):
        print(f"  [{p.principle_type}] {p.content}")
        print(f"    importance={p.importance_score:.1f}, confidence={p.confidence:.2f}")

    # Print stats
    print("\n" + "=" * 60)
    print("STATISTICS:")
    print("=" * 60)
    stats = mem.get_stats()
    print(f"  Episodes: {stats['episode_count']}")
    print(f"  Experiences: {stats['memory']['total']}")
    print(f"  Hypotheses: {stats['hypotheses']['total']}")
    print(f"  Principles: {stats['principles']['total']}")

    # Save state
    mem.save_state()
    print(f"\nState saved to {config.save_path}")

    # Shutdown
    mem.shutdown()


if __name__ == "__main__":
    main()
