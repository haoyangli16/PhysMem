"""
Example: Integrating PhysMem with the Reflect-VLM assembly task.

This shows the integration pattern for using PhysMem with the
MS-HAB puzzle assembly environment and a VLM agent.

NOTE: This requires the reflect-vlm environment to be set up.
      See thirdparty/reflect-vlm/ for installation instructions.

Usage:
    python run_with_physmem.py --provider qwen --n_episodes 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# PhysMem imports
from physmem import PhysMem, ScientificLearningConfig, Principle
from physmem.llm import create_llm


# ============================================================================
# Symbolic State Extraction (Domain-Specific)
# ============================================================================

def extract_symbolic_state(env_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract discrete symbolic state from the assembly environment.

    This is the domain-specific part - you define what state features
    are relevant for your task. PhysMem stores and clusters by these.

    Args:
        env_info: Environment observation/info dict.

    Returns:
        Dict of discrete state features.
    """
    # Example for MS-HAB assembly task:
    return {
        "action_type": env_info.get("action_type", "unknown"),
        "is_holding": env_info.get("is_holding", False),
        "progress": env_info.get("progress", 0.0),
        "num_remaining": env_info.get("num_remaining", 0),
        "dependencies_satisfied": env_info.get("dependencies_satisfied", True),
        "last_action_success": env_info.get("last_action_success", None),
        "last_fail_tag": env_info.get("last_fail_tag", None),
    }


# ============================================================================
# Main Integration Loop
# ============================================================================

def run_episode(
    mem: PhysMem,
    vlm_agent: Any,
    env: Any,
    episode_id: str,
) -> Dict[str, Any]:
    """
    Run a single episode with PhysMem integration.

    This demonstrates the core integration pattern:
    1. Get principles -> inject into VLM prompt
    2. Agent acts -> observe outcome
    3. Record experience -> PhysMem learns
    """
    obs = env.reset()
    done = False
    step = 0
    episode_success = False
    actions = []

    while not done:
        # 1. Get current knowledge for prompt injection.
        #    get_active_knowledge_prompt() returns BOTH verified
        #    principles and active (proposed/verified) hypotheses,
        #    matching the paper's three-tier memory injection.
        principles = mem.get_principles()
        knowledge_text = mem.get_active_knowledge_prompt()

        # 2. Get environment info for symbolic state
        env_info = env.get_info()  # domain-specific
        symbolic_state = extract_symbolic_state(env_info)

        # 3. Agent generates action (with principles + hypotheses in prompt)
        action = vlm_agent.act(
            observation=obs,
            principles=knowledge_text,  # Inject learned knowledge
        )
        actions.append(action)

        # 4. Execute action
        obs, reward, done, info = env.step(action)

        # 5. Determine outcome
        success = info.get("success", False)
        fail = info.get("fail", False)
        fail_tag = info.get("fail_tag", None)
        oracle_action = info.get("oracle_action", None)

        # 6. Record experience in PhysMem
        eid, is_surprising = mem.record_experience(
            action=action,
            success=success,
            fail=fail,
            fail_tag=fail_tag,
            symbolic_state=symbolic_state,
            state_vec=info.get("state_vec", None),  # optional embedding
            oracle_action=oracle_action,
            active_principles=principles if principles else None,
        )

        if is_surprising:
            print(f"  [SURPRISE] Step {step}: {action} -> {'success' if success else f'fail ({fail_tag})'}")

        step += 1
        if success and info.get("task_complete", False):
            episode_success = True

    # 7. End episode
    stats = mem.end_episode(
        success=episode_success,
        episode_id=episode_id,
    )

    return {
        "episode_id": episode_id,
        "success": episode_success,
        "steps": step,
        "actions": actions,
        **stats,
    }


def main():
    parser = argparse.ArgumentParser(description="PhysMem + Reflect-VLM Integration")
    parser.add_argument("--provider", default="qwen", help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--save_dir", default="./physmem_reflectvlm_output")
    parser.add_argument("--consolidation_interval", type=int, default=50)
    args = parser.parse_args()

    # Create LLM
    print(f"Creating LLM: {args.provider}")
    try:
        llm = create_llm(args.provider, model=args.model)
    except Exception as e:
        print(f"Failed to create LLM: {e}")
        print("Running without LLM (rule-based hypothesis generation)")
        llm = None

    # Create PhysMem
    config = ScientificLearningConfig(
        memory_name="reflect_vlm_assembly",
        max_memory_size=3000,
        consolidation_interval=args.consolidation_interval,
        max_hypotheses_per_cluster=3,
        max_principles_in_prompt=5,
        save_path=args.save_dir,
        auto_save_interval=50,
    )
    mem = PhysMem(config=config, llm=llm)

    print(f"PhysMem initialized. Running {args.n_episodes} episodes...")
    print(f"Save directory: {args.save_dir}")

    # NOTE: Replace these with actual environment and agent initialization
    # from thirdparty/reflect-vlm:
    #
    #   from roboworld.agent.vlm_api import UnifiedVLM
    #   vlm_agent = UnifiedVLM(provider=args.provider, model=args.model)
    #   env = make_assembly_env(...)
    #
    # For now, we show the integration pattern:

    print("\n" + "=" * 60)
    print("NOTE: This example requires the reflect-vlm environment.")
    print("The integration pattern is shown in run_episode().")
    print("To run with a real environment, replace the agent/env")
    print("initialization above with your actual setup.")
    print("=" * 60)

    # Example: simulate some episodes for demonstration
    for ep in range(min(5, args.n_episodes)):
        # Simulated experience recording
        for step in range(3):
            success = np.random.random() > 0.3
            mem.record_experience(
                action=f"step_{step}",
                success=success,
                fail=not success,
                fail_tag="collision" if not success else None,
                symbolic_state={
                    "action_type": "grasp" if step == 0 else "insert",
                    "is_holding": step > 0,
                    "progress": step / 3,
                },
            )
        mem.end_episode(success=True, episode_id=f"demo_ep_{ep}")

    # Print results
    stats = mem.get_stats()
    print(f"\nFinal stats: {json.dumps(stats, indent=2, default=str)}")

    mem.save_state()
    mem.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
