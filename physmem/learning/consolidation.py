"""
Consolidation Engine for the Scientific Learning Loop.

Responsibilities:
1. Periodically clustering raw experiences by similarity
2. Analyzing patterns within clusters
3. Generating hypotheses from patterns (both success and failure)
4. Managing the consolidation lifecycle (sync or async)
"""

from __future__ import annotations

import json
import uuid
import threading
import queue
import time
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from physmem.core.experience import Experience, MemoryBank
from physmem.core.hypothesis import (
    Hypothesis,
    HypothesisStore,
    ExperienceCluster,
    PrincipleType,
)
from physmem.llm.base import BaseLLM

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ConsolidationConfig:
    """Configuration for the consolidation engine."""
    consolidation_interval: int = 10
    min_experiences_for_consolidation: int = 3
    min_cluster_size: int = 2
    max_clusters: int = 30
    similarity_threshold: float = 0.6
    min_experiences_for_hypothesis: int = 2
    max_hypotheses_per_cluster: int = 3
    use_semantic_embedding: bool = False
    semantic_model_name: str = "all-MiniLM-L6-v2"
    run_async: bool = True
    max_queue_size: int = 1000


# ============================================================================
# Default Hypothesis Generation Prompt
# ============================================================================

HYPOTHESIS_GENERATION_PROMPT = """You are a scientist analyzing experiences to generate testable hypotheses.

## Existing Knowledge (DO NOT DUPLICATE)
{existing_knowledge_context}

## Cluster Analysis ({n_experiences} experiences)
### Pattern Summary
{pattern_summary}

### Outcomes: {n_success} successes, {n_fail} failures ({success_rate:.1%} success rate)

### Sample Experiences:
{sample_experiences}

## Task
Generate EXACTLY 1 hypothesis that:
- Is ABSTRACT and GENERAL (no specific counts or instance-specific details)
- Captures a pattern that applies across multiple scenarios
- Is NOT redundant with existing knowledge
- Is TESTABLE with clear cause and effect

{hypothesis_type_instruction}

Output ONLY valid JSON:
[
    {{
        "statement": "When [CONDITION], [ACTION] leads to [OUTCOME] because [REASON]",
        "hypothesis_type": "PREFER"
    }}
]

Required: "statement" (string), "hypothesis_type" (AVOID/PREFER/SEQUENCE/COMPARE/GENERAL)
"""

HYPOTHESIS_TYPE_INSTRUCTIONS = {
    "failure": 'Focus on AVOIDANCE: "Avoid [ACTION] when [CONDITION] because it leads to [FAILURE]"',
    "success": 'Focus on PREFERENCE: "Prefer [ACTION] when [CONDITION] because it leads to [SUCCESS]"',
    "mixed": 'Focus on COMPARISON: "When [CONDITION], [A] succeeds but [B] fails because [REASON]"',
}


# ============================================================================
# Consolidation Engine
# ============================================================================

class ConsolidationEngine:
    """
    Engine for consolidating raw experiences into hypotheses.

    Clusters similar experiences and generates hypotheses using an LLM
    (or rule-based fallback when no LLM is provided).

    Usage::

        engine = ConsolidationEngine(memory, hypothesis_store, llm=my_llm)
        engine.add_experience(exp)

        # Manual consolidation
        clusters = engine.consolidate()
        hypotheses = engine.generate_hypotheses(clusters)

        # Or run as background thread
        engine.start_background()
        ...
        engine.stop_background()
    """

    def __init__(
        self,
        memory: MemoryBank,
        hypothesis_store: HypothesisStore,
        config: Optional[ConsolidationConfig] = None,
        llm: Optional[BaseLLM] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        principle_store: Optional[Any] = None,
    ):
        self.memory = memory
        self.hypothesis_store = hypothesis_store
        self.principle_store = principle_store
        self.config = config or ConsolidationConfig()
        self.llm = llm
        self.embedder = embedder

        self._consolidated_exp_ids: set = set()
        self.clusters: List[ExperienceCluster] = []

        # Background processing
        self._experience_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._consolidation_count = 0
        self._hypothesis_count = 0

        # Initialize semantic model if configured
        self.semantic_model = None
        if self.config.use_semantic_embedding and HAS_SBERT:
            try:
                self.semantic_model = SentenceTransformer(self.config.semantic_model_name)
            except Exception as e:
                print(f"[ConsolidationEngine] Warning: Failed to load SBERT: {e}")

    # =========================================================================
    # Experience Tracking
    # =========================================================================

    def add_experience(self, exp: Experience) -> None:
        """Add an experience for consolidation (thread-safe)."""
        if self.config.run_async:
            try:
                self._experience_queue.put_nowait(exp)
            except queue.Full:
                pass

    def get_unconsolidated(self, limit: int = 100) -> List[Experience]:
        """Get experiences that haven't been consolidated yet."""
        result = []
        for exp in self.memory.experiences:
            if exp.eid not in self._consolidated_exp_ids:
                result.append(exp)
                if len(result) >= limit:
                    break
        return result

    def mark_consolidated(self, exp_id: str) -> None:
        self._consolidated_exp_ids.add(exp_id)

    # =========================================================================
    # Clustering
    # =========================================================================

    def consolidate(
        self, experiences: Optional[List[Experience]] = None
    ) -> List[ExperienceCluster]:
        """Cluster experiences and identify patterns."""
        if experiences is None:
            experiences = self.get_unconsolidated(limit=200)

        if len(experiences) < self.config.min_experiences_for_consolidation:
            return []

        embeddings = self._get_experience_embeddings(experiences)
        if embeddings is None or len(embeddings) == 0:
            return []

        clusters = self._cluster_experiences(experiences, embeddings)

        for exp in experiences:
            self.mark_consolidated(exp.eid)

        self._consolidation_count += 1
        self.clusters.extend(clusters)
        return clusters

    def _get_experience_embeddings(self, experiences: List[Experience]) -> Optional[np.ndarray]:
        """Get embeddings for experiences."""
        if self.semantic_model:
            texts = [self._symbolic_to_text(e.symbolic_state) for e in experiences]
            return self.semantic_model.encode(texts, show_progress_bar=False)

        embeddings = []
        for exp in experiences:
            if self.embedder and exp.symbolic_state:
                text = self._symbolic_to_text(exp.symbolic_state)
                emb = np.array(self.embedder(text), dtype=np.float32)
                embeddings.append(emb)
            elif exp.state_vec is not None:
                embeddings.append(exp.state_vec)
            elif exp.symbolic_state:
                emb = self._symbolic_to_embedding(exp.symbolic_state)
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(128, dtype=np.float32))

        if not embeddings:
            return None

        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return embeddings / norms

    def _symbolic_to_text(self, sym: Optional[Dict[str, Any]]) -> str:
        """Convert symbolic state to text for semantic embedding."""
        if not sym:
            return "unknown state"
        parts = []
        for key, val in sorted(sym.items()):
            if isinstance(val, (str, int, float, bool)):
                parts.append(f"{key}: {val}")
            elif isinstance(val, list) and len(val) <= 5:
                parts.append(f"{key}: {val}")
        return "; ".join(parts) if parts else "unknown state"

    def _symbolic_to_embedding(self, symbolic_state: Dict[str, Any]) -> np.ndarray:
        """Convert symbolic state to a simple embedding (fallback)."""
        features = []
        for key in sorted(symbolic_state.keys()):
            val = symbolic_state[key]
            if isinstance(val, bool):
                features.append(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, str):
                features.append(hash(val) % 1000 / 1000.0)
        # Pad to fixed length
        while len(features) < 128:
            features.append(0.0)
        return np.array(features[:128], dtype=np.float32)

    def _cluster_experiences(
        self, experiences: List[Experience], embeddings: np.ndarray
    ) -> List[ExperienceCluster]:
        """Cluster experiences using hierarchical clustering."""
        if not HAS_SKLEARN or len(experiences) < self.config.min_cluster_size:
            # Fallback: single cluster
            cluster = ExperienceCluster(
                experience_ids=[e.eid for e in experiences],
                outcome_distribution={
                    "success": sum(1 for e in experiences if e.success),
                    "fail": sum(1 for e in experiences if e.fail),
                },
                size=len(experiences),
            )
            return [cluster]

        n_clusters = min(self.config.max_clusters, len(experiences) // self.config.min_cluster_size)
        if n_clusters < 1:
            n_clusters = 1

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        clusters = []
        for label in set(labels):
            mask = labels == label
            cluster_exps = [experiences[i] for i in range(len(experiences)) if mask[i]]

            if len(cluster_exps) < self.config.min_cluster_size:
                continue

            cluster = ExperienceCluster(
                experience_ids=[e.eid for e in cluster_exps],
                outcome_distribution={
                    "success": sum(1 for e in cluster_exps if e.success),
                    "fail": sum(1 for e in cluster_exps if e.fail),
                },
                size=len(cluster_exps),
            )
            clusters.append(cluster)

        return clusters

    # =========================================================================
    # Hypothesis Generation
    # =========================================================================

    def generate_hypotheses(
        self, clusters: List[ExperienceCluster]
    ) -> List[Hypothesis]:
        """Generate hypotheses from clusters."""
        all_hypotheses = []

        for cluster in clusters:
            if cluster.size < self.config.min_experiences_for_hypothesis:
                continue

            hypotheses = self._generate_for_cluster(cluster)
            for h in hypotheses[: self.config.max_hypotheses_per_cluster]:
                hid = self.hypothesis_store.add(h)
                if hid:
                    cluster.generated_hypotheses.append(hid)
                    all_hypotheses.append(h)

        self._hypothesis_count += len(all_hypotheses)
        return all_hypotheses

    def _generate_for_cluster(self, cluster: ExperienceCluster) -> List[Hypothesis]:
        """Generate hypotheses for a single cluster."""
        if self.llm:
            return self._generate_with_llm(cluster)
        return self._generate_rule_based(cluster)

    def _generate_with_llm(self, cluster: ExperienceCluster) -> List[Hypothesis]:
        """Use LLM to generate hypotheses from cluster patterns."""
        experiences = [
            self.memory.get(eid) for eid in cluster.experience_ids
        ]
        experiences = [e for e in experiences if e is not None]

        if not experiences:
            return []

        # Determine hypothesis type
        if cluster.is_mostly_failures:
            h_type_instruction = HYPOTHESIS_TYPE_INSTRUCTIONS["failure"]
        elif cluster.is_mostly_successes:
            h_type_instruction = HYPOTHESIS_TYPE_INSTRUCTIONS["success"]
        else:
            h_type_instruction = HYPOTHESIS_TYPE_INSTRUCTIONS["mixed"]

        # Build existing knowledge context
        existing_context = self._get_existing_knowledge_context()

        # Build sample experiences text
        samples = experiences[:10]
        sample_text = ""
        for i, exp in enumerate(samples):
            outcome = "SUCCESS" if exp.success else f"FAIL ({exp.fail_tag or 'unknown'})"
            action = exp.extra_metrics.get("action", "unknown") if exp.extra_metrics else "unknown"
            sample_text += f"\n{i+1}. Action: {action} | Outcome: {outcome}"
            if exp.symbolic_state:
                sample_text += f" | State: {self._symbolic_to_text(exp.symbolic_state)}"
            if exp.extra_metrics and exp.extra_metrics.get("oracle_action"):
                sample_text += f" | Oracle: {exp.extra_metrics['oracle_action']}"

        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            existing_knowledge_context=existing_context,
            n_experiences=cluster.size,
            pattern_summary=cluster.common_pattern or "Automatic cluster",
            n_success=cluster.outcome_distribution.get("success", 0),
            n_fail=cluster.outcome_distribution.get("fail", 0),
            success_rate=cluster.success_rate,
            sample_experiences=sample_text,
            hypothesis_type_instruction=h_type_instruction,
        )

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt="You are a scientist generating testable hypotheses from experience data. Output ONLY valid JSON.",
                max_tokens=512,
                temperature=0.5,
            )
            return self._parse_hypothesis_response(response, cluster)
        except Exception as e:
            print(f"[ConsolidationEngine] LLM hypothesis generation failed: {e}")
            return self._generate_rule_based(cluster)

    def _get_existing_knowledge_context(self) -> str:
        """Build context of existing hypotheses and principles."""
        lines = []
        # Existing hypotheses
        for h in self.hypothesis_store.hypotheses[-20:]:
            lines.append(f"- [H] {h.statement}")
        # Existing principles
        if self.principle_store:
            for p in self.principle_store.principles[-20:]:
                lines.append(f"- [P] {p.content}")
        return "\n".join(lines) if lines else "None yet."

    def _parse_hypothesis_response(
        self, response: str, cluster: ExperienceCluster
    ) -> List[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        hypotheses = []
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(response[start:end])
                for item in items:
                    h_type_str = item.get("hypothesis_type", "GENERAL").upper()
                    try:
                        h_type = PrincipleType(h_type_str.lower())
                    except ValueError:
                        h_type = PrincipleType.GENERAL

                    h = Hypothesis(
                        statement=item.get("statement", ""),
                        hypothesis_type=h_type,
                        source_experience_ids=cluster.experience_ids[:5],
                        source_cluster_id=cluster.cid,
                        action_types=item.get("action_types", []),
                    )
                    if h.statement:
                        hypotheses.append(h)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[ConsolidationEngine] Failed to parse LLM response: {e}")
        return hypotheses

    def _generate_rule_based(self, cluster: ExperienceCluster) -> List[Hypothesis]:
        """Generate hypotheses using rules (no LLM needed).

        The rule-based path extracts the dominant action and the dominant
        symbolic-state features from the cluster so that the resulting
        hypotheses are *actionable*: downstream policies can filter on
        ``action_types`` / ``trigger_conditions`` without needing an LLM.
        """
        hypotheses: List[Hypothesis] = []

        # Fetch the actual experiences backing this cluster.
        experiences: List[Experience] = []
        for eid in cluster.experience_ids:
            exp = self.memory.get(eid)
            if exp is not None:
                experiences.append(exp)
        if not experiences:
            return hypotheses

        dominant_action = self._dominant_action(experiences)
        dominant_state, trigger_conditions = self._dominant_symbolic_state(experiences)

        n_fail = cluster.outcome_distribution.get("fail", 0)
        n_succ = cluster.outcome_distribution.get("success", 0)
        total = max(n_fail + n_succ, 1)

        def _format(stmt_prefix: str, outcome_count: int) -> str:
            parts = [stmt_prefix]
            if dominant_action:
                parts.append(f"'{dominant_action}'")
            if dominant_state:
                parts.append(f"when {dominant_state}")
            parts.append(f"({outcome_count}/{total})")
            return " ".join(parts)

        action_types = [dominant_action] if dominant_action else []

        if cluster.is_mostly_failures:
            h = Hypothesis(
                statement=_format("Avoid", n_fail),
                hypothesis_type=PrincipleType.AVOID,
                source_experience_ids=cluster.experience_ids[:5],
                source_cluster_id=cluster.cid,
                action_types=action_types,
                trigger_conditions=trigger_conditions,
            )
            hypotheses.append(h)
        elif cluster.is_mostly_successes:
            h = Hypothesis(
                statement=_format("Prefer", n_succ),
                hypothesis_type=PrincipleType.PREFER,
                source_experience_ids=cluster.experience_ids[:5],
                source_cluster_id=cluster.cid,
                action_types=action_types,
                trigger_conditions=trigger_conditions,
            )
            hypotheses.append(h)

        return hypotheses

    def _dominant_action(self, experiences: List[Experience]) -> Optional[str]:
        """Return the most common action string across the cluster."""
        counter: Counter = Counter()
        for exp in experiences:
            if exp.extra_metrics:
                action = exp.extra_metrics.get("action")
                if action:
                    counter[str(action)] += 1
        if not counter:
            return None
        action, _ = counter.most_common(1)[0]
        return action

    def _dominant_symbolic_state(
        self, experiences: List[Experience]
    ) -> Tuple[str, List[str]]:
        """Extract symbolic-state keys whose values are dominant in the cluster.

        Returns ``(human_readable_summary, trigger_conditions_list)``.
        A key is considered dominant when its modal value occurs in at
        least 60% of experiences that reported that key.
        """
        per_key_counter: Dict[str, Counter] = defaultdict(Counter)
        per_key_total: Dict[str, int] = defaultdict(int)
        for exp in experiences:
            if not exp.symbolic_state:
                continue
            for key, val in exp.symbolic_state.items():
                if isinstance(val, (str, int, float, bool)):
                    per_key_counter[key][val] += 1
                    per_key_total[key] += 1

        dominant_parts: List[str] = []
        trigger_conditions: List[str] = []
        for key, counter in per_key_counter.items():
            val, count = counter.most_common(1)[0]
            total = per_key_total[key]
            if total > 0 and count / total >= 0.6:
                dominant_parts.append(f"{key}={val}")
                trigger_conditions.append(f"{key}={val}")

        return ", ".join(dominant_parts), trigger_conditions

    # =========================================================================
    # Background Processing
    # =========================================================================

    def start_background(self):
        """Start background consolidation thread."""
        if self._background_thread and self._background_thread.is_alive():
            return
        self._stop_event.clear()
        self._background_thread = threading.Thread(
            target=self._background_loop, daemon=True
        )
        self._background_thread.start()

    def stop_background(self):
        """Stop background consolidation thread."""
        self._stop_event.set()
        if self._background_thread:
            self._background_thread.join(timeout=5)

    def _background_loop(self):
        """Background loop that periodically runs consolidation."""
        while not self._stop_event.is_set():
            # Drain the queue
            while not self._experience_queue.empty():
                try:
                    self._experience_queue.get_nowait()
                except queue.Empty:
                    break

            # Run consolidation
            try:
                clusters = self.consolidate()
                if clusters:
                    self.generate_hypotheses(clusters)
            except Exception as e:
                print(f"[ConsolidationEngine] Background error: {e}")

            self._stop_event.wait(timeout=self.config.consolidation_interval)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "consolidation_count": self._consolidation_count,
            "hypothesis_count": self._hypothesis_count,
            "cluster_count": len(self.clusters),
            "unconsolidated": len(self.memory) - len(self._consolidated_exp_ids),
        }
