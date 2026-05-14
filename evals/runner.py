"""
FORGE — EvalRunner
The main orchestrator for running evaluation suites.

Usage:
    runner = EvalRunner(
        adapter=get_adapter("claude", "claude-sonnet-4-6"),
        scorers=default_scorers(judge_adapter),
        concurrency=4,
    )
    results = runner.run(dataset)
    report  = runner.report(results)
"""

from __future__ import annotations
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

from forge.core.models import EvalItem, EvalResult
from forge.core.adapters import BaseAdapter
from forge.evals.dataset import EvalDataset
from forge.evals.scorers.scorers import BaseScorer, default_scorers

logger = logging.getLogger("forge.evals")


class EvalRunner:
    """
    Runs an EvalDataset against a model adapter, scoring each response
    across all configured scorers.

    Args:
        adapter:      The model adapter to evaluate (ClaudeAdapter, GPTAdapter, etc.)
        scorers:      List of scorers to apply. Defaults to all four FORGE scorers.
        concurrency:  Number of parallel API calls. Default 1 (safe for rate limits).
        system:       Optional system prompt to prepend to every eval item.
        on_result:    Optional callback fired after each result (for streaming UIs).
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        scorers: list[BaseScorer] | None = None,
        concurrency: int = 1,
        system: str = "",
        on_result: Callable[[EvalResult], None] | None = None,
    ):
        self.adapter     = adapter
        self.scorers     = scorers or default_scorers()
        self.concurrency = concurrency
        self.system      = system
        self.on_result   = on_result

    # ── Main run ─────────────────────────────

    def run(self, dataset: EvalDataset) -> list[EvalResult]:
        """Run all items in the dataset. Returns list of EvalResults."""
        logger.info(f"Starting eval: {dataset.name} | {len(dataset)} items | model={self.adapter.model}")

        if self.concurrency > 1:
            return self._run_parallel(dataset)
        return self._run_sequential(dataset)

    def _run_sequential(self, dataset: EvalDataset) -> list[EvalResult]:
        results = []
        for i, item in enumerate(dataset, 1):
            result = self._eval_item(item)
            results.append(result)
            if self.on_result:
                self.on_result(result)
            logger.debug(f"[{i}/{len(dataset)}] {item.item_id} → {result.overall_score:.2f}")
        return results

    def _run_parallel(self, dataset: EvalDataset) -> list[EvalResult]:
        results = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._eval_item, item): item for item in dataset}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if self.on_result:
                    self.on_result(result)
        return sorted(results, key=lambda r: r.timestamp)

    def _eval_item(self, item: EvalItem) -> EvalResult:
        t0 = time.time()
        try:
            response = self.adapter.complete(item.prompt, system=self.system)
            actual      = response.text
            latency_ms  = response.latency_ms
            token_count = response.input_tokens + response.output_tokens
        except Exception as e:
            logger.warning(f"Model call failed for item {item.item_id}: {e}")
            actual     = f"[ERROR: {e}]"
            latency_ms = round((time.time() - t0) * 1000, 2)
            token_count = 0

        scores = [
            s.score(item.expected, actual, item.prompt)
            for s in self.scorers
        ]

        return EvalResult(
            item_id=item.item_id,
            model=self.adapter.model,
            provider=self._infer_provider(),
            prompt=item.prompt,
            expected=item.expected,
            actual=actual,
            scores=scores,
            latency_ms=latency_ms,
            token_count=token_count,
        )

    def _infer_provider(self):
        from forge.core.models import ModelProvider
        name = type(self.adapter).__name__.lower()
        if "claude" in name:   return ModelProvider.ANTHROPIC
        if "openai" in name:   return ModelProvider.OPENAI
        if "gemini" in name:   return ModelProvider.GOOGLE
        if "mistral" in name:  return ModelProvider.MISTRAL
        return ModelProvider.OLLAMA

    # ── Reporting ────────────────────────────

    def report(self, results: list[EvalResult]) -> EvalReport:
        return EvalReport(results)

    def save(self, results: list[EvalResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "run_id":        r.run_id,
                "item_id":       r.item_id,
                "model":         r.model,
                "overall_score": r.overall_score,
                "passed":        r.passed,
                "latency_ms":    r.latency_ms,
                "token_count":   r.token_count,
                "scores":        [{"dim": s.dimension, "score": s.score, "passed": s.passed, "explanation": s.explanation} for s in r.scores],
                "timestamp":     r.timestamp.isoformat(),
            }
            for r in results
        ]
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Results saved → {path}")


# ─────────────────────────────────────────────
# EvalReport — summary and CI gate
# ─────────────────────────────────────────────

class EvalReport:
    """
    Summary statistics over a list of EvalResults.
    Use .ci_gate() to fail a CI pipeline if quality drops below threshold.
    """

    def __init__(self, results: list[EvalResult]):
        self.results   = results
        self.timestamp = datetime.utcnow()

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.overall_score for r in self.results) / self.total, 4)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.latency_ms for r in self.results) / self.total, 2)

    def by_dimension(self) -> dict[str, float]:
        """Average score per scoring dimension."""
        dim_scores: dict[str, list[float]] = {}
        for r in self.results:
            for s in r.scores:
                dim_scores.setdefault(s.dimension, []).append(s.score)
        return {
            dim: round(sum(scores) / len(scores), 4)
            for dim, scores in dim_scores.items()
        }

    def ci_gate(self, min_pass_rate: float = 0.80, min_avg_score: float = 0.70) -> bool:
        """
        Returns True if the eval passes CI quality gates.
        Call this in GitHub Actions — exit(0) = pass, exit(1) = fail.

        Example in CI:
            if not report.ci_gate(min_pass_rate=0.85):
                sys.exit(1)
        """
        ok = self.pass_rate >= min_pass_rate and self.avg_score >= min_avg_score
        if not ok:
            logger.warning(
                f"CI gate FAILED — pass_rate={self.pass_rate:.0%} (min {min_pass_rate:.0%}), "
                f"avg_score={self.avg_score:.2f} (min {min_avg_score:.2f})"
            )
        return ok

    def __str__(self) -> str:
        dims = self.by_dimension()
        lines = [
            "─" * 50,
            f"  FORGE Eval Report  —  {self.timestamp:%Y-%m-%d %H:%M} UTC",
            "─" * 50,
            f"  Model    : {self.results[0].model if self.results else 'n/a'}",
            f"  Total    : {self.total}",
            f"  Passed   : {self.passed}  ({self.pass_rate:.0%})",
            f"  Failed   : {self.failed}",
            f"  Avg score: {self.avg_score:.4f}",
            f"  Avg latency: {self.avg_latency_ms:.0f} ms",
            "",
            "  Scores by dimension:",
        ]
        for dim, score in dims.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"    {dim:<16} {bar} {score:.2f}")
        lines.append("─" * 50)
        return "\n".join(lines)
