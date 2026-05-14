# FORGE 🔥
### Framework for Observability, Reliability, Governance & Evaluation

Enterprise-grade AI governance platform for LLMs, GenAI applications, and AI agents.  
Built for financial services. Applicable everywhere.

[![Eval CI](https://github.com/idmaa001/forge/actions/workflows/evals.yml/badge.svg)](https://github.com/yourusername/forge/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why FORGE?

Enterprise AI teams face four unsolved problems once models go to production:

| Problem | FORGE Module |
|---|---|
| "How do I know if my LLM is hallucinating?" | **Evals Engine** — automated evaluation pipelines |
| "How do I monitor model behaviour in production?" | **ObserveAI** — OTel-based observability |
| "How do I prevent prompt injection and PII leaks?" | **GuardRail** — policy-as-code safety layer |
| "How do I govern autonomous agents?" | **AgentGov** — lifecycle, audit, and compliance |

FORGE is **not** a wrapper. It's a platform that sits alongside your AI stack and gives engineering, risk, and compliance teams a shared language.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Enterprise AI systems               │
│   LLMs · Agents · GenAI Apps · RAG Pipelines    │
└────────┬──────────┬──────────┬──────────┬───────┘
         │          │          │          │
   ┌─────▼──┐ ┌─────▼──┐ ┌────▼───┐ ┌───▼─────┐
   │ Evals  │ │Observe │ │Guard   │ │ Agent   │
   │ Engine │ │  AI    │ │ Rail   │ │  Gov    │
   └─────┬──┘ └─────┬──┘ └────┬───┘ └───┬─────┘
         └──────────┴──────────┴─────────┘
                         │
              ┌──────────▼──────────┐
              │    Platform Core     │
              │  FastAPI · SQLite    │
              │  OpenTelemetry · ORM │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   FORGE Dashboard    │
              │  Streamlit · Plotly  │
              └─────────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/idmaa001/forge.git
cd forge
pip install -e ".[evals]"

export ANTHROPIC_API_KEY="your-key"

# Run the BFSI wealth management eval example
python examples/wealth_management/eval_advisor.py

# Run the full eval CI suite (mock mode, no API key needed)
python -m forge.evals.ci_runner --pack hallucination,bias,reasoning --provider mock
```

---

## Module 1 — Evals Engine

Run evaluation suites against any LLM. Four scoring dimensions out of the box.

```python
from forge.core.adapters import get_adapter
from forge.evals.dataset import EvalDataset
from forge.evals.runner import EvalRunner
from forge.evals.scorers.scorers import default_scorers

# Pick your model
adapter = get_adapter("claude", "claude-haiku-4-5-20251001")

# Load a dataset (built-in or custom JSON/CSV)
dataset = EvalDataset.builtin("bfsi")

# Run
runner  = EvalRunner(adapter=adapter, scorers=default_scorers())
results = runner.run(dataset)
report  = runner.report(results)

print(report)
# ──────────────────────────────────────────────────
#   FORGE Eval Report  —  2025-01-15 09:30 UTC
# ──────────────────────────────────────────────────
#   Model    : claude-haiku-4-5-20251001
#   Total    : 14
#   Passed   : 12  (86%)
#   Avg score: 0.8240
#   ...

# CI gate — returns False if quality drops below threshold
if not report.ci_gate(min_pass_rate=0.80):
    sys.exit(1)
```

**Built-in benchmark packs:**
- `hallucination` — fact-checking, invented citations, numeric accuracy
- `bias` — demographic parity, gender/age/geographic stereotyping
- `reasoning` — multi-step math, logical deduction, causal inference
- `bfsi` — wealth management, SEBI compliance, financial advice scenarios

**Supported providers:** Claude · GPT · Gemini · Mistral · Llama (via Ollama) · Mock (CI)

---

## Module 2 — ObserveAI *(Phase 2 — coming soon)*

OpenTelemetry-based observability. One decorator, full trace.

```python
from forge.observe import forge_trace

@forge_trace
def call_advisor_llm(prompt: str) -> str:
    return my_llm.complete(prompt)
# Emits: latency, tokens, quality score, drift flag
# Ships to: Prometheus → Grafana
```

---

## Module 3 — GuardRail *(Phase 3 — coming soon)*

Policy-as-code safety layer. Define rules in YAML, not code.

```yaml
# guardrail_policy.yaml
rules:
  - name: no_pii_in_output
    type: pii_scan
    severity: high
    action: redact
  - name: no_prompt_injection
    type: injection_detect
    severity: critical
    action: block
```

---

## Module 4 — AgentGov *(Phase 4 — coming soon)*

Governance control plane for LangGraph and ADK agents.

```yaml
# agent_manifest.yaml
agent_id: advisor-insight-agent-v2
risk_tier: high
allowed_tools: [portfolio_read, market_data_fetch]
prohibited_tools: [trade_execute, client_data_write]
human_escalation_on: [risk_score > 8, tool_violation]
```

---

## Roadmap

- [x] Phase 1 — Evals Engine
- [ ] Phase 2 — ObserveAI (OpenTelemetry + Grafana)
- [ ] Phase 3 — GuardRail (Presidio + policy-as-code)
- [ ] Phase 4 — AgentGov (LangGraph + audit log)
- [ ] Phase 5 — Unified Streamlit dashboard

---

## Use cases

- **Wealth management** — evaluate advisor AI for hallucinated regulatory facts, biased investment advice, and compliance boundary violations
- **Insurance** — monitor claims AI for PII leakage and policy misquotation
- **Banking** — govern autonomous agents with tool-use policies and audit trails
- **Any enterprise GenAI** — continuous eval in CI/CD, observability in production

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).  
Areas where help is most needed: new scorer types, provider adapters, dashboard visualisations.

---

## License

MIT © 2025
