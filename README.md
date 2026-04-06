# Customer Support Triage Environment (OpenEnv)
#Description
This environment simulates real-world customer support workflows, where agents must classify, prioritize, route, and respond to support tickets.

Unlike toy environments, this task reflects actual enterprise support systems, including:
- Ticket classification
- Priority assignment
- Team routing
- Escalation decisions
- Response drafting

The environment supports multi-step decision-making and provides partial rewards based on correctness and response quality.
#Tasks
## Tasks

### Easy
Login issue triage for a single user.

### Medium
Refund delay response with policy-safe communication.

### Hard
Multi-ticket queue planning with prioritization and escalation.

## Reward Design

Reward is based on:
- Classification accuracy
- Priority correctness
- Routing decisions
- Escalation correctness
- Response quality (keywords + tone)

Partial rewards are given for each component, enabling learning across steps.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

python scripts/baseline_rule.py

## 🔹 Baseline Results
```md
## Baseline Results

| Task | Score |
|------|------|
| Easy | 0.80 |
| Medium | 0.84 |
| Hard | 0.54 |
| Average | 0.72 |

2. OpenAI baseline
In baseline_openai.py, ensure:

import os
api_key = os.getenv("OPENAI_API_KEY")
