from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from customer_support_env import CustomerSupportEnv, SupportAction, TaskId, TicketPlan

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI()


def _system_prompt() -> str:
    return (
        "You are a support operations agent. Read the task and output ONLY valid JSON. "
        "Return keys: task_id, finalize, plans. Each plan must contain ticket_id, category, priority, owner_team, reply, escalate, eta_hours, status."
    )


def infer_action(observation: Dict[str, Any]) -> SupportAction:
    prompt = json.dumps(observation, indent=2)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        top_p=1,
        seed=42,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    data = json.loads(text)
    return SupportAction.model_validate(data)


def run_task(task_id: TaskId) -> float:
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset(task_id=task_id)
    action = infer_action(obs.model_dump())
    result = env.step(action)
    return float(result.info["task_score"])


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    scores = {task.value: run_task(task) for task in TaskId}
    avg = sum(scores.values()) / len(scores)
    print(json.dumps({"model": MODEL, "scores": scores, "average": round(avg, 4)}, indent=2))


if __name__ == "__main__":
    main()
