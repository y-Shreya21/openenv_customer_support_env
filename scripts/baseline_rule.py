from __future__ import annotations

import json

from customer_support_env import CustomerSupportEnv, SupportAction, TaskId, TicketPlan


def heuristic_action(task_id: TaskId):
    if task_id == TaskId.EASY:
        return SupportAction(
            task_id=task_id,
            finalize=True,
            plans=[TicketPlan(ticket_id="T-1001", category="login_access", priority=2, owner_team="identity_support", reply="Sorry about the login trouble. Please reset your password once more and verify the email address on the account so we can restore access quickly.")],
        )
    if task_id == TaskId.MEDIUM:
        return SupportAction(
            task_id=task_id,
            finalize=True,
            plans=[TicketPlan(ticket_id="T-2001", category="billing_refund", priority=3, owner_team="billing_ops", reply="Sorry for the delay. I am checking the refund status now and will confirm the next update once the payment processor posts it back to your bank account.")],
        )
    return SupportAction(
        
    task_id=task_id,
    step_type="respond",
    finalize=True,
    plans=[
        TicketPlan(
            ticket_id="T-3001",
            category="general",
            priority=3,
            owner_team="support_team",
            reply="We will check this.",
            escalate=False
        ),
        TicketPlan(
            ticket_id="T-3002",
            category="general",
            priority=3,
            owner_team="support_team",
            reply="We will check this.",
            escalate=False
        ),
        TicketPlan(
            ticket_id="T-3003",
            category="general",
            priority=3,
            owner_team="support_team",
            reply="We will check this.",
            escalate=False
        ),
    ],
)
        
    
    

    


def run_task(task_id: TaskId) -> float:
    env = CustomerSupportEnv(task_id=task_id)
    env.reset(task_id=task_id)
    result = env.step(heuristic_action(task_id))
    return float(result.info["task_score"])


def main() -> None:
    scores = {task.value: run_task(task) for task in TaskId}
    avg = sum(scores.values()) / len(scores)
    print(json.dumps({"scores": scores, "average": round(avg, 4)}, indent=2))


if __name__ == "__main__":
    main()
