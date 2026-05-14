from __future__ import annotations

from customer_support_env import SupportAction, SupportObservation, TaskId, TicketPlan


def _easy_action() -> SupportAction:
    return SupportAction(
        task_id=TaskId.EASY,
        finalize=True,
        plans=[
            TicketPlan(
                ticket_id="T-1001",
                category="login_access",
                priority=2,
                owner_team="identity_support",
                escalate=False,
                reply=(
                    "Sorry for the login trouble. Please try one password reset, "
                    "confirm the account email, and we'll help restore access quickly."
                ),
            )
        ],
    )


def _medium_action() -> SupportAction:
    return SupportAction(
        task_id=TaskId.MEDIUM,
        finalize=True,
        plans=[
            TicketPlan(
                ticket_id="T-2001",
                category="billing_refund",
                priority=3,
                owner_team="billing_ops",
                escalate=False,
                reply=(
                    "Sorry about the delay. I'm checking your refund timeline now and "
                    "will share the next update after payment processor confirmation."
                ),
            )
        ],
    )


def _hard_action() -> SupportAction:
    return SupportAction(
        task_id=TaskId.HARD,
        finalize=True,
        plans=[
            TicketPlan(
                ticket_id="T-3001",
                category="incident_outage",
                priority=1,
                owner_team="infra_oncall",
                escalate=True,
                reply="We confirm an outage and escalated to infra on-call right away.",
            ),
            TicketPlan(
                ticket_id="T-3002",
                category="billing_refund",
                priority=2,
                owner_team="billing_ops",
                escalate=False,
                reply="Sorry for the duplicate charge, billing will review a refund now.",
            ),
            TicketPlan(
                ticket_id="T-3003",
                category="product_feedback",
                priority=4,
                owner_team="product_ops",
                escalate=False,
                reply="Thanks for the feature request, product team will review it.",
            ),
        ],
    )


def predict(observation: SupportObservation) -> SupportAction:
    if observation.task_id == TaskId.EASY:
        return _easy_action()
    if observation.task_id == TaskId.MEDIUM:
        return _medium_action()
    return _hard_action()
