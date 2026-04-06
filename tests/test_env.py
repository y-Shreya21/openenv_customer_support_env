from customer_support_env import CustomerSupportEnv, SupportAction, TaskId, TicketPlan


def test_reset_returns_observation():
    env = CustomerSupportEnv(task_id=TaskId.EASY)
    obs = env.reset()
    assert obs.task_id == TaskId.EASY
    assert obs.done is False
    assert len(obs.tickets) == 1


def test_easy_scoring_is_deterministic():
    env = CustomerSupportEnv(task_id=TaskId.EASY)
    env.reset()
    action = SupportAction(
        task_id=TaskId.EASY,
        step_type="respond",
        finalize=True,
        plans=[TicketPlan(ticket_id="T-1001", category="login_access", priority=2, owner_team="identity_support", reply="Sorry about the login trouble. Please reset your password and verify your account details so we can help restore access.")],
    )
    result = env.step(action)
    assert 0.8 <= result.info["task_score"] <= 1.0
    assert 0.0 <= result.reward.value <= 1.0
    assert result.done is True


def test_hard_task_requires_three_plans():
    env = CustomerSupportEnv(task_id=TaskId.HARD)
    env.reset()
    action = SupportAction(
        task_id=TaskId.HARD,
        step_type="respond",
        finalize=True,
        plans=[
            TicketPlan(ticket_id="T-3001", category="incident_outage", priority=1, owner_team="infra_oncall", escalate=True, reply="We are escalating the outage immediately."),
            TicketPlan(ticket_id="T-3002", category="billing_refund", priority=2, owner_team="billing_ops", escalate=False, reply="Sorry about the duplicate charge; billing will review it."),
            TicketPlan(ticket_id="T-3003", category="product_feedback", priority=4, owner_team="product_ops", escalate=False, reply="Thanks for the feature idea; product will review it."),
        ],
    )
    result = env.step(action)
    assert result.info["task_score"] > 0.7
