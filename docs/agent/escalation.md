# Escalation Guide

When the agent hits a blocking issue it cannot resolve, it escalates: marks the step `blocked` and writes an escalation document with full context.

## What triggers escalation

- `revert_step` returns `exhausted=true` (max retries reached)
- Agent determines the blocker is outside its capability (missing credentials, external system failure, ambiguous requirements, etc.)

## How to escalate (agent)

Call `report_blocking_issue`:

```python
report_blocking_issue(
    plan_id="<plan_id>",
    step_id="<step_id>",
    issue="Clear description of what is blocking",
    what_was_tried="Approaches attempted",
    suggested_resolution="Optional: what might fix it",
)
```

Or via slash command:

```
/plan escalate <step_id> <reason>
```

Both write `.agent/plans/{plan_id}_{step_id}_escalation.md` and mark the step `blocked`.

## Escalation document structure

```
# Blocking Issue: <step description>

Created / Plan ID / Step ID

## Plan Goal
## Shared Context       ← plan.context
## Step Introduction    ← step.introduction
## Step Description
## Acceptance Criteria
## Blocking Issue       ← what the agent reported
## What Was Tried
## Suggested Resolution
```

## Handing off to a stronger model or human

1. Open the escalation doc: `.agent/plans/<plan_id>_<step_id>_escalation.md`
2. Share it verbatim with the reviewer (copy-paste or attach).
3. Provide any relevant files / error logs not already in the doc.
4. Ask the reviewer to resolve the blocker, then resume:

```
/plan step <step_id> pending
/plan resume
```

Or instruct a stronger model:

> "Read the escalation doc below. Resolve the blocking issue, implement the step, then mark it complete."
> [paste doc]

## Checking blocked steps

```
/plan show
```

Steps marked `⚠ BLOCKED` are escalated. Check `.agent/plans/` for `*_escalation.md` files.

## After resolution

Once resolved (by human or stronger model), reset the step:

```
/plan step <step_id> pending
```

Then continue plan execution normally. The plan runner will pick up ready steps.
