# Backlog UI — decisions worth not re-litigating

2026-07-25. The store (`.agent/ideas.db`) is old; the UI on top of it is new.
These are the choices that shaped it, and the reasons, so the next change does
not quietly undo one of them.

## Destructive actions are two clicks

Marking a task done or rejected lives behind a `⋯` menu, not as `✓`/`✗` in the
row. In a dense list of one-line rows the buttons sit under the pointer while
you are reading, and closing someone's task is silent — the row simply stops
being there. A menu costs one extra click on the rare action and removes the
mis-click on it entirely.

The row itself is not inert: clicking it opens the task. That is the safe,
frequent action, so it gets the whole row.

## Ordering is per project, and that is the whole feature

`rank` is a sparse float column in the project's own database. Dropping an item
between two rows is one UPDATE (their midpoint); when a gap gets too small to
halve, the list is respaced and the move still lands.

**A merged cross-project backlog will not offer drag-and-drop.** Two projects
have two independent orders, and interleaving them needs a rule for what happens
when you drag an item from A between two items of B. Every such rule is either
arbitrary (a hidden global tiebreaker nobody can see) or leaks (project A's
order changes when B is touched). The honest options were: a global rank stored
outside any project — which then disappears on a fresh clone and needs conflict
resolution between parallel edits — or no manual order at all.

So: manual order within one project, and a merged view (if it is ever built)
sorts by priority and groups by project, with dragging disabled and a reason
shown. Sacrificing the feature in the merged view is better than a feature whose
behaviour cannot be explained in one sentence.

A drop is sent to the server as **the two items it landed between**, never as an
index. An index means whatever the client had on screen; a filtered or stale
list turns it into a move nobody asked for.

Manual order also outranks priority in the listing. Dragging is an explicit
instruction, and a later priority edit must not silently undo it.

## The editor takes over the centre column

Managing work and talking to the agent are different activities, so the task
editor replaces the transcript rather than floating over it. The description
field is the reason: at drawer width (~250px) nobody writes a real description,
and a modal that covers the chat is a worse version of the same trade.

The transcript is hidden, not discarded — going back must not cost the
conversation.

Adding an item from the panel opens it in the editor immediately: a title alone
is rarely the task, and the moment you have just typed it is the moment you
still remember the rest.

## Still deliberately absent

- **Delete.** `rejected` is the end state. Removing the record removes the
  record of the decision.
- **Assignees, sprints, workflow rules, notifications.** This is a place to keep
  work items until there is a real tracker, and `agent todo export` exists so
  that move is a script, not a migration.
- **Live updates.** The panel refreshes on open, on `⟳`, and after every action.
  Two people editing one project's backlog simultaneously is not the case this
  is built for.
