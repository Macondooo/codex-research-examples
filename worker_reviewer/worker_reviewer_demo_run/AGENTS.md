# Project instructions

Read `workflow.md` before starting. Use `worker.md` for implementation and `reviewer.md` for the independent check.

Act as the Main Agent. Give the implementation to one Worker, wait for its handoff, then ask a different Reviewer to inspect the result without editing it.

If the Reviewer finds a real problem, send that specific issue back to the Worker and review the updated result again. Otherwise, summarize the result and finish.

Keep the task small and do not claim that a run or review succeeded without seeing the evidence. These role names are only used for this demo, not as official Codex modes.

