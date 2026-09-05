# MNIST Live Demo

This Codex demo trains and evaluates a handwritten-digit classifier from the supplied MNIST CSV files, then writes metrics, predictions, figures, and a short report.

## Run it with Codex

Open this folder as the Codex workspace and send:

> Implement and run this MNIST experiment. Verify the outputs and summarize the results.

Codex loads `AGENTS.md` when it opens the project. `TASK.md` defines the experiment, and `REQUEST.md` adds the presentation and reporting requirements.

## Starting layout

```text
.
├── AGENTS.md
├── README.md
├── REQUEST.md
├── TASK.md
├── data/
│   ├── README.md
│   └── raw/
│       ├── mnist_train.csv
│       └── mnist_test.csv
├── expected/
│   └── README.md
├── .agents/skills/mnist-live-demo/SKILL.md
└── .codex/agents/work-review.toml
```

Codex creates `src/`, `tests/`, `outputs/`, the dependency file, and `REPORT.md` during the demo.
