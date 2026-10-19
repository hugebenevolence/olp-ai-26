# Learning notebook design

[Notebook pairs](../notebooks/learning/README.md) · [Learning scorecard](LEARNING_SCORECARD.md)

These lessons use the guided exercise pattern discussed in the DeepLearning.AI course research:
explain a concept, implement a bounded function, test it, interpret its result and reuse it.
The course's [DLS mentor guidance](https://community.deeplearning.ai/t/how-are-we-expected-to-approach-the-coding-assignments-in-course-1/877155)
and [official public assignment example](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/assignment/C2W2_Assignment.ipynb)
illustrate that pattern. Our task explanations, exercises and checks are original competition
preparation material; these are not DeepLearning.AI assignments or leaderboard reproductions.

## Study one component

1. Read the task, prediction unit and official metric before writing code.
2. Read an exercise's arguments, output shape and invariants.
3. Attempt the marked body. Use the conceptual hint first, then the implementation hint.
4. Run the nearby small check. A failing assertion is feedback; inspect it before revealing code.
5. Explain the result in the stage's **Stop and explain** prompt.
6. Reuse the component in the actual fit/validation/delivery workflow.
7. Close the notebook and perform its final transfer/reconstruction task.

Each baseline has eight exercises. Advanced notebooks supply their implementations and focus on
prediction, inspection, checks and controlled comparisons. A deterministic toy score can have
an exact expected value; real model scores depend on data, split, runtime and resource policy.

Every baseline code cell retains a closed full reference directly afterward, including supplied
and check cells. This preserves convenient comparison without forcing you to open an answer.
The hint and reference HTML details start closed. Renderer support determines their appearance;
use JupyterLab when another viewer expands or strips details.

## Supplied code and reconstruction

Storage, safe acquisition, precise graders and integrity checks stay inline to keep each notebook
self-contained. Their definitions have the `provided` tag and standard Jupyter source-fold metadata.
They must still execute. Folding is a presentation convenience, not a security boundary.

The first cell is a short import cell. Runtime helpers are separated; serialization, prediction,
training and delivery helpers are introduced in their relevant stages. Longer supplied class or
download definitions may remain a single code cell so the notebook stays valid Python.

Reconstruct the data contract, split, preprocessing, model/loss, update, scorer and ID-aligned
delivery first. Robust storage and reporting are supplied during guided study. The final drill
starts from an empty notebook; success with supplied helpers alone is not contest readiness.
For the 2k assistant, provide one signature, its shapes and one failing example at a time.

## Maintain a lesson

- Edit the adjacent `*_template.py`, then run `uv run olp-ai build-notebooks`.
- Use a personal notebook copy for attempts; regeneration replaces maintained `.ipynb` artifacts.
- Keep all runtime logic inline. No project imports, companion scripts or downloaded code.
- Keep eleven competition stages and optional acquisition, but choose cell boundaries by learning
  purpose rather than a fixed code-cell count.
- Tag code cells as `provided`, `configuration`, `exercise`, `worked` or `check`, with their
  `stage-N` tag. Keep configuration and the optional download switch visible.
- Mark only the missing algorithm body with `# >>> exercise:` and `# <<< exercise`.
- Keep the function contract outside the hidden body. Baseline function exercises should remain
  small enough to inspect and debug without a notebook-sized assistant prompt.
- Use independent tiny inputs to check behavior: new categories, empty text, alternate label
  counts, reversed rankings, empty masks, non-square geometry, unchanged inferior checkpoints,
  similar/zero-feature gallery items and shuffled literal IDs.
- Run shape/metric checks before full training. Do not hard-code expected real-dataset scores.
- Preserve pretrained/local-only defaults, leakage boundaries and the exact submission contract.

Tests compile individual cells and execute completed baseline answers plus advanced code with
network and repository imports blocked. Regression checks also inject plausible wrong functions
to verify that the embedded feedback rejects them. Semantic stage grouping in the test harness
supports fixture configuration; it is not a runtime dependency of any copied notebook.
