# clearml-tutorial

A learning project: train a Fashion-MNIST CNN with ClearML and deploy it to Triton
automatically, on a local Docker Compose stack (MinIO stands in for S3).

## Phases

| # | Phase | What | Status |
|---|-------|------|--------|
| 0 | Infrastructure | ClearML server, MinIO + buckets, ClearML agent (Docker Compose) | ✅ |
| 1 | Training | Preprocess task (raw → processed dataset) and train task, model registered in ClearML | ✅ |
| 2 | Manual deploy | ONNX export of preprocess / model / postprocess, generated `config.pbtxt`, Triton ensemble, tests | ✅ |
| 3 | ClearML pipeline | `model_id`-driven pipeline: convert, upload the model repository, Triton smoke test (Docker-mode agent) | ⏳ |
| 4 | DVC + PR | Push model files to a DVC remote on MinIO, open a deploy PR via the GitHub API | ⏳ |
| 5 | CI | Self-hosted GitHub runner, workflow that publishes models | ⏳ |
| 6 | GitOps rollout | k3d + ArgoCD (or a Compose deployer) serves the new model; rollback = `git revert` | ⏳ |
| 7 | Fully automatic | Publishing a model in ClearML triggers the pipeline | ⏳ |

## Layout

- `src/train.py`, `src/preprocess.py`: ClearML training tasks
- `src/convert.py`: ONNX export + Triton configs; `src/triton_ensemble.py`: ensemble config
- `src/test_triton.py`: manual Triton vs PyTorch check (needs a running Triton)
- `tests/`: pytest (ONNX vs PyTorch)
