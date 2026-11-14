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

## Next (Phase 3)

- [x] `src/deploy.py`: `model_id` → Triton model repository artifact (works locally)
- [x] ML image: Dockerfile with locked deps from `pyproject.toml` / `uv.lock` (fixes missing `tritonclient[grpc]` on the agent)
- [x] Agent in Docker mode (socket mount, compose network, host mount)
- [x] Deploy task runs on the agent inside `clearml-tutorial:v1` (image set in the UI)
- [ ] `setup_task` sets the image via `task.set_base_docker(...)` instead of the UI
- [ ] Triton check step: fetch `model_repository`, start `tritonserver`, run the check, fail on error
- [ ] Wire both steps into a `PipelineDecorator` pipeline
- Later: local container registry (stand-in for ECR)

## Layout

- `src/preprocess.py`, `src/train.py`: ClearML data and training tasks
- `src/deploy.py`: ClearML deploy task (`model_id` → `model_repository` artifact)
- `src/convert.py`: `ConvertSpec`, ONNX export + Triton configs; `src/triton_ensemble.py`: ensemble config
- `tests/test_convert.py`: ONNX vs PyTorch; `tests/test_ensemble_triton.py`: Triton vs PyTorch (skips without Triton)

## Running locally

- Always `export CLEARML_CONFIG_FILE="$PWD/clearml.conf"` (otherwise `~/clearml.conf` is used)
- Build the task image after dependency changes: `docker build -t clearml-tutorial:v<N> .` (bump the tag; the agent uses the local image, no registry)
- `/etc/hosts` needs `127.0.0.1 minio`; `clearml.conf` uses `minio:9000`, same as the agent
