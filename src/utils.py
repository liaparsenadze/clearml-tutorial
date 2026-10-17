import argparse
import json

from clearml import Task
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project-name", type=str, required=False, default=None)
    parser.add_argument("-t", "--task-name", type=str, required=False, default=None, help="runner task name")
    parser.add_argument("-c", "--config-path", type=Path, required=False)
    parser.add_argument("-q", "--queue-name", type=str, required=False, default=None)
    args = parser.parse_args()
    return args.project_name, args.task_name, args.config_path, args.queue_name


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return json.load(f)


def setup_task(
    project_name: str | None,
    task_name: str | None,
    config_path: Path | None,
    queue_name: str | None,
    task_type=Task.TaskTypes.training
) -> Task:
    task = Task.init(project_name=project_name, task_name=task_name, task_type=task_type)
    config = task.connect_configuration(load_config(config_path), name='config') if config_path else None
    if queue_name:
        task.execute_remotely(queue_name=queue_name, exit_process=True)
    return task, config