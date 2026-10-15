from clearml import Task

task = Task.init(project_name="lab", task_name="hello")

task.execute_remotely(queue_name="default")

logger = task.get_logger()
for step in range(10):
    logger.report_scalar(title="loss", series="train", value=1.0 / (step + 1), iteration=step)

task.upload_artifact("note", artifact_object={"hello": "world"})
print("done")
