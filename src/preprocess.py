from utils import setup_task, parse_arguments

def main(project_name, task_name, config_path, queue_name):
    task, config = setup_task(project_name, task_name, config_path, queue_name)
    print(config)

if __name__ == '__main__':
    main(*parse_arguments())