import os


#路径工具
def get_project_path() -> str:
    """"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_path = os.path.dirname(current_dir)
    return project_path

def get_absolute_path(relative:str) -> str:
    """"""
    project_path = get_project_path()
    return os.path.join(project_path, relative)

if __name__ == '__main__':
    print(get_project_path())
    print(get_absolute_path('doc/prompts'))