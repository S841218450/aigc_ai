import os
from typing import Literal

# 提示词基础目录
PROMPTS_BASE_DIR = os.path.join(os.path.dirname(__file__))

workType = Literal['text_to_image','image_to_image', 'text_to_text', 'text_to_video','common','knowledge']


def get_prompt(work_type: workType, prompt_name: str) -> str:
    """
    获取指定工作流类型的提示词模板
    
    Args:
        work_type: 工作流类型 (text_to_image, text_to_text, text_to_video)
        prompt_name: 提示词名称 (不含.txt后缀)
    
    Returns:
        提示词模板内容
    """
    # 构建提示词文件路径
    prompt_dir = os.path.join(PROMPTS_BASE_DIR, work_type)
    prompt_file = os.path.join(prompt_dir, f"{prompt_name}.txt")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
    except Exception as e:
        raise Exception(f"读取提示词文件出错: {e}")


if __name__ == "__main__":
    # 测试加载提示词
    print(get_prompt('text_to_image', 'descScopeJudge'))
    print("---")
    print(get_prompt('text_to_image', 'decisionRouter'))
    print("---")
    print(get_prompt('text_to_image', 'optionGenerate'))
    print("---")
    print(get_prompt('text_to_image', 'imageSummaryJudge'))