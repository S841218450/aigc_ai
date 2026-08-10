from app.workflows.text_to_text.state import TextToTextState
from app.tools.text_generation.llm_tool import LLMTool


async def validate_prompt_node(state: TextToTextState) -> TextToTextState:
    """Validate and preprocess the text prompt"""
    # Add validation logic here
    return state


async def generate_text_node(state: TextToTextState) -> TextToTextState:
    """Generate text using LLM tool"""
    tool = LLMTool()
    result = await tool.generate(state.prompt, state.params)
    state.text = result["text"]
    state.metadata = result.get("metadata")
    return state


async def post_process_node(state: TextToTextState) -> TextToTextState:
    """Post-process the generated text"""
    # Add post-processing logic here
    return state