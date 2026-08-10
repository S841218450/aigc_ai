from langgraph.graph import StateGraph, END
from app.workflows.text_to_text.state import TextToTextState
from app.workflows.text_to_text.nodes import (
    validate_prompt_node,
    generate_text_node,
    post_process_node
)


class TextToTextGraph:
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(TextToTextState)
        
        # Add nodes
        workflow.add_node("validate_prompt", validate_prompt_node)
        workflow.add_node("generate_text", generate_text_node)
        workflow.add_node("post_process", post_process_node)
        
        # Set entry point
        workflow.set_entry_point("validate_prompt")
        
        # Add edges
        workflow.add_edge("validate_prompt", "generate_text")
        workflow.add_edge("generate_text", "post_process")
        workflow.add_edge("post_process", END)
        
        return workflow.compile()
    
    async def run(self, prompt: str, params: dict = None) -> dict:
        initial_state = TextToTextState(
            prompt=prompt,
            params=params or {},
            text=None,
            metadata=None
        )
        result = await self.graph.ainvoke(initial_state)
        return result