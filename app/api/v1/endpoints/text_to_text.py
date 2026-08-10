from fastapi import APIRouter, HTTPException
from app.models.schemas.text_to_text import TextToTextRequest, TextToTextResponse
from app.workflows.text_to_text.graph import TextToTextGraph

router = APIRouter()


@router.post("/generate", response_model=TextToTextResponse)
async def generate_text(request: TextToTextRequest):
    """Generate text from text prompt"""
    try:
        graph = TextToTextGraph()
        result = await graph.run(request.prompt, request.params)
        return TextToTextResponse(text=result["text"], metadata=result.get("metadata"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))