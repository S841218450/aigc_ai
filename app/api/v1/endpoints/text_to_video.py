from fastapi import APIRouter, HTTPException
from app.models.schemas.text_to_video import TextToVideoRequest, TextToVideoResponse
from app.workflows.text_to_video.graph import TextToVideoGraph

router = APIRouter()


@router.post("/generate", response_model=TextToVideoResponse)
async def generate_video(request: TextToVideoRequest):
    """Generate video from text prompt"""
    try:
        graph = TextToVideoGraph()
        result = await graph.run(request.prompt, request.params)
        return TextToVideoResponse(video_url=result["video_url"], metadata=result.get("metadata"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))