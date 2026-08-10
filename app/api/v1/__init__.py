from fastapi import APIRouter
from app.api.v1.endpoints import text_to_image, text_to_video, text_to_text, knowledge_base, image_to_image, prompt_generate

router = APIRouter()

router.include_router(text_to_image.router, prefix="/text-to-image", tags=["Text to Image"])
router.include_router(text_to_video.router, prefix="/text-to-video", tags=["Text to Video"])
router.include_router(text_to_text.router, prefix="/text-to-text", tags=["Text to Text"])
router.include_router(knowledge_base.router, prefix="/knowledge-base", tags=["Knowledge Base"])
router.include_router(image_to_image.router, prefix="/image-to-image", tags=["Image to Image"])
router.include_router(prompt_generate.router, prefix="/prompt-generate", tags=["Prompt Generate"])