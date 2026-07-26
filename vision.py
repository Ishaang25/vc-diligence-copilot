"""
Async vision analysis with RAG embedding generation.
"""
from __future__ import annotations
import asyncio
import logging
from openai import AsyncOpenAI
from PIL import Image
import json

from cache import cache_key, disk_cache_get, disk_cache_set
from models import SlideAnalysis, SlideType
from parser import perform_tesseract_ocr
from prompts import SLIDE_ANALYSIS_PROMPT, SLIDE_CLASSIFIER_PROMPT
from utils import call_openai_json_async, get_embeddings, image_to_base64, prepare_image_for_api

logger = logging.getLogger("dd_copilot.vision")
EXTRACTION_MODEL = "gpt-4o-mini"
TEXT_ONLY_THRESHOLD = 1000

async def classify_slide_type(client: AsyncOpenAI, text: str) -> SlideType:
    if not text or len(text) < 20:
        return SlideType.OTHER
    try:
        logger.warning(
            "Slide %d extracted text:\n%s",
            slide_number,
            extracted_text[:3000]
        )
        data = await call_openai_json_async(
            client, [{"role": "user", "content": SLIDE_CLASSIFIER_PROMPT + f"\n\nTEXT:\n{text}"}],
            model=EXTRACTION_MODEL, temperature=0.0, max_tokens=50, agent_name="Slide Classifier"
        )
        return SlideType(data.get("slide_type", "OTHER").lower())
    except Exception:
        return SlideType.OTHER

async def analyze_slide(
    client: AsyncOpenAI,
    image: Image.Image,
    slide_number: int,
    file_hash: str,
    extracted_text: str = "",
    is_scanned: bool = False,
) -> SlideAnalysis:
    cache_k = cache_key(file_hash, "slide_v4", slide_number, extracted_text[:50])
    cached = disk_cache_get(cache_k)
    if cached is not None:
        return SlideAnalysis.model_validate(cached)

    use_vision = len(extracted_text) < TEXT_ONLY_THRESHOLD or is_scanned
    
    if use_vision and is_scanned:
        ocr_text = perform_tesseract_ocr(image)
        if len(ocr_text.strip()) > TEXT_ONLY_THRESHOLD:
            extracted_text = ocr_text
            use_vision = False

    prompt = SLIDE_ANALYSIS_PROMPT
    if use_vision:
        image = prepare_image_for_api(image, max_size=1536)
        image_b64 = image_to_base64(image, fmt="JPEG", quality=75)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}}
        ]}]
    else:
        messages = [{
            "role": "user",
            "content": f"""{SLIDE_ANALYSIS_PROMPT}

    EXTRACTED TEXT:

    {extracted_text}
    
    Analyze this slide using the extracted text above.
    Do not say that information is unavailable if it is present in the extracted text.
    """
        }]
    try:
        data = await call_openai_json_async(
            client, messages, model=EXTRACTION_MODEL, temperature=0.2, 
            max_tokens=1500, agent_name=f"Slide {slide_number}"
        )
        data["slide_number"] = slide_number
        data["slide_type"] = await classify_slide_type(client, extracted_text or data.get("summary", ""))
        logger.warning("RAW SLIDE JSON %d: %s", slide_number, json.dumps(data))
        result = SlideAnalysis.model_validate(data)
        print("\n" + "="*80)
        print(f"SLIDE {slide_number}")
        print("="*80)
        print("SUMMARY:")
        print(result.summary)
        
        print("\nNUMBERS:")
        print(result.numbers)
        
        print("\nCLAIMS:")
        print(result.claims)
        
        print("\nRISKS:")
        print(result.risks)
        
        print("="*80)
    except Exception as exc:
        logger.error("Slide %d failed: %s", slide_number, exc)
        result = SlideAnalysis(slide_number=slide_number, summary=f"Failed: {exc}")

    disk_cache_set(cache_k, result.model_dump())
    return result

async def analyze_all_slides(
    client: AsyncOpenAI,
    images: list[Image.Image],
    file_hash: str,
    pages: list,
    is_scanned: bool,
    progress_callback=None,
    concurrency_limit: int = 5
) -> list[SlideAnalysis]:
    """Analyze slides concurrently with a semaphore to prevent rate limiting."""
    semaphore = asyncio.Semaphore(concurrency_limit)
    total = len(images)
    
    async def process_slide(i: int):
        async with semaphore:
            page = pages[i]
            if page.is_junk:
                res = SlideAnalysis(slide_number=i+1, is_skipped=True, summary="Skipped: Junk slide")
                if progress_callback: progress_callback(i+1, total, res)
                return res
            
            res = await analyze_slide(client, images[i], i+1, file_hash, page.text, is_scanned)
            if progress_callback: progress_callback(i+1, total, res)
            return res

    tasks = [process_slide(i) for i in range(total)]
    return await asyncio.gather(*tasks)

async def embed_slides(client: AsyncOpenAI, analyses: list[SlideAnalysis]) -> None:
    """Generate embeddings for all slide summaries for RAG."""
    valid_slides = [a for a in analyses if not a.is_skipped]
    texts = [a.summary for a in valid_slides]
    
    if not texts:
        return
        
    embeddings = await get_embeddings(client, texts)
    for slide, emb in zip(valid_slides, embeddings):
        slide.embedding = emb
