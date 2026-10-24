"""Loads Qwen2.5-VL-7B-Instruct once (from local HF cache) and exposes a simple
generate() call used both for index-time figure captioning and query-time
grounded answer generation. No qwen-vl-utils dependency: images are passed
directly to the processor."""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
    return _model, _processor


def generate(text_prompt, images=None, max_new_tokens=400):
    """images: list of PIL.Image or None for text-only prompts."""
    model, processor = get_model()
    content = []
    if images:
        content.extend({"type": "image"} for _ in images)
    content.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content}]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text], images=images if images else None, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
