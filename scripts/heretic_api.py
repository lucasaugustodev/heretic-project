"""OpenAI-compatible API server for Heretic Mistral 7B."""
import sys, os, json, time, uuid
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'D:\heretic\python-libs')
os.environ['HF_HOME'] = 'D:\heretic\hf-cache'

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import torch
import uvicorn

print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "D:\heretic\output\mistral-7b-heretic-3x")
tokenizer = AutoTokenizer.from_pretrained("D:\heretic\output\mistral-7b-heretic-3x")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model loaded!")

app = FastAPI()

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "heretic-mistral-7b", "object": "model", "owned_by": "local"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    print(f"REQUEST: {json.dumps(body, default=str)[:500]}", flush=True)

    # Extract messages, handling both string and array content
    messages = []
    for m in body.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, list):
            # Cline sends content as array of objects
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", str(part)))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        messages.append({"role": m.get("role", "user"), "content": content})

    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)
    stream = body.get("stream", False)

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(inputs)

    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=inputs, attention_mask=attention_mask,
            max_new_tokens=max_tokens or 2048,
            temperature=max(temperature or 0.7, 0.01),
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        def stream_response():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            for text in streamer:
                chunk = {
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": "heretic-mistral-7b",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": "heretic-mistral-7b",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        with torch.no_grad():
            output = model.generate(
                inputs, attention_mask=attention_mask,
                max_new_tokens=max_tokens or 2048,
                temperature=max(temperature or 0.7, 0.01),
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "heretic-mistral-7b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": inputs.shape[1], "completion_tokens": output.shape[1] - inputs.shape[1], "total_tokens": output.shape[1]}
        }

if __name__ == "__main__":
    print("Starting API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
