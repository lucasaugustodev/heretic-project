import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'D:\heretic\python-libs')
os.environ['HF_HOME'] = 'D:\heretic\hf-cache'

print("Loading model + 3x LoRA adapter...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

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
print("Model loaded! Type your messages (type 'quit' to exit)\n")

while True:
    try:
        user_input = input("You: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.strip().lower() in ('quit', 'exit', 'sair'):
        break
    
    messages = [{"role": "user", "content": user_input}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(inputs)
    
    with torch.no_grad():
        output = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=512, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\nHeretic: {response}\n")

print("Bye!")
