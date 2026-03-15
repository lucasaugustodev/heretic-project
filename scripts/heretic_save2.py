import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, 'D:\heretic\python-libs')
os.environ['HF_HOME'] = 'D:\heretic\hf-cache'
os.environ['TMPDIR'] = 'D:\heretic\tmp'
os.environ['TEMP'] = 'D:\heretic\tmp'
os.environ['TMP'] = 'D:\heretic\tmp'

import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock

checkpoint = "C:/Users/PC/checkpoints/mistralai--Mistral-7B-Instruct-v0--3.jsonl"
lock_obj = JournalFileOpenLock(checkpoint)
backend = JournalFileStorage(checkpoint, lock_obj=lock_obj)
storage = JournalStorage(backend)
studies = storage.get_all_studies()
study = optuna.load_study(study_name=studies[0].study_name, storage=storage)
trials = [t for t in study.trials if t.state.name == 'COMPLETE']
best = min(trials, key=lambda t: t.values[0])
print(f"Best trial #{best.number}")

direction_index = best.user_attrs['direction_index']
parameters = best.user_attrs['parameters']

# Multiply weights by 3x for more aggressive ablation
MULTIPLIER = 3.0
for k, v in parameters.items():
    v['max_weight'] *= MULTIPLIER
    v['min_weight'] *= MULTIPLIER
print(f"Ablation weights multiplied by {MULTIPLIER}x")
print(f"  attn.o_proj: max={parameters['attn.o_proj']['max_weight']:.2f} min={parameters['attn.o_proj']['min_weight']:.2f}")
print(f"  mlp.down_proj: max={parameters['mlp.down_proj']['max_weight']:.2f} min={parameters['mlp.down_proj']['min_weight']:.2f}")

print("Loading model...")
from heretic.model import Model, AbliterationParameters
from heretic.config import Settings
from heretic.utils import load_prompts
import torch, torch.nn.functional as F, gc

settings_json = study.user_attrs['settings']
settings = Settings.model_validate_json(settings_json)
settings.batch_size = 32
model = Model(settings)
print("Model loaded!")

print("Loading prompts...")
good_prompts = load_prompts(settings, settings.good_prompts)
bad_prompts = load_prompts(settings, settings.bad_prompts)
print(f"  {len(good_prompts)} good, {len(bad_prompts)} bad")

print("Getting good residuals...")
good_residuals = model.get_residuals_batched(good_prompts)
print("  Done")
print("Getting bad residuals...")
bad_residuals = model.get_residuals_batched(bad_prompts)
print("  Done")

good_means = good_residuals.mean(dim=0)
bad_means = bad_residuals.mean(dim=0)
refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

del good_residuals, bad_residuals, good_prompts, bad_prompts
gc.collect()
torch.cuda.empty_cache()

print(f"Applying 3x ablation...")
model.reset_model()
model.abliterate(
    refusal_directions,
    direction_index,
    {k: AbliterationParameters(**v) for k, v in parameters.items()},
)
print("Abliteration applied!")

save_dir = "D:\heretic\output\mistral-7b-heretic-3x"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving LoRA adapter to {save_dir}...")
model.model.save_pretrained(save_dir)
model.tokenizer.save_pretrained(save_dir)
print(f"DONE! Saved to {save_dir}")
