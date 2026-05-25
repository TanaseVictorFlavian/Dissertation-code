import os
import math
import modal

app = modal.App("xlstm-training-run")

image = modal.Image.debian_slim().pip_install("torch", "transformers").add_local_dir("./data", "/app/data")

checkpoint_volume = modal.Volume.from_name(
    "xlstm-checkpoints",
    create_if_missing=True
)

# 2. Remote Function Definition
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400, # 24-hour timeout for training loops
    volumes={"/app/checkpoints": checkpoint_volume}
)
def train_model():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    import random
    
    # --- Dataset Definition ---
    class RandomSampleDataset(Dataset):
        def __init__(self, continuous_tokens, total_steps, window_size=512):
            self.tokens = continuous_tokens
            self.total_steps = total_steps
            self.window_size = window_size
            self.max_start_idx = len(self.tokens) - self.window_size

        def __len__(self):
            return self.total_steps

        def __getitem__(self, idx):
            start = random.randint(0, self.max_start_idx)
            chunk = self.tokens[start : start + self.window_size].to(torch.long)
            return {
                "input_ids": chunk,
                "labels": chunk.clone() 
            }

    # --- Evaluation Function ---
    @torch.no_grad() 
    def evaluate_perplexity(model, test_tensor, context_length=512, batch_size=128):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        num_windows = len(test_tensor) // context_length
        truncated_test = test_tensor[:num_windows * context_length]
        eval_chunks = truncated_test.view(num_windows, context_length).to(torch.long)
        
        for i in range(0, num_windows, batch_size):
            batch_chunks = eval_chunks[i : i + batch_size].to(model.device)
            inputs = {"input_ids": batch_chunks, "labels": batch_chunks}
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            total_batches += 1
            
        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss)
        model.train() 
        return perplexity

    # --- Model Setup ---
    print("Loading xLSTM 7B Config...")
    import json
    from huggingface_hub import hf_hub_download
    
    config_file = hf_hub_download(repo_id="NX-AI/xLSTM-7b", filename="config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)
        
    config_dict["step_kernel"] = "native"
    config_dict["chunkwise_kernel"] = "chunkwise--native_autograd"
    config_dict["sequence_kernel"] = "native_sequence__native"
    config_dict.pop("model_type", None)
    
    xlstm_config = AutoConfig.for_model("xlstm", **config_dict)

    # Scaling down to a custom scratch model architecture based on your prompt
    xlstm_config.hidden_size = 768
    xlstm_config.embedding_dim = 768
    xlstm_config.num_heads = 4
    xlstm_config.num_hidden_layers = 12
    xlstm_config.num_blocks = 12
    xlstm_config.use_cache = False

    with torch.device("cuda"):
        xlstm = AutoModelForCausalLM.from_config(config=xlstm_config)

    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b", config=xlstm_config)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Total parameters: {xlstm.num_parameters():,}")

    # --- Data Loading (Path updated to Modal Mount) ---
    print("Loading tensors from mounted data directory...")
    train_tensor = torch.load("/app/data/train_tokens_list.pt")
    val_tensor = torch.load("/app/data/validation_tokens_list.pt")
    test_tensor = torch.load("/app/data/test_tokens_list.pt")
    print(f"Loaded {len(train_tensor):,} train tokens.")

    # --- Hyperparameters ---
    CONTEXT_LENGTH = 512
    TRAINING_STEPS = 100
    WARMUP_STEPS = 10
    MAX_LR = 1e-3
    EFFECTIVE_BATCH_SIZE = 256
    MICRO_BATCH_SIZE = 32
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
    TOTAL_FORWARD_PASSES = TRAINING_STEPS * ACCUMULATION_STEPS
    TOTAL_SAMPLES = TOTAL_FORWARD_PASSES * MICRO_BATCH_SIZE

    dataset = RandomSampleDataset(train_tensor, TOTAL_SAMPLES, CONTEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(xlstm.parameters(), lr=MAX_LR)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TRAINING_STEPS
    )

    # --- Training Loop ---
    xlstm.train()
    optimizer.zero_grad()
    global_step = 0 

    print("\n--- Step 0 (Baseline) Evaluation ---")
    baseline_ppl = evaluate_perplexity(xlstm, test_tensor, CONTEXT_LENGTH)
    print(f"Untrained Validation Perplexity: {baseline_ppl:.2f}")

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(xlstm.device) for k, v in batch.items()}
        
        outputs = xlstm(**batch)
        loss = outputs.loss / ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % ACCUMULATION_STEPS == 0:
            # torch.nn.utils.clip_grad_norm_(xlstm.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # --- Checkpointing ---
            if global_step % 100 == 0:
                print(f"\n--- Step {global_step} Evaluation ---")
                ppl = evaluate_perplexity(xlstm, test_tensor, CONTEXT_LENGTH)
                print(f"Validation Perplexity: {ppl:.2f}")
                
                # Save to the attached Modal Volume
                checkpoint_dir = f"/app/checkpoints/step_{global_step}"
                print(f"Saving checkpoint to {checkpoint_dir}...")
                
                xlstm.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # CRITICAL: Commit the volume so changes are pushed to cloud storage
                checkpoint_volume.commit() 
                print("Checkpoint synced to cloud storage.")
                print("--------------------------------\n")

# 3. Local Entrypoint
@app.local_entrypoint()
def main():
    print("Initiating remote training job on Modal...")
    train_model.remote()
    print("Training complete!")