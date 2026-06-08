import os
import modal

# 1. Define the Modal App
app = modal.App("xlstm-7b-custom-shots")

# 2. Define the container image with CUDA development tools, Python 3.11, and necessary packages
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "lm-eval>=0.4.0",
        "datasets",
        "accelerate",
        "xlstm",
        "mlstm_kernels",
        "triton"
    )
)

# 3. Mount a Hugging Face cache Volume to avoid re-downloading model weights (14+ GB) on every run
hf_cache_volume = modal.Volume.from_name(
    "hf-cache",
    create_if_missing=True
)

results_volume = modal.Volume.from_name(
    "eval-results",
    create_if_missing=True
)

# 4. Remote function to load configuration, model weights, and run evaluation
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=18000,  # 5-hour timeout to prevent cuts on slow runs
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/results": results_volume
    }
)
def evaluate_custom_shots(batch_size: str = "auto"):
    import json
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    import lm_eval

    # Step 4a. Load base config from Hugging Face Hub
    print("Downloading base config for NX-AI/xLSTM-7b...")
    config_file = hf_hub_download(repo_id="NX-AI/xLSTM-7b", filename="config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # Use default Triton optimized kernels
    print("Using default Triton optimized kernels...")
    config_dict.pop("model_type", None)
    xlstm_config = AutoConfig.for_model("xlstm", **config_dict)

    print("Loading xLSTM 7B...")
    # Load model in bfloat16 to fit in memory efficiently
    xlstm = AutoModelForCausalLM.from_pretrained(
        "NX-AI/xLSTM-7b", 
        config=xlstm_config, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Step 4d. Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
    tokenizer.pad_token = tokenizer.eos_token
    print("Model and Tokenizer loaded successfully")

    # Step 4e. Wrap in lm-eval's HFLM helper
    print("Wrapping model in HFLM wrapper...")
    eval_model = HFLM(pretrained=xlstm, tokenizer=tokenizer)

    # Parse batch size argument
    if batch_size.isdigit():
        eval_batch_size = int(batch_size)
    else:
        eval_batch_size = batch_size  # e.g., "auto"

    # Define the requested tasks and their few-shot values
    task_shots = {
        # "arc_challenge": 25,
        "hellaswag": 10,
        # "winogrande": 5,
        # "piqa": 5
    }

    all_results = {}
    
    # Run evaluation sequentially on target tasks
    for task_name, shots in task_shots.items():
        print(f"\n--- Starting evaluation on {task_name} with {shots}-shot (batch_size={eval_batch_size}) ---")
        
        try:
            res = lm_eval.simple_evaluate(
                model=eval_model,
                tasks=[task_name],
                num_fewshot=shots,
                batch_size=eval_batch_size
                # Note: HFLM wrapped model with device_map="auto" handles device placement natively.
                # Passing device="cuda:0" might conflict, so we omit it here since it's already on GPU.
            )
            
            # Extract metrics for this task
            if "results" in res and task_name in res["results"]:
                all_results[task_name] = res["results"][task_name]
            else:
                all_results[task_name] = res.get("results", {})
                
            print(f"Results for {task_name}: {json.dumps(all_results[task_name], indent=2)}")
        
        except Exception as e:
            print(f"Evaluation failed for {task_name}: {str(e)}")
            all_results[task_name] = {"error": str(e)}

    # Step 4g. Print the final metrics
    print("\n=================================")
    print("--- Final Evaluation Results ---")
    print(json.dumps(all_results, indent=2))
    print("=================================")

    # Commit HF cache changes back to the volume
    hf_cache_volume.commit()
    print("Hugging Face cache volume updated.")

    return all_results

# 5. Local Entrypoint
@app.local_entrypoint()
def main(batch_size: str = "auto"):
    import json
    import os
    
    print(f"Initiating remote custom few-shot evaluation job on Modal (batch_size={batch_size})...")
    outputs = evaluate_custom_shots.remote(batch_size=batch_size)
    
    print("Saving aggregated results to local machine's disk...")
    os.makedirs("lm_eval_results", exist_ok=True)
    local_path = os.path.join("lm_eval_results", "xlstm_custom_shots.json")
    
    with open(local_path, "w") as f:
        json.dump(outputs, f, indent=4)
        
    print(f"Saved aggregated results to {local_path}")
    print("Evaluation complete!")
