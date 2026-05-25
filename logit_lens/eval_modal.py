import os
import modal

# 1. Define the Modal App
app = modal.App("xlstm-7b-evaluation")

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

# 4. Remote function to load configuration, model weights, and run evaluation
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=18000,  # 5-hour timeout to prevent cuts on slow runs
    volumes={"/root/.cache/huggingface": hf_cache_volume}
)
def evaluate_model(batch_size: str = "auto", tasks: str = "lambada_openai", kernel: str = "triton"):
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

    # Step 4b. Configure kernels based on parameter
    if kernel == "native":
        print("Configuring native kernels (as in wikitext-103.py)...")
        config_dict["step_kernel"] = "native"
        config_dict["chunkwise_kernel"] = "chunkwise--native_autograd"
        config_dict["sequence_kernel"] = "native_sequence__native"
    else:
        print("Using default Triton optimized kernels...")
        
    config_dict.pop("model_type", None)
    xlstm_config = AutoConfig.for_model("xlstm", **config_dict)

    # Step 4c. Load full pre-trained 7B weights with the custom config
    print("Loading pre-trained xLSTM-7b weights in bfloat16...")
    xlstm = AutoModelForCausalLM.from_pretrained(
        "NX-AI/xLSTM-7b",
        config=xlstm_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Step 4d. Load the matching tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

    # Step 4e. Wrap in lm-eval's HFLM helper
    print("Wrapping model in HFLM wrapper...")
    eval_model = HFLM(pretrained=xlstm, tokenizer=tokenizer)

    # Parse batch size argument
    if batch_size.isdigit():
        eval_batch_size = int(batch_size)
    else:
        eval_batch_size = batch_size  # e.g., "auto"

    # Parse task list argument
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    # Step 4f. Run evaluation on target tasks
    print(f"Starting evaluation on tasks {task_list} (batch_size={eval_batch_size}, kernel={kernel})...")
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=task_list,
        batch_size=eval_batch_size,
        device="cuda:0"
    )

    # Step 4g. Print the final metrics
    print("\n--- Evaluation Results ---")
    print(json.dumps(results["results"], indent=2))

    # Commit HF cache changes back to the volume
    hf_cache_volume.commit()
    print("Hugging Face cache volume updated.")

    return results["results"]

# 5. Local Entrypoint
@app.local_entrypoint()
def main(batch_size: str = "auto", tasks: str = "lambada_openai", kernel: str = "triton"):
    print(f"Initiating remote evaluation job on Modal (batch_size={batch_size}, tasks={tasks}, kernel={kernel})...")
    evaluate_model.remote(batch_size=batch_size, tasks=tasks, kernel=kernel)
    print("Evaluation complete!")
