import os
import modal

# 1. Define the Modal App
app = modal.App("mistral-7b-squad-eval")

# 2. Define the container image with CUDA, Python 3.11, and necessary packages
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "datasets",
        "accelerate",
        "pandas",
        "evaluate",
        "scikit-learn"
    )
    .add_local_file("squad_curated.json", "/root/squad_curated.json")
)

# 3. Mount cache Volumes to persist downloaded model and SQuAD metrics/results
hf_cache_volume = modal.Volume.from_name(
    "hf-cache",
    create_if_missing=True
)

squad_volume_mistral = modal.Volume.from_name(
    "squad-eval-volume-mistral",
    create_if_missing=True
)

# Configure secrets dynamically: use local HF_TOKEN environment variable if set,
# otherwise fallback to the Modal-managed "huggingface-secret".
secrets_list = []
if os.environ.get("HF_TOKEN"):
    secrets_list.append(modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN")}))
else:
    secrets_list.append(modal.Secret.from_name("huggingface-secret"))

# 4. Remote SQuAD evaluation function for Mistral
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=18000,  # 5-hour timeout
    secrets=secrets_list,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/squad_results_mistral": squad_volume_mistral
    }
)
def run_squad_eval_mistral():
    import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import pandas as pd
    import json
    import os
    import time

    # Load dataset
    print("Loading squad_curated.json...")
    dataset = pd.read_json("/root/squad_curated.json")
    print(f"Dataset loaded. Total rows: {len(dataset)}")

    print("Loading Mistral 7B...")
    model_id = "mistralai/Mistral-7B-v0.1"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded successfully")

    @torch.no_grad()
    def predict(
        model,
        tokenizer,
        context: str,
        question: str,
        max_new_tokens: int = 25,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        prompt = f"Context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        q_ids = tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(model.device)

        out = model.generate(
            q_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        new_tokens = out[0, q_ids.shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

    predictions = []
    references = []

    print("Starting prediction loop (Mistral 7B)...")
    for i, row in dataset.iterrows():
        context = row["context"]
        question = row["question"]
        title = row["title"]

        print(f"Processing item index: {i}, Group Title: {title}")

        start_time = time.perf_counter()
        pred = predict(
            model,
            tokenizer,
            context,
            question,
        )
        duration = time.perf_counter() - start_time

        # store prediction
        predictions.append({
            "id": str(i),
            "prediction_text": pred,
            "generation_time_sec": duration
        })

        # store reference
        references.append({
            "id": str(i),
            "answers": row["answers"]
        })

    print("Predictions complete. Loading SQuAD metric...")
    metric = evaluate.load("squad")

    print("Computing metrics...")
    results = metric.compute(
        predictions=[{"id": p["id"], "prediction_text": p["prediction_text"]} for p in predictions],
        references=references
    )

    # Compute and add time performance metrics
    durations = [p["generation_time_sec"] for p in predictions]
    total_gen_time = sum(durations)
    avg_gen_time = total_gen_time / len(durations) if len(durations) > 0 else 0.0

    results["total_generation_time_sec"] = total_gen_time
    results["avg_generation_time_sec"] = avg_gen_time

    print("\n--- SQuAD Evaluation Results & Timing (Mistral 7B) ---")
    print(json.dumps(results, indent=2))

    # Save to the attached Modal Volume
    print("Writing files to Modal Volume...")
    os.makedirs("/root/squad_results_mistral", exist_ok=True)
    
    with open("/root/squad_results_mistral/squad_curated.json", "w") as f:
        dataset.to_json(f, orient="records", indent=4)
        
    with open("/root/squad_results_mistral/predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
    with open("/root/squad_results_mistral/references.json", "w") as f:
        json.dump(references, f, indent=4)
        
    with open("/root/squad_results_mistral/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Sync volume changes to cloud storage
    squad_volume_mistral.commit()
    hf_cache_volume.commit()
    print("Modal volumes updated successfully.")

    return {
        "results": results,
        "predictions": predictions,
        "references": references
    }

# 5. Local Entrypoint
@app.local_entrypoint()
def main():
    print("Initiating remote SQuAD evaluation job on Modal (Mistral 7B)...")
    outputs = run_squad_eval_mistral.remote()
    print("SQuAD Evaluation (Mistral 7B) complete!")

    # Save the responses locally on the host machine's disk
    import json
    import os
    print("Saving responses to local machine's disk...")
    os.makedirs("squad_results_mistral", exist_ok=True)
    
    with open("squad_results_mistral/predictions.json", "w") as f:
        json.dump(outputs["predictions"], f, indent=4)
        
    with open("squad_results_mistral/references.json", "w") as f:
        json.dump(outputs["references"], f, indent=4)
        
    with open("squad_results_mistral/results.json", "w") as f:
        json.dump(outputs["results"], f, indent=4)

    print("Local save complete. Files written to local './squad_results_mistral/' directory.")

    # Calculate sum and print timing info at the end
    durations = [p["generation_time_sec"] for p in outputs["predictions"]]
    total_time = sum(durations)
    print(f"\n--- Run Performance Summary (Mistral 7B) ---")
    print(f"Total questions answered: {len(durations)}")
    print(f"Total time taken to answer all questions: {total_time:.2f} seconds")
    print(f"Average generation time per question: {total_time / len(durations):.4f} seconds" if durations else "Average generation time per question: 0.0 seconds")
