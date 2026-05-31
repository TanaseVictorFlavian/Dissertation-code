import os
import modal

# 1. Define the Modal App
app = modal.App("xlstm-7b-squad-eval")

# 2. Define the container image with CUDA, Python 3.11, and necessary packages
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "datasets",
        "accelerate",
        "xlstm",
        "mlstm_kernels",
        "triton",
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

squad_volume = modal.Volume.from_name(
    "squad-eval-volume",
    create_if_missing=True
)

# 4. Remote SQuAD evaluation function
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=18000,  # 5-hour timeout to prevent cuts on slow runs
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/squad_results": squad_volume
    }
)
def run_squad_eval():
    import evaluate
    from transformers.models.xlstm.modeling_xlstm import xLSTMCache
    from dataclasses import dataclass
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    import pandas as pd
    import json
    import os

    # Load dataset
    print("Loading squad_curated.json...")
    dataset = pd.read_json("/root/squad_curated.json")
    print(f"Dataset loaded. Total rows: {len(dataset)}")

    print("Loading xLSTM 7B...")
    xlstm_config = AutoConfig.from_pretrained("NX-AI/xLSTM-7b")

    model = AutoModelForCausalLM.from_pretrained(
        "NX-AI/xLSTM-7b", config=xlstm_config, device_map="auto"
    )
    xlstm_tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
    xlstm_tokenizer.pad_token = xlstm_tokenizer.eos_token
    print("Model loaded successfully")
    xLSTMCache.is_compileable = False

    @dataclass
    class ContextState:
        cache: xLSTMCache
        context: str

    @torch.no_grad()
    def init_context(
        model,
        context_str,
        tokenizer,
    ) -> ContextState:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        print(f"Using device: {device}")
        context_str = f"Context:\n\n{context_str}"
        enc = tokenizer(context_str, return_tensors="pt")
        input_ids = enc.input_ids.to(device)

        print(f"INITIALIZING CONTEXT: {context_str[:100]}...")

        cache = xLSTMCache(
            config=model.config,
            max_batch_size=1,
            device=device,
            dtype=dtype
        )

        _ = model(input_ids=input_ids, cache_params=cache, use_cache=True)

        return ContextState(
            cache=cache,
            context=context_str[:100]
        )

    def clone_context(ctx: ContextState) -> ContextState:
        new_cache = xLSTMCache(
            config=ctx.cache.config,
            max_batch_size=ctx.cache.rnn_state[0][0].shape[0],
            device=ctx.cache.rnn_state[0][0].device,
            dtype=ctx.cache.dtype,
        )
        for i, (C, n, m) in ctx.cache.rnn_state.items():
            new_cache.rnn_state[i][0].copy_(C)
            new_cache.rnn_state[i][1].copy_(n)
            new_cache.rnn_state[i][2].copy_(m)

        new_cache.seqlen_offset = ctx.cache.seqlen_offset

        return ContextState(
            cache=new_cache,
            context=ctx.context
        )

    @torch.no_grad()
    def predict(
        model,
        tokenizer,
        question: str,
        per_call_ctx,
        max_new_tokens: int = 25,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        question = f"\n\nQuestion: {question}\nAnswer:"
        q_ids = tokenizer(
            question, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        out = model.generate(
            q_ids,
            cache_params=per_call_ctx.cache,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        new_tokens = out[0, q_ids.shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

    import time

    predictions = []
    references = []

    current_title = dataset.iloc[0]["title"]

    print("Starting prediction loop...")
    for i, row in dataset.iterrows():
        context = row["context"]
        question = row["question"]
        answer = row["answers"]["text"][0]
        title = row["title"]

        if i == 0 or title != current_title:
            print(f"Processing item index: {i}, Group Title: {title}")
            current_title = title
            # reset context
            ctx = init_context(model, context, xlstm_tokenizer)

        per_call_ctx = clone_context(ctx)

        start_time = time.perf_counter()
        pred = predict(
            model,
            xlstm_tokenizer,
            question,
            per_call_ctx,
        )

        duration = time.perf_counter() - start_time
        print(pred)

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

    print("\n--- SQuAD Evaluation Results & Timing ---")
    print(json.dumps(results, indent=2))

    # Save to the attached Modal Volume
    print("Writing files to Modal Volume...")
    os.makedirs("/root/squad_results", exist_ok=True)
    
    with open("/root/squad_results/squad_curated.json", "w") as f:
        dataset.to_json(f, orient="records", indent=4)
        
    with open("/root/squad_results/predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
    with open("/root/squad_results/references.json", "w") as f:
        json.dump(references, f, indent=4)
        
    with open("/root/squad_results/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Sync volume changes to cloud storage
    squad_volume.commit()
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
    print("Initiating remote SQuAD evaluation job on Modal...")
    outputs = run_squad_eval.remote()
    print("SQuAD Evaluation complete!")

    # Save the responses locally on the host machine's disk
    import json
    import os
    print("Saving responses to local machine's disk...")
    os.makedirs("squad_results", exist_ok=True)
    
    with open("squad_results/predictions.json", "w") as f:
        json.dump(outputs["predictions"], f, indent=4)
        
    with open("squad_results/references.json", "w") as f:
        json.dump(outputs["references"], f, indent=4)
        
    with open("squad_results/results.json", "w") as f:
        json.dump(outputs["results"], f, indent=4)

    print("Local save complete. Files written to local './squad_results/' directory.")

    # Calculate sum and print timing info at the end
    durations = [p["generation_time_sec"] for p in outputs["predictions"]]
    total_time = sum(durations)
    print(f"\n--- Run Performance Summary ---")
    print(f"Total questions answered: {len(durations)}")
    print(f"Total time taken to answer all questions: {total_time:.2f} seconds")
    print(f"Average generation time per question: {total_time / len(durations):.4f} seconds" if durations else "Average generation time per question: 0.0 seconds")

