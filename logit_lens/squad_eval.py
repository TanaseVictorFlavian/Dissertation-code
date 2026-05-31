import evaluate
from transformers.models.xlstm.modeling_xlstm import xLSTMCache
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import pandas as pd

dataset = pd.read_json("squad_curated.json")


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading xLSTM 7B...")
xlstm_config = AutoConfig.from_pretrained("NX-AI/xLSTM-7b")
xlstm_config.step_kernel = "native"
xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
xlstm_config.sequence_kernel = "native_sequence__native"

model = AutoModelForCausalLM.from_pretrained(
    "NX-AI/xLSTM-7b", config=xlstm_config, device_map="auto")
xlstm_tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
xlstm_tokenizer.pad_token = xlstm_tokenizer.eos_token
print("Model loaded succesfully")
xLSTMCache.is_compileable=False

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
    max_new_tokens: int = 100,
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


predictions = []
references = []

current_title = dataset.iloc[0]["title"]

for i, row in dataset.iterrows():
    context = row["context"]
    question = row["question"]
    answer = row["answers"]["text"][0]
    title = row["title"]

    if i == 0 or title != current_title:
        print(i)
        current_title = title
        # reset context
        ctx = init_context(model, context, xlstm_tokenizer)

    per_call_ctx = clone_context(ctx)

    pred = predict(
        model,
        xlstm_tokenizer,
        question,
        per_call_ctx,
    )

    # store prediction
    predictions.append({
        "id": str(i),
        "prediction_text": pred
    })

    # store reference
    references.append({
        "id": str(i),
        "answers": row["answers"]
    })


metric = evaluate.load("squad")

results = metric.compute(
    predictions=predictions,
    references=references
)

print(results)
