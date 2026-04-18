from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

xlstm_config = AutoConfig.from_pretrained("NX-AI/xLSTM-7b")
xlstm_config.step_kernel = "native"
xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
xlstm_config.sequence_kernel = "native_sequence__native"

xlstm = AutoModelForCausalLM.from_pretrained("NX-AI/xLSTM-7b",
                                             config=xlstm_config, device_map="auto")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

# Your prompt
prompt = "Explain quantum computing in simple terms."

# Tokenize and send to the same device as the model
inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(xlstm.device)

# Get the BOS token ID from the tokenizer
bos_id = tokenizer.bos_token_id

# Prepend BOS
bos_tensor = torch.tensor([[bos_id]], device=xlstm.device, dtype=inputs.dtype)
tokens_with_bos = torch.cat([bos_tensor, inputs], dim=1)

# Generate
outputs = xlstm.generate(
    tokens_with_bos,
    max_new_tokens=200,   # adjust for output length
    temperature=0.7,      # randomness
    top_p=0.9,             # nucleus sampling
    do_sample=True
)

# Decode and print
print(tokenizer.decode(outputs[0]))

# verify selected kernels
from pprint import pprint
pprint(xlstm.backbone.blocks[0].mlstm_layer.config)


