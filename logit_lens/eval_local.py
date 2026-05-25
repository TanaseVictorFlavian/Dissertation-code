import torch
import math
from transformers import AutoModelForCausalLM

@torch.no_grad() 
def evaluate_perplexity(model, test_tensor, context_length=512, batch_size=16):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    num_windows = len(test_tensor) // context_length
    truncated_test = test_tensor[:num_windows * context_length]
    eval_chunks = truncated_test.view(num_windows, context_length).to(torch.long)
    
    print(f"Total chunks to evaluate: {num_windows} (batch size: {batch_size})")
    
    for i in range(0, num_windows, batch_size):
        batch_chunks = eval_chunks[i : i + batch_size].to(model.device)
        inputs = {"input_ids": batch_chunks, "labels": batch_chunks}
        
        outputs = model(**inputs)
        total_loss += outputs.loss.item()
        total_batches += 1
        
        if total_batches % 10 == 0:
            print(f"Processed {total_batches} / {math.ceil(num_windows / batch_size)} batches...")
            
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return perplexity

def main():
    print("Loading test tensor from ./data/test_tokens_list.pt...")
    test_tensor = torch.load("./data/test_tokens_list.pt")
    print(f"Loaded {len(test_tensor):,} test tokens.")

    print("\nLoading model from ./model_checkpoints/step_100...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from local checkpoint
    model = AutoModelForCausalLM.from_pretrained("./model_checkpoints/step_100")
    model.to(device)

    print("\nEvaluating perplexity on local GPU...")
    ppl = evaluate_perplexity(model, test_tensor, context_length=512, batch_size=16)
    print(f"\nFinal Test Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
