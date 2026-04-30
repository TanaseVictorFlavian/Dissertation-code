import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore HDBSCAN future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("Loading tokenizer and model...")
    xlstm_config = AutoConfig.from_pretrained("NX-AI/xLSTM-7b")
    xlstm_config.step_kernel = "native"
    xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
    xlstm_config.sequence_kernel = "native_sequence__native"

    model = AutoModelForCausalLM.from_pretrained(
        "NX-AI/xLSTM-7b",
        config=xlstm_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

    print("Reading text...")
    with open("big_text_prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)
    bos_id = tokenizer.bos_token_id
    bos_tensor = torch.tensor([[bos_id]], device=model.device, dtype=inputs.dtype)
    tokens_with_bos = torch.cat([bos_tensor, inputs], dim=1)

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(tokens_with_bos, output_hidden_states=True)

    hidden_states = outputs.hidden_states # Tuple of 33 tensors of (1, seq_len, 4096)
    
    print("Extracting last token activations across layers...")
    # For predicting the next token after the entire sequence, 
    # the relevant activation is the one at the very last sequence position.
    last_token_activations = []
    for h_state in hidden_states:
        # h_state is (1, seq_len, 4096)
        last_tok_emb = h_state[0, -1, :].float().cpu()
        last_token_activations.append(last_tok_emb)
        
    # Shape: (33, 4096)
    X = torch.stack(last_token_activations)
    
    # Normalize
    X_norm = F.normalize(X, p=2, dim=1).numpy()
    
    print("Running t-SNE and clustering on the 33 layers...")
    
    # t-SNE with low perplexity since we only have 33 samples
    tsne = TSNE(n_components=2, perplexity=8, random_state=42)
    X_2d = tsne.fit_transform(X_norm)
    
    # 1. HDBSCAN
    hdbscan_labels = HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_epsilon=0.1).fit_predict(X_norm)
    
    # 2. KMeans (k=2, 3, 4)
    # Using n_init=10 explicitly to avoid future warnings
    kmeans_2_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_norm)
    kmeans_3_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_norm)
    kmeans_4_labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_norm)
    
    clusterings = [
        ("HDBSCAN", hdbscan_labels),
        ("K-Means (k=2)", kmeans_2_labels),
        ("K-Means (k=3)", kmeans_3_labels),
        ("K-Means (k=4)", kmeans_4_labels)
    ]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for ax, (title, labels) in zip(axes, clusterings):
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        palette = sns.color_palette("husl", max(1, n_clusters))
        
        # Plot noise (only applicable for HDBSCAN)
        noise_mask = (labels == -1)
        if np.any(noise_mask):
            ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], c='lightgrey', s=100, alpha=0.7, label='Noise')
            for i in np.where(noise_mask)[0]:
                ax.annotate(str(i), (X_2d[i, 0], X_2d[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9, color='grey')
                
        # Plot clusters
        cluster_labels = sorted([l for l in unique_labels if l != -1])
        for idx, label in enumerate(cluster_labels):
            mask = (labels == label)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], color=palette[idx], s=100, label=f'Cluster {label}')
            for i in np.where(mask)[0]:
                ax.annotate(str(i), (X_2d[i, 0], X_2d[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
                
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(loc='best')
        
    plt.suptitle("Clustering of the Last Token's Activation Across 33 Layers", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("hdbscan_layers_plot.png", dpi=300, bbox_inches='tight')
    print("Plot saved to hdbscan_layers_plot.png")

if __name__ == "__main__":
    main()
