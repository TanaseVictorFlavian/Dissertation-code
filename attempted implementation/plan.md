# Thesis Plan: Logit Lens Analysis of xLSTM-7B

## Working Title
*"What Does a Recurrent LLM Believe? Applying the Logit Lens to xLSTM-7B"*

**Timeline:** 10 weekends, 4-6 hours each (40-60 hours total)  
**Deliverable:** ~30-page thesis

Legend:
- **[E]** = Essential -- bare minimum to keep the project on track
- **[+]** = Enhancement -- improves quality but can be skipped or deferred

---

## Weekend 1 -- Environment Setup and First Forward Pass

**Goal:** Get xLSTM-7B running and confirm you can extract hidden states.

### Important: Model Loading Options

The **weights** live exclusively on HuggingFace (`NX-AI/xLSTM-7b`). The NX-AI
GitHub repo (`github.com/NX-AI/xlstm`) only has the architecture code, not
the weights. You have two loading paths:

**Path A (recommended): HuggingFace Transformers.** xLSTM was merged into
mainline `transformers` in July 2025. Load with:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

xlstm_config = AutoConfig.from_pretrained("NX-AI/xLSTM-7b")
# Fall back to native kernels if Triton gives trouble:
xlstm_config.step_kernel = "native"
xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
xlstm_config.sequence_kernel = "native_sequence__native"

model = AutoModelForCausalLM.from_pretrained(
    "NX-AI/xLSTM-7b",
    config=xlstm_config,
    device_map="auto",
    torch_dtype=torch.float16,  # if VRAM is tight
)
tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
```
This gives you `output_hidden_states=True` for free, which is exactly what
you need for the logit lens.

**Path B (fallback): Standalone NX-AI package.** The standalone implementation
in `xlstm/xlstm_large/model.py` requires `mlstm_kernels` but has no dependency
on the HF transformers integration. You would still download weights from
HuggingFace but load them manually into the standalone model class. Only use
this if Path A has issues.

**Note:** There are known bugs in the HF integration for configs smaller than
7B (GitHub issue #43208), but this will not affect you since you are only
doing inference on the 7B checkpoint.

### xLSTM-7B Module Tree (HuggingFace)

When you load via `AutoModelForCausalLM`, the internal structure is:
```
model
├── backbone
│   ├── embedding        # token embeddings
│   ├── blocks           # ModuleList of 32 xLSTM blocks
│   │   └── [0..31]
│   │       ├── mlstm_layer          # the mLSTM sequence mixer
│   │       │   ├── mlstm_backend    # core recurrence (hooks go here for sub-component analysis)
│   │       │   └── ...
│   │       └── ffn                  # the SwiGLU MLP (channel mixer)
│   └── post_blocks_norm             # final RMSNorm (apply this before unembedding)
└── lm_head                          # unembedding matrix (hidden_dim → vocab_size)
```

For the logit lens you need:
- Hidden states after each block: `model.backbone.blocks[l]` output
- Final norm: `model.backbone.post_blocks_norm`
- Unembedding: `model.lm_head`

For the component decomposition (Weekend 5):
- Post-mLSTM state: `model.backbone.blocks[l].mlstm_layer` output
- Post-MLP state: `model.backbone.blocks[l].ffn` output

Run `print(model)` on Weekend 1 to verify these paths match your
transformers version. They may differ slightly across releases.

### Tasks

- **[E] Install dependencies** -- `pip install transformers xlstm mlstm_kernels`.
  If on the HF path, verify you have transformers >= 4.44 (the version that
  includes xLSTM). Verify CUDA/GPU access. If Triton kernels fail, switch to
  native kernels using the config overrides shown above.
  *(~1 hour)*

- **[E] Load the model and run inference** -- use Path A above. Run a basic
  forward pass on a short prompt and confirm you get logits out. Confirm
  generation works with `model.generate()` on a simple prompt like
  "The capital of France is". If Path A fails, fall back to Path B (clone
  repo, `pip install -e .`, load weights manually).
  *(~1.5 hours)*

- **[E] Extract hidden states** -- run a forward pass with
  `output_hidden_states=True` and verify you get a tuple of 33 tensors
  (embedding layer + 32 blocks), each shaped `(batch, seq_len, 4096)`.
  ```python
  inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
  with torch.no_grad():
      outputs = model(**inputs, output_hidden_states=True)
  hidden_states = outputs.hidden_states  # tuple of 33 tensors
  print(len(hidden_states), hidden_states[0].shape)
  ```
  If `output_hidden_states` is not supported for xLSTM in your transformers
  version, fall back to `register_forward_hook` on each `model.backbone.blocks[l]`.
  *(~1.5 hours)*

- **[+] Also load Llama-2-7B** -- load `meta-llama/Llama-2-7b-hf` (or
  Llama-3-8B if you have access), confirm you can extract hidden states
  the same way with `output_hidden_states=True`. This is your Transformer
  comparison model. Can wait until Weekend 3 if time is tight.
  *(~1 hour)*

**Weekend 1 exit criterion:** You can run `model(input_ids)` for xLSTM-7B and
have a list of 32+ hidden state tensors with correct shapes.

---

## Weekend 2 -- Implement Logit Lens and First Visualizations

**Goal:** Produce your first layer-by-layer token prediction plots for xLSTM.

### Tasks

- **[E] Identify the unembedding matrix** -- in the xLSTM-7B architecture, locate
  the LM head / output projection that maps hidden states to vocabulary logits.
  Confirm its shape is `(hidden_dim, vocab_size)` or similar. Also check
  if there is a final LayerNorm/RMSNorm before the head -- you will need to
  apply it to intermediate states too (xLSTM uses `add_out_norm=True` by
  default, which is an RMSNorm).
  *(~0.5 hours)*

- **[E] Implement vanilla logit lens** -- for each layer `l`, compute:
  ```python
  norm = model.backbone.post_blocks_norm
  lm_head = model.lm_head
  
  logits_per_layer = []
  for hs in hidden_states[1:]:  # skip embedding, keep 32 block outputs
      logits_l = lm_head(norm(hs))
      logits_per_layer.append(logits_l)
  
  # top prediction at each layer
  top_tokens = [l.argmax(dim=-1) for l in logits_per_layer]
  ```
  Decode `top_tokens` back to strings. Print the results for a test prompt.
  *(~1.5 hours)*

- **[E] Build the heatmap visualization** -- for a single prompt, create a
  `(layers x tokens)` grid showing:
  - The top-1 predicted token at each layer/position (as text in cells)
  - Color by the probability or logit of the top prediction
  - Highlight cells where the top prediction matches the final layer prediction

  Use matplotlib or plotly. This is the signature figure of the thesis.
  *(~2 hours)*

- **[+] Test on 3-5 diverse prompts** -- try factual ("The capital of France is"),
  syntactic ("The dogs in the park are"), rare token, and long-context prompts.
  Screenshot everything -- these become thesis figures later.
  *(~1 hour)*

**Weekend 2 exit criterion:** You have at least one heatmap plot showing
token prediction evolution across all 32 layers of xLSTM-7B.

---

## Weekend 3 -- Transformer Baseline and Quantitative Metrics

**Goal:** Run the same analysis on Llama-2-7B and define your comparison metrics.

### Tasks

- **[E] Implement logit lens for Llama-2-7B** -- identical pattern:
  ```python
  llama = AutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.float16
  )
  outputs = llama(**inputs, output_hidden_states=True)
  # Llama norm + head:
  norm = llama.model.norm       # RMSNorm
  lm_head = llama.lm_head       # unembedding
  ```
  Generate the same heatmap plots for the same prompts you used on xLSTM.
  The code is nearly identical — just swap the norm/head references.
  *(~1.5 hours)*

- **[E] Implement core metrics** -- for each layer, compute:
  1. **Correct token rank** -- rank of the ground-truth next token in the
     logit-lens distribution (lower = the model "knows" earlier)
  2. **KL divergence from final layer** -- `KL(final_distribution || layer_distribution)`.
     Measures how far each layer's belief is from the converged prediction.
  3. **Entropy** -- `H(layer_distribution)`. Measures certainty at each layer.

  Store these as `(num_layers,)` arrays averaged over a batch of prompts.
  *(~2 hours)*

- **[E] Create comparison line plots** -- plot each metric (y-axis) vs. layer index
  (x-axis) with two lines: xLSTM and Llama. These are the core quantitative
  figures of the thesis.
  *(~1 hour)*

- **[+] Run on a larger prompt set** -- sample 100-200 sentences from WikiText-103
  validation set. Compute metrics in batch. This strengthens statistical claims.
  *(~1.5 hours)*

**Weekend 3 exit criterion:** You have side-by-side comparison plots of xLSTM
vs Llama on at least correct-token-rank and KL-divergence metrics.

---

## Weekend 4 -- Categorized Analysis and Interesting Findings

**Goal:** Dig into *where* the two architectures differ and find the story.

### Tasks

- **[E] Categorize prompts by type** -- split your evaluation prompts into groups:
  - Factual recall ("The president of the USA in 2020 was")
  - Syntactic prediction ("The cats on the mat")
  - Common collocations ("bread and")
  - Rare / surprising continuations

  Compute per-category metrics for both models. Look for categories where
  xLSTM converges earlier or later than Llama.
  *(~2 hours)*

- **[E] Document key observations** -- write bullet-point notes on:
  - At which layer does xLSTM typically "lock in" its prediction?
  - Are there prompt types where xLSTM trajectory is qualitatively different?
  - Do early layers in xLSTM produce more or less "nonsense" than Llama?

  These notes become the Results section of the thesis.
  *(~1 hour)*

- **[+] Cherry-pick compelling examples** -- find 3-4 prompts where the difference
  between xLSTM and Llama is most visually striking. Generate high-quality
  heatmaps for these -- they will be your showcase figures.
  *(~1 hour)*

- **[+] Confidence calibration** -- for both models, check whether the logit-lens
  probabilities at intermediate layers are well-calibrated (does a 0.8
  probability actually mean correct 80% of the time?). This is a nice
  extra analysis that adds depth.
  *(~1.5 hours)*

**Weekend 4 exit criterion:** You have categorized results with written
observations about xLSTM vs. Llama prediction trajectories.

---

## Weekend 5 -- Extension: Component Decomposition

**Goal:** Go one level deeper -- separate the mLSTM from the MLP within each block.

### Tasks

- **[E] Hook sub-components** -- within each xLSTM block, there are two parts:
  the mLSTM (sequence mixer) and the SwiGLU MLP (channel mixer). Place
  `register_forward_hook` on `model.backbone.blocks[l].mlstm_layer` and
  `model.backbone.blocks[l].ffn` to capture post-mLSTM and post-MLP states
  separately. This gives you two "snapshots" per layer.
  *(~1.5 hours)*

- **[E] Run logit lens at sub-component level** -- apply the unembedding to the
  post-mLSTM state and the post-MLP state separately. Plot the difference:
  does the mLSTM or the MLP contribute more to shifting the prediction at
  each layer?
  *(~1.5 hours)*

- **[+] Compare with Llama attention vs. MLP split** -- in Llama, do the same
  decomposition (post-attention vs. post-MLP). This gives a direct comparison
  of "how do recurrent vs. attention sequence mixers differ in their
  contribution to prediction refinement?"
  *(~1.5 hours)*

- **[+] Logit attribution plot** -- for each layer, compute the change in
  correct-token logit from the mLSTM vs. the MLP. Stacked bar chart showing
  relative contribution. This is a strong figure.
  *(~1 hour)*

**Weekend 5 exit criterion:** You know whether the mLSTM or MLP is doing
the heavy lifting at each depth, and have at least one figure showing this.

---

## Weekend 6 -- Extension: Layer Knockout (if ahead) / Buffer (if behind)

**Goal:** If you are on schedule, add a layer ablation analysis. If behind, catch up.

### If on schedule:

- **[+] Implement layer knockout** -- for each layer `l` (0-31), skip it during
  forward pass and measure the change in:
  - Final perplexity on a validation set (~200 sentences)
  - Top-1 accuracy

  This tells you which layers are critical and which are redundant.
  *(~2 hours)*

- **[+] Plot knockout impact curve** -- x-axis = layer removed, y-axis = perplexity
  increase. Compare xLSTM and Llama. Transformers typically show a pattern
  where middle layers are more removable -- does xLSTM show the same?
  *(~1 hour)*

- **[+] Cross-reference with logit lens** -- correlate: are the layers where
  predictions change most (from logit lens) also the most critical in
  knockout? This ties the two analyses together.
  *(~1 hour)*

### If behind schedule:

- **[E] Catch up on any incomplete essential tasks from Weekends 1-5.**
- **[E] Consolidate all figures and results into a single notebook/folder.**
  *(~3 hours)*

**Weekend 6 exit criterion:** All experimental work is complete. Every figure
you plan to include in the thesis exists in at least draft form.

---

## Weekend 7 -- Write Introduction, Background, and Methods

**Goal:** Draft the first half of the thesis.

### Tasks

- **[E] Introduction (~3 pages)** -- frame the problem:
  - Transformers dominate but alternatives like xLSTM are emerging
  - Interpretability work has focused almost entirely on Transformers
  - Research question: how do predictions form inside a recurrent LLM?
  - Brief summary of contributions and findings
  *(~2 hours)*

- **[E] Background (~4 pages)** -- cover:
  - The logit lens / tuned lens lineage (cite nostalgebraist, Belrose et al.)
  - xLSTM architecture (mLSTM, exponential gating, matrix memory)
  - Briefly: Transformer architecture for comparison
  - Related interpretability work on non-Transformers (cite the Mamba IOI paper)
  *(~2 hours)*

- **[+] Methods (~3 pages)** -- describe:
  - Models used (xLSTM-7B, Llama-2-7B, their training data and params)
  - Logit lens implementation details (including which norm to apply)
  - Metrics (KL divergence, rank, entropy)
  - Prompt dataset construction
  - Component decomposition setup
  *(~1.5 hours -- can continue next weekend)*

**Weekend 7 exit criterion:** Introduction and Background sections are drafted.

---

## Weekend 8 -- Write Results and Analysis

**Goal:** Draft the core of the thesis -- presenting your findings.

### Tasks

- **[E] Finish Methods if incomplete** from Weekend 7.
  *(~0.5 hours)*

- **[E] Results (~6-8 pages)** -- present findings in order:
  1. Qualitative heatmaps (cherry-picked examples)
  2. Quantitative metrics (rank, KL, entropy curves)
  3. Per-category breakdown
  4. Component decomposition (mLSTM vs. MLP)
  5. Layer knockout (if completed)

  For each: figure/table, then describe what it shows, then interpret.
  *(~3 hours)*

- **[E] Analysis / Discussion (~3 pages)** -- synthesize:
  - What do these results tell us about how xLSTM builds predictions?
  - How does it compare to the iterative refinement pattern in Transformers?
  - What are the implications for interpretability of recurrent architectures?
  - Limitations (single model, logit lens biases, no tuned lens)
  *(~2 hours)*

**Weekend 8 exit criterion:** Results and Discussion sections are drafted.

---

## Weekend 9 -- Write Conclusion, Abstract and Polish Figures

**Goal:** Complete the first full draft.

### Tasks

- **[E] Conclusion (~1-2 pages)** -- summarize findings, restate contributions,
  suggest future work (tuned lens for xLSTM, scaling to other sizes,
  SAE-based analysis, hybrid architectures).
  *(~0.5 hours)*

- **[E] Abstract (~250 words)** -- write last, after everything else is done.
  *(~0.5 hours)*

- **[E] Polish all figures** -- ensure consistent styling, readable fonts,
  proper axis labels, captions. Export as PDF/SVG for crisp rendering.
  *(~1.5 hours)*

- **[E] References** -- compile bibliography. You should have roughly 20-30 refs.
  Core ones: nostalgebraist (2020), Belrose et al. (2023), Beck et al.
  (2024, 2025), the Mamba IOI paper, Rai et al. (2024) MI survey,
  Zimerman et al. (2024) implicit attention.
  *(~1 hour)*

- **[+] Appendix** -- dump additional heatmaps, full metric tables, or
  implementation details that do not fit in the main text.
  *(~1 hour)*

**Weekend 9 exit criterion:** Complete first draft exists from Abstract
to References.

---

## Weekend 10 -- Revise, Proofread, Submit

**Goal:** Turn the draft into a finished thesis.

### Tasks

- **[E] Full read-through** -- read the entire thesis start to finish. Fix
  logical flow, remove repetition, tighten language. Check that every
  figure is referenced in the text and every claim is supported.
  *(~2 hours)*

- **[E] Check formatting** -- page numbers, heading styles, figure numbering,
  table of contents, margins per your institution's requirements.
  *(~1 hour)*

- **[+] Get feedback** -- if possible, send the draft to your advisor or a
  peer by midweek. Incorporate feedback this weekend.
  *(~1 hour for revisions)*

- **[E] Final proofread** -- one last pass for typos, broken references,
  figure resolution.
  *(~1 hour)*

- **[E] Submit.**

---

## Risk Mitigation Notes

**Biggest risk: Weekend 1 environment issues.**
- If Triton kernels fail: switch to native PyTorch kernels via config overrides
  (shown in Weekend 1). This is slower but functionally identical for inference.
- If you cannot fit xLSTM-7B in VRAM: (a) use `torch_dtype=torch.float16`,
  (b) use `device_map="auto"` for CPU offload, (c) use a cloud GPU (Colab
  Pro A100 or Lambda Labs). Test this in the first hour.
- If the HuggingFace transformers integration has issues: fall back to the
  standalone NX-AI package (Path B). Clone the repo, install with
  `pip install -e .`, install `mlstm_kernels`, and load weights manually
  from the HF hub. The demo notebook at `notebooks/xlstm_large/demo.ipynb`
  shows how.

**If you fall behind:** Weekends 5 and 6 are your buffer. The component
decomposition and layer knockout are enhancements. A thesis with just the
logit lens comparison (Weekends 1-4) plus a solid writeup (Weekends 7-10)
is already a complete and publishable study.

**If you are ahead:** Beyond the extensions listed, you could add:
- A third comparison model (Mamba-2.8B, though smaller, or RWKV-7B if available)
- Perplexity-stratified analysis (how does the logit lens behave on tokens
  the model gets right vs. wrong?)
- Positional analysis (does xLSTM recurrent state show different patterns
  at position 10 vs. position 500?)

---

## Thesis Structure Summary (~30 pages)

| Section              | Pages | Produced in   |
|----------------------|-------|---------------|
| Abstract             | 0.5   | Weekend 9     |
| Introduction         | 3     | Weekend 7     |
| Background           | 4     | Weekend 7     |
| Methods              | 3     | Weekend 7-8   |
| Results              | 8     | Weekend 8     |
| Discussion           | 3     | Weekend 8     |
| Conclusion           | 1.5   | Weekend 9     |
| References           | 2     | Weekend 9     |
| Appendix             | 5     | Weekend 9     |
| **Total**            | **30**|               |