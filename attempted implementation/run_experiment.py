import sys
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
from dacite import from_dict

from experiments.data.formal_language.formal_language_dataset import FormLangDatasetGenerator
from experiments.lr_scheduler import LinearWarmupCosineAnnealing

# Import local architecture
from xlstm import xLSTMConfig, xLSTM

torch_dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def load_dataset(kwargs):
    # The kwargs config matches FormLangDatasetConfig
    return FormLangDatasetGenerator(from_dict(FormLangDatasetGenerator.config_class, OmegaConf.to_container(kwargs)))

def get_weight_decay_optim_groups(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and LayerNorm/GroupNorm/RMSNorm weights shouldn't be decayed
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay

def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.training.seed)

    dataset = load_dataset(cfg.dataset.kwargs)
    train_loader = DataLoader(dataset.train_split, batch_size=cfg.training.batch_size)
    val_loaders = {
        key: DataLoader(val_ds, batch_size=cfg.training.batch_size)
        for key, val_ds in dataset.validation_split.items()
    }

    device = cfg.training.device
    
    # We extract the base device type for autocast (e.g. "cuda:0" -> "cuda")
    autocast_device = device.split(":")[0] if isinstance(device, str) else "cuda"
    
    train_metrics = dataset.train_metrics.to(device=device)
    val_metrics = dataset.validation_metrics.to(device=device)

    # Determine blocks layout
    slstm_at = cfg.model.get("slstm_at", [])
    block_types = ["s" if i in slstm_at else "m" for i in range(cfg.model.num_blocks)]
    
    num_heads = 1
    if "mlstm_block" in cfg.model and "mlstm" in cfg.model.mlstm_block:
        num_heads = cfg.model.mlstm_block.mlstm.get("num_heads", 1)

    xlstm_config = xLSTMConfig(
        vocab_size=cfg.model.vocab_size,
        dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_blocks,
        num_heads=num_heads,
        block_types=block_types,
        slstm_use_conv1d=False,
        slstm_normal_init=True,
    )
    
    print(f"Creating xLSTM model with blocks: {block_types}")
    model = xLSTM(xlstm_config).to(device=device)
    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])
    print(f"Total trainable parameters: {model.num_parameters(trainable_only=True):,}")
    return
    decay, no_decay = get_weight_decay_optim_groups(model)
    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.lr,
    )

    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )

    # Training loop
    step = 0
    epoch = 1
    running_loss = 0.0
    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=0, initial=0)
        for inputs, labels in monitoring:
            monitoring.set_description_str(f"Steps {step+1}/{cfg.training.num_steps} (Epoch: {epoch})")
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            model.train()
            optimizer.zero_grad()
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, cfg.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-1,
                )
                
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            running_loss = running_loss * step / (step + 1) + loss.item() * 1 / (step + 1)
            step += 1
            train_metrics.update(outputs, labels)

            if step % cfg.training.val_every_step == 0:
                print(
                    f"\nStep [{step}/{cfg.training.num_steps}] (Epoch: {epoch}), Loss: {running_loss:.4f}, "
                    f"Metrics: {train_metrics.compute()}"
                )
                # Validation
                for vl_name, val_loader in val_loaders.items():
                    model.eval()
                    val_loss = 0.0
                    val_metrics.reset()
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs = val_inputs.to(device=device)
                            val_labels = val_labels.to(device=device)
                            with torch.autocast(
                                device_type=autocast_device,
                                dtype=torch_dtype_map[cfg.training.amp_precision],
                                enabled=cfg.training.enable_mixed_precision,
                            ):
                                val_outputs = model(val_inputs)
                                loss = nn.functional.cross_entropy(
                                    val_outputs.view(-1, cfg.model.vocab_size),
                                    val_labels.view(-1),
                                    ignore_index=-1,
                                )
                                val_loss += loss.item()
                                val_metrics.update(val_outputs, val_labels)
                        print(f"Validation[{vl_name}] Loss: {val_loss/len(val_loader):.4f}, Metrics: {val_metrics.compute()}")

            if step >= cfg.training.num_steps:
                break
        epoch += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="experiments/parity_xlstm11.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
        
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg)
