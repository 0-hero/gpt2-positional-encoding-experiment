# torchrun --standalone --nproc_per_node=2 train.py --batch_size=96

# train.py
import os
import time
import math
from contextlib import nullcontext
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas as pd

import tiktoken
from model import GPTConfig, GPT

# Import wandb and tqdm
import wandb
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Default configuration with added positional encoding options
# I/O
out_dir = 'out'
eval_interval = 100  # Evaluate every 100 iterations
log_interval = 1      # Log every iteration
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' | 'resume' | 'checkpoint'
checkpoint_path = ''   # Path to a specific checkpoint to load
# wandb logging
wandb_log = True
wandb_project = 'gpt2_positional_encodings_100B'
wandb_run_name = 'experiment'
# data
dataset = 'fineweb'
gradient_accumulation_steps = 40
batch_size = 12
block_size = 512
# model
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 10000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# Positional Encodings
embedding_types = ['learned', 'sinusoidal', 'polynomial_legendre',
                   'polynomial_chebyshev', 'random_fourier', 'wavelet']
attention_types = ['default']
# Data collection options
collect_attention_patterns = False  # Set to True to collect attention patterns
collect_activations = False         # Set to True to collect activations
# Evaluation datasets
eval_datasets = ['wikitext-103-v1', 'ptb', 'lambada']  # WikiText-103 and Penn Treebank
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, tuple))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

def is_compatible(embedding_type, attention_type):
    # Incompatible combinations can be specified here
    incompatible_combinations = [
        # If specific combinations are incompatible
    ]

    # If embedding_type or attention_type is 'none', some attention methods may not function properly
    if embedding_type == 'none' and attention_type in ['relative', 'rope']:
        return False

    # 'rope' attention requires even dimension per head
    if attention_type == 'rope' and ((n_embd // n_head) % 2 != 0):
        return False

    return (embedding_type, attention_type) not in incompatible_combinations

def main():
    # Initialize DDP if needed
    global gradient_accumulation_steps
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device_local = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device_local)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device_local = device  # Use the default device

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    if master_process:
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    # Set random seed
    global seed
    seed += seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device_local else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load tokenizer using tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # Prepare evaluation datasets
    eval_data = {}
    for eval_dataset in eval_datasets:
        eval_data_path = os.path.join('data', eval_dataset)
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Dataset {eval_dataset} not found. Please run prepare_evaluation_data.py first.")

        if eval_dataset in ['wikitext-2-v1', 'wikitext-103-v1']:
            train_file = [f for f in os.listdir(eval_data_path) if f.startswith('train')][0]
            val_file = [f for f in os.listdir(eval_data_path) if f.startswith('validation')][0]

            train_df = pd.read_parquet(os.path.join(eval_data_path, train_file))
            val_df = pd.read_parquet(os.path.join(eval_data_path, val_file))

            train_text = '\n'.join(train_df['text'])
            val_text = '\n'.join(val_df['text'])

        elif eval_dataset == 'ptb':
            with open(os.path.join(eval_data_path, 'train.txt'), 'r') as f:
                train_text = f.read()
            with open(os.path.join(eval_data_path, 'valid.txt'), 'r') as f:
                val_text = f.read()

        elif eval_dataset == 'lambada':
            with open(os.path.join(eval_data_path, 'lambada_test.jsonl'), 'r') as f:
                data = [json.loads(line) for line in f]
            test_text = '\n'.join([item['text'] for item in data])
            train_text = test_text[:len(test_text)//2]  # Use first half as pseudo-train
            val_text = test_text[len(test_text)//2:]  # Use second half as pseudo-val

        else:
            raise ValueError(f"Unknown dataset: {eval_dataset}")

        # Tokenize
        train_ids = tokenizer.encode_ordinary(train_text)
        val_ids = tokenizer.encode_ordinary(val_text)

        # Convert to numpy arrays
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)

        eval_data[eval_dataset] = {'train': train_ids, 'val': val_ids}

    # Data loading
    data_dir = os.path.join('data', dataset)
    # Update the get_batch function to handle evaluation datasets
    def get_batch(split, dataset='main'):
        if dataset == 'main':
            if split == 'train':
                data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
            else:
                data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        else:
            data = eval_data[dataset][split]

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device_local, non_blocking=True), y.pin_memory().to(device_local, non_blocking=True)
        else:
            x, y = x.to(device_local), y.to(device_local)
        return x, y

    # Attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.json')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta_vocab_size = meta['vocab_size']
        if master_process:
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Helps estimate loss and collect attention patterns and activations
    @torch.no_grad()
    def estimate_loss(model, collect_attention_patterns=False, collect_activations=False, save_dir=None, max_batches_to_save=None):
        out = {}
        model.eval()
        # Access the underlying model if wrapped with DDP
        raw_model = model.module if hasattr(model, 'module') else model

        # Set tracking flags on the underlying model
        raw_model.config.track_attention_patterns = collect_attention_patterns
        raw_model.config.track_activations = collect_activations

        if collect_attention_patterns or collect_activations:
            if save_dir is None:
                raise ValueError("save_dir must be specified when collecting attention patterns or activations.")
            if master_process:
                os.makedirs(save_dir, exist_ok=True)

        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            save_count = 0  # Counter for saved batches
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
                # Collect and save attention patterns and activations
                if (collect_attention_patterns or collect_activations) and save_count < (max_batches_to_save or eval_iters):
                    if collect_attention_patterns or collect_activations:
                        if master_process:
                            batch_dir = os.path.join(save_dir, f"{split}_batch_{k}")
                            os.makedirs(batch_dir, exist_ok=True)
                            # Save activations
                            if collect_activations and hasattr(raw_model, 'activations'):
                                for idx, activation in enumerate(raw_model.activations):
                                    activation_path = os.path.join(batch_dir, f"activation_layer_{idx}.pt")
                                    torch.save(activation, activation_path)
                            # Save attention patterns
                            if collect_attention_patterns and hasattr(raw_model, 'attention_patterns'):
                                for idx, attention in enumerate(raw_model.attention_patterns):
                                    attention_path = os.path.join(batch_dir, f"attention_layer_{idx}.pt")
                                    torch.save(attention, attention_path)
                            # Clear activations and attention patterns from the model
                            raw_model.activations = []
                            raw_model.attention_patterns = []
                        save_count += 1
            out[split] = losses.mean().item()

        # Evaluate on additional datasets
        for eval_dataset in eval_datasets:
            split_losses = {}
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                save_count = 0  # Counter for saved batches
                for k in range(eval_iters):
                    X, Y = get_batch(split, dataset=eval_dataset)
                    with ctx:
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()
                    # Collect and save attention patterns and activations
                    if (collect_attention_patterns or collect_activations) and save_count < (max_batches_to_save or eval_iters):
                        if collect_attention_patterns or collect_activations:
                            if master_process:
                                batch_dir = os.path.join(save_dir, f"{eval_dataset}_{split}_batch_{k}")
                                os.makedirs(batch_dir, exist_ok=True)
                                # Save activations
                                if collect_activations and hasattr(raw_model, 'activations'):
                                    for idx, activation in enumerate(raw_model.activations):
                                        activation_path = os.path.join(batch_dir, f"activation_layer_{idx}.pt")
                                        torch.save(activation, activation_path)
                                # Save attention patterns
                                if collect_attention_patterns and hasattr(raw_model, 'attention_patterns'):
                                    for idx, attention in enumerate(raw_model.attention_patterns):
                                        attention_path = os.path.join(batch_dir, f"attention_layer_{idx}.pt")
                                        torch.save(attention, attention_path)
                                # Clear activations and attention patterns from the model
                                raw_model.activations = []
                                raw_model.attention_patterns = []
                            save_count += 1
                split_losses[split] = losses.mean().item()
            out[eval_dataset] = split_losses
        model.train()
        # Reset tracking flags
        raw_model.config.track_attention_patterns = False
        raw_model.config.track_activations = False
        return out

    # Learning rate decay scheduler
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # Training loop over positional encoding combinations
    for embedding_type in embedding_types:
        for attention_type in attention_types:
            if not is_compatible(embedding_type, attention_type):
                if master_process:
                    print(f"Skipping incompatible combination: Embedding={embedding_type}, Attention={attention_type}")
                continue

            # Configure model arguments
            model_args = dict(
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                block_size=block_size,
                bias=bias,
                vocab_size=None,
                dropout=dropout,
                embedding_type=embedding_type,
                attention_type=attention_type,
                track_activations=False,
                track_attention_patterns=False,
            )

            # Initialize or resume model
            iter_num = 0
            best_val_loss = 1e9  # initialize best val loss to a high value
            checkpoint = None
            run_id = None  # Initialize run_id to None

            if init_from == 'scratch':
                if master_process:
                    print(f"\nInitializing new model with embedding_type={embedding_type}, attention_type={attention_type}")
                if meta_vocab_size is None:
                    if master_process:
                        print("Defaulting to vocab_size of GPT-2 to 50257")
                model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
                gptconf = GPTConfig(**model_args)
                model = GPT(gptconf)
            elif init_from == 'resume':
                # Resume from the latest checkpoint
                ckpt_path = os.path.join(out_dir, f"ckpt_{embedding_type}_{attention_type}.pt")
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
                if master_process:
                    print(f"\nResuming training from checkpoint {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device_local)
                gptconf = GPTConfig(**checkpoint['model_args'])
                model = GPT(gptconf)
                model.load_state_dict(checkpoint['model'])
                iter_num = checkpoint['iter_num']
                best_val_loss = checkpoint['best_val_loss']
                seed = checkpoint.get('seed', seed)
                run_id = checkpoint.get('wandb_run_id', None)
            elif init_from == 'checkpoint':
                # Resume from a specific checkpoint
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
                if master_process:
                    print(f"\nLoading model from checkpoint {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device_local)
                gptconf = GPTConfig(**checkpoint['model_args'])
                model = GPT(gptconf)
                model.load_state_dict(checkpoint['model'])
                iter_num = checkpoint['iter_num']
                best_val_loss = checkpoint['best_val_loss']
                seed = checkpoint.get('seed', seed)
                run_id = checkpoint.get('wandb_run_id', None)
            else:
                raise ValueError(f"Unknown init_from '{init_from}'")

            # Set random seed
            seed += seed_offset
            torch.manual_seed(seed)
            np.random.seed(seed)

            model.to(device_local)
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
            optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

            # Load optimizer state if resuming
            if checkpoint is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])

            if compile:
                if master_process:
                    print("Compiling the model... (takes a ~minute)")
                unoptimized_model = model
                model = torch.compile(model)

            if ddp:
                model = DDP(model, device_ids=[ddp_local_rank])

            # Logging with WandB
            if wandb_log and master_process:
                run_name = f"{embedding_type}_{attention_type}_{wandb_run_name}"
                # Initialize WandB
                wandb.init(project=wandb_project, name=run_name, config=config, resume='allow', id=run_id)
                # Save the run ID for resuming later
                run_id = wandb.run.id
            else:
                run_id = None

            # Training loop
            X, Y = get_batch('train')
            t0 = time.time()
            local_iter_num = 0
            raw_model = model.module if hasattr(model, 'module') else model
            running_mfu = -1.0
            progress_bar = tqdm(total=max_iters, initial=iter_num, desc=f"Training {embedding_type} + {attention_type}", disable=not master_process)
            progress_bar_update_freq = 1  # Update progress bar every iteration

            while True:
                # Determine learning rate
                lr = get_lr(iter_num) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Evaluate and checkpoint
                if iter_num % eval_interval == 0 and iter_num > 0:
                    # Define save_dir for collected data
                    eval_data_dir = os.path.join('data', 'eval_data', f"{embedding_type}_{attention_type}", f"step_{iter_num}")
                    # Set a limit on the number of batches to save during evaluation
                    max_batches_to_save = 10  # Adjust this number as needed to control storage usage
                    losses = estimate_loss(model,
                                           collect_attention_patterns=collect_attention_patterns,
                                           collect_activations=collect_activations,
                                           save_dir=eval_data_dir,
                                           max_batches_to_save=max_batches_to_save)
                    if master_process:
                        print(f"\nStep {iter_num}:")
                        print(f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")
                        for eval_dataset in eval_datasets:
                            print(f"{eval_dataset} - Train loss: {losses[eval_dataset]['train']:.4f}, Val loss: {losses[eval_dataset]['val']:.4f}")
                        # Log to wandb
                        if wandb_log:
                            wandb_metrics = {
                                "iter": iter_num,
                                "train/loss": losses['train'],
                                "val/loss": losses['val'],
                                "lr": lr,
                                "mfu": running_mfu * 100,
                            }
                            for eval_dataset in eval_datasets:
                                wandb_metrics[f"{eval_dataset}/train_loss"] = losses[eval_dataset]['train']
                                wandb_metrics[f"{eval_dataset}/val_loss"] = losses[eval_dataset]['val']
                            wandb.log(wandb_metrics, step=iter_num)
                    if losses['val'] < best_val_loss or always_save_checkpoint:
                        best_val_loss = losses['val']
                        if iter_num > 0:
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'model_args': model_args,
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': config,
                                'seed': seed,
                                'wandb_run_id': run_id
                            }
                            ckpt_path = os.path.join(out_dir, f"ckpt_{embedding_type}_{attention_type}.pt")
                            if master_process:
                                print(f"Saving checkpoint to {ckpt_path}")
                            torch.save(checkpoint, ckpt_path)
                    # Update progress bar postfix
                    if master_process:
                        postfix_dict = {
                            'train_loss': f"{losses['train']:.4f}",
                            'val_loss': f"{losses['val']:.4f}"
                        }
                        for eval_dataset in eval_datasets:
                            postfix_dict[f"{eval_dataset}_val_loss"] = f"{losses[eval_dataset]['val']:.4f}"
                        progress_bar.set_postfix(postfix_dict)

                if eval_only:
                    break

                # Forward backward update
                for micro_step in range(gradient_accumulation_steps):
                    if ddp:
                        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                    with ctx:
                        logits, loss = model(X, Y)
                        loss = loss / gradient_accumulation_steps
                    X, Y = get_batch('train')
                    scaler.scale(loss).backward()
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if iter_num % log_interval == 0:
                    lossf = loss.item() * gradient_accumulation_steps
                    if local_iter_num >= 5:
                        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    if master_process:
                        progress_bar.set_postfix({
                            'loss': f"{lossf:.4f}",
                            'lr': f"{lr:.2e}",
                            'mfu': f"{running_mfu*100:.2f}%",
                            'time_per_iter_ms': f"{dt * 1000:.2f}ms",
                        })
                        if wandb_log:
                            wandb.log({
                                "iter": iter_num,
                                "train/loss": lossf,
                                "lr": lr,
                                "mfu": running_mfu * 100,
                                "time_per_iter_ms": dt * 1000,
                            }, step=iter_num)
                iter_num += 1
                local_iter_num += 1
                if master_process:
                    progress_bar.update(progress_bar_update_freq)
                # Termination conditions
                if iter_num > max_iters:
                    break

            if master_process:
                progress_bar.close()
            if wandb_log and master_process:
                wandb.finish()

    # Destroy the process group after all models have been trained
    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()