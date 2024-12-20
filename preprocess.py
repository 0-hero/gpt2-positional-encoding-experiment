import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers in .map() call
num_proc = 64

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split='train', num_proc=num_proc_load_dataset)

    # Create train and validation splits
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

    # Define the encoding function
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Print some statistics
    print(f"Train has ~{arr_len:,} tokens")
    val_size = np.sum(tokenized['val']['len'], dtype=np.uint64)
    print(f"Val has ~{val_size:,} tokens")

    # Save tokenizer information
    import json
    meta = {
        'vocab_size': enc.n_vocab,
    }
    with open('meta.json', 'w') as f:
        json.dump(meta, f)

# To read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')