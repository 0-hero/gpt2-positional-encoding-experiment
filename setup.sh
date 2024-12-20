pip install torch torchvision torchaudio scipy wandb tiktoken tqdm huggingface_hub pyarrow pandas flash_attn datasets

mkdir data
mkdir data/fineweb

wget https://huggingface.co/0-hero/fineweb-edu-10BT-GPT2-tokenized/resolve/main/val.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-10BT-GPT2-tokenized/resolve/main/train.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-10BT-GPT2-tokenized/resolve/main/meta.json -P data/fineweb

wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/val.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_001.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_002.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_003.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_004.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_005.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_006.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/train_007.bin -P data/fineweb
wget https://huggingface.co/0-hero/fineweb-edu-100BT-GPT2-tokenized/resolve/main/meta.json -P data/fineweb

python prepare_evaluation_data.py
python train.py

export HF_HOME=/workspace/cache