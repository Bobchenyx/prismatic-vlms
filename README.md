# VLM for wwc

## Installation

```bash
git clone https://github.com/Bobchenyx/prismatic-vlms.git
cd prismatic-vlms

conda create -n prismatic python=3.10 -y
conda activate prismatic

pip install torch==2.1.0 torchvision==0.16.0

pip install transformers==4.51.3

pip install -e .

# Install Flash Attention 2 
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
MAX_JOBS=4 pip install flash-attn==2.3.3 --no-build-isolation

pip install peft

pip install 'numpy<2'
```

## Usage

Download model from [HF](https://huggingface.co/bobchenyx/vlm-for-wwc)

create a folder called `runs`

place `prism-moxin-dinosiglip-224px+7b+stage-finetune+x7` in `runs`

put your HF token in `.hf_token`(create this file first)

```bash
python scripts/generate.py --model_path runs/prism-moxin-dinosiglip-224px+7b+stage-finetune+x7

```

default image url `https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png`

demo questions `tell me what can you see in this image`

