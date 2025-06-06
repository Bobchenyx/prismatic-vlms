"""
models.py

Draccus Dataclass Definition for a ModelConfig object, with various registered subclasses for each model family and
variant thereof. A given model variant configures the following attributes:
    - Pretrained Visual Representation (e.g., OpenAI CLIP ViT-L/14) + Pretrained LLM Backbone (e.g., LLaMa-2 7B)
    - VLM Configuration + Parameters (e.g., MLP Projector, Image Preprocessing, etc.)
    - [Optional] Stage 1 (`align`) Optimization Hyperparameters
    - Stage 2 (`finetune`) Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

from draccus import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    # fmt: off
    model_id: str                                           # Unique Model ID that fully specifies a given variant
    arch_specifier: str                                     # Architecture specifier string (e.g., "gelu-mlp")

    # Pretrained Backbones
    vision_backbone_id: str                                 # Pretrained Visual Featurizer (from TIMM) to load
    llm_backbone_id: str                                    # Pretrained LLM (from HF Transformers) to load

    # Backbone Parameters
    image_resize_strategy: str                              # Resizing strategy in < crop | letterbox | corner-pad >
    llm_max_length: int                                     # Maximum context length for LLM (can be < than max!)

    # === Multi-Stage Optimization Hyperparameters ===
    # By default, we assume an AdamW optimizer with FSDP (Gradient Sharding or Full Sharding depending on stage)

    # Align Stage Optimization Parameters
    align_epochs: int                                       # Epochs to Run (in case `max_steps` is not specified)
    align_max_steps: Optional[int]                          # [Optional] Max Gradient Steps (overrides epochs)
    align_global_batch_size: int                            # Global Batch Size (divided across processes)
    align_per_device_batch_size: int                        # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    align_learning_rate: float                              # Peak Learning Rate (lr_scheduler sets warmup/decay)
    align_weight_decay: float                               # Weight Decay for AdamW Optimizer
    align_max_grad_norm: float                              # Max Grad Norm (for global gradient clipping)
    align_lr_scheduler_type: str                            # LR Scheduler (default: "linear-warmup+cosine-decay")
    align_warmup_ratio: float                               # Fraction of total steps to warmup

    align_train_strategy: str                               # Align Train Strategy (default: "fsdp-shard-grad-op")

    # Finetune Stage Optimization Parameters
    finetune_epochs: int                                    # Epochs to Run (in case `max_steps` is not specified)
    finetune_max_steps: Optional[int]                       # [Optional] Max Gradient Steps (overrides epochs)
    finetune_global_batch_size: int                         # Global Batch Size (divided across processes)
    finetune_per_device_batch_size: int                     # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    finetune_learning_rate: float                           # Peak Learning Rate (lr_scheduler sets warmup/decay)
    finetune_weight_decay: float                            # Weight Decay for AdamW Optimizer
    finetune_max_grad_norm: float                           # Max Grad Norm (for global gradient clipping)
    finetune_lr_scheduler_type: str                         # LR Scheduler (default: "linear-warmup+cosine-decay")
    finetune_warmup_ratio: float                            # Fraction of total steps to warmup

    finetune_train_strategy: str                            # Finetune Train Strategy (default: "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True

    # Enable Traditional Mixed Precision Training via Torch Native AMP (`autocast`)

    # I changed the following line from True to False to use fp16 instead of bfloat16
    enable_mixed_precision_training: bool = True            # Whether to enable mixed precision training
    # Did not change this line
    reduce_in_full_precision: bool = False                  # Whether to run gradient reduction in FP32

    # fmt: on


# === LLaVa v1.5 Reproduction - Fully Specified Configurations ===
@dataclass
class LLaVa_v15_Reproduction_7B(ModelConfig):
    model_id: str = "reproduction-llava-v15+7b"
    arch_specifier: str = "gelu-mlp"

    vision_backbone_id: str = "clip-vit-l-336px"
    llm_backbone_id: str = "vicuna-v15-7b"

    image_resize_strategy: str = "letterbox"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_max_steps: Optional[int] = None
    align_global_batch_size: int = 8
    align_per_device_batch_size: int = 8

    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03

    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 1
    finetune_max_steps: Optional[int] = None
    finetune_global_batch_size: int = 128
    finetune_per_device_batch_size: int = 16

    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03

    finetune_train_strategy: str = "fsdp-full-shard"


@dataclass
class LLaVa_v15_Reproduction_13B(LLaVa_v15_Reproduction_7B):
    model_id: str = "reproduction-llava-v15+13b"
    llm_backbone_id: str = "vicuna-v15-13b"


# === Section 4.1 :: Optimization Procedure ===


# Section 4.1A :: 🚀 --> Necessity of Multi-Stage Training
@dataclass
class Exp_7B_One_Stage(LLaVa_v15_Reproduction_7B):
    model_id: str = "one-stage+7b"
    arch_specifier: str = "no-align+gelu-mlp"


@dataclass
class Exp_13B_One_Stage(LLaVa_v15_Reproduction_13B):
    model_id: str = "one-stage+13b"
    arch_specifier: str = "no-align+gelu-mlp"


# Section 4.1B :: 🛠️ --> Full Finetuning through Visual Backbones
#   =>> Note :: Run with `--stage full-finetune`
@dataclass
class Exp_7B_Full_Finetune_Multi_Stage(LLaVa_v15_Reproduction_7B):
    model_id: str = "full-ft-multi-stage+7b"


@dataclass
class Exp_7B_Full_Finetune_One_Stage(Exp_7B_One_Stage):
    model_id: str = "full-ft-one-stage+7b"


# === Section 4.3 :: Language Models ===

# ~ LLM Backbones
@dataclass
class Ext_Exp_7B_Moxin(Exp_7B_One_Stage):
    model_id: str = "moxin+7b"
    llm_backbone_id: str = "moxin-7b-pure"

# === Section 4.4 :: Scaling Properties - Train Time & Data ===


# Section 4.4A :: ⏰ --> Scaling Train Time
@dataclass
class Exp_7B_1p25_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-1.25-epochs+7b"
    finetune_max_steps: int = 6500


@dataclass
class Exp_7B_1p5_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-1.5-epochs+7b"
    finetune_max_steps: int = 7800


@dataclass
class Exp_7B_2_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-2-epochs+7b"
    finetune_epochs: int = 2


@dataclass
class Exp_7B_3_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-3-epochs+7b"
    finetune_epochs: int = 3


# Section 4.4B :: 📚 --> Scaling Data
#   =>> Note :: Run with `--dataset.type "llava-lvis4v"`
@dataclass
class Exp_7B_LLaVa_LVIS4V(Exp_7B_One_Stage):
    model_id: str = "llava-lvis4v+7b"


#   =>> Note :: Run with `--dataset.type "llava-lrv"`
@dataclass
class Exp_7B_LLaVa_LRV(Exp_7B_One_Stage):
    model_id: str = "llava-lrv+7b"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Exp_7B_LLaVa_LVIS4V_LRV(Exp_7B_One_Stage):
    model_id: str = "llava-lvis4v-lrv+7b"


# === Section 5 :: Prisms ===


# Prism-CLIP
@dataclass
class Prism_7B_CLIP_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-clip-controlled+7b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"


@dataclass
class Prism_13B_CLIP_Controlled(Exp_13B_One_Stage):
    model_id: str = "prism-clip-controlled+13b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_CLIP(Exp_7B_One_Stage):
    model_id: str = "prism-clip+7b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    finetune_epochs: int = 2


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_13B_CLIP(Exp_13B_One_Stage):
    model_id: str = "prism-clip+13b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    finetune_epochs: int = 2


# Prism-SigLIP
@dataclass
class Prism_7B_SigLIP_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-siglip-controlled+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"


@dataclass
class Prism_13B_SigLIP_Controlled(Exp_13B_One_Stage):
    model_id: str = "prism-siglip-controlled+13b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_SigLIP(Exp_7B_One_Stage):
    model_id: str = "prism-siglip+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    finetune_epochs: int = 2


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_13B_SigLIP(Exp_13B_One_Stage):
    model_id: str = "prism-siglip+13b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    finetune_epochs: int = 2


# Prism-DINOSigLIP
@dataclass
class Prism_7B_DINOSigLIP_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip-controlled+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"


@dataclass
class Prism_13B_DINOSigLIP_Controlled(Exp_13B_One_Stage):
    model_id: str = "prism-dinosiglip-controlled+13b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_DINOSigLIP(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_13B_DINOSigLIP(Exp_13B_One_Stage):
    model_id: str = "prism-dinosiglip+13b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2


# [Inference-Optimized] 224px Prism Models
@dataclass
class Prism_7B_DINOSigLIP_224px_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip-224px-controlled+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_DINOSigLIP_224px(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip-224px+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2


# Added the new Moxin LLM Backbone here
@dataclass
class Prism_Moxin_7B_DINOSigLIP_224px(Exp_7B_One_Stage):
    model_id: str = "prism-moxin-dinosiglip-224px+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "moxin-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2

# === Define a Model Registry Enum for Reference & Validation ===
@unique
class ModelRegistry(Enum):
    # === LLaVa v1.5 Base Reproductions ===
    REPRODUCTION_7B = LLaVa_v15_Reproduction_7B
    REPRODUCTION_13B = LLaVa_v15_Reproduction_13B

    # === Section 4.1 :: Optimization Procedure ===
    EXP_ONE_STAGE_7B = Exp_7B_One_Stage
    EXP_ONE_STAGE_13B = Exp_13B_One_Stage

    EXP_FULL_FT_MULTI_STAGE = Exp_7B_Full_Finetune_Multi_Stage
    EXP_FULL_FT_ONE_STAGE = Exp_7B_Full_Finetune_One_Stage

    # === Section 4.2 :: Image Processing and Visual Representations ===
    EXP_IN1K_224PX = Exp_7B_IN1K_ViT_L_p16_224px
    EXP_DINOV2_224PX = Exp_7B_DINOv2_ViT_L_p14_224px
    EXP_CLIP_224PX = Exp_7B_CLIP_ViT_L_p14_224px
    EXP_SIGLIP_224PX = Exp_7B_SigLIP_ViT_SO_p14_224px

    EXP_CLIP_336PX_RESIZE_CROP = Exp_7B_CLIP_ViT_L_p14_336px_Resize_Crop
    EXP_CLIP_336PX_RESIZE_NAIVE = Exp_7B_CLIP_ViT_L_p14_336px_Resize_Naive
    EXP_SIGLIP_384PX_LETTERBOX = Exp_7B_SigLIP_ViT_SO_p14_384px_Letterbox
    EXP_SIGLIP_384PX_RESIZE_CROP = Exp_7B_SigLIP_ViT_SO_p14_384px_Resize_Crop
    EXP_SIGLIP_384PX_RESIZE_NAIVE = Exp_7B_SigLIP_ViT_SO_p14_384px_Resize_Naive

    EXP_DINOCLIP_336PX_LETTERBOX = Exp_7B_DINOCLIP_ViT_L_p14_336px_Letterbox
    EXP_DINOCLIP_336PX_RESIZE_NAIVE = Exp_7B_DINOCLIP_ViT_L_p14_336px_Resize_Naive
    EXP_DINOSIGLIP_384PX_LETTERBOX = Exp_7B_DINOSigLIP_ViT_L_p14_384px_Letterbox
    EXP_DINOSIGLIP_384PX_RESIZE_NAIVE = Exp_7B_DINOSigLIP_ViT_L_p14_384px_Resize_Naive

    # === Section 4.3 :: Language Models ===
    EXP_LLAMA2_7B = Exp_7B_Llama2
    EXP_LLAMA2_13B = Exp_13B_Llama2

    # ~ Additional LLM Backbone Experiments :: LLaMa-2 Chat, Mistral v0.1, Mistral v0.1 Instruct, Phi-2 ~
    EXT_EXP_LLAMA2_CHAT_7B = Ext_Exp_7B_Llama2_Chat
    EXT_EXP_LLAMA2_CHAT_13B = Ext_Exp_13B_Llama2_Chat
    EXT_EXP_MISTRAL_V1_7B = Ext_Exp_7B_Mistral_V1
    EXT_EXP_MISTRAL_INSTRUCT_V1_7B = Ext_Exp_7B_Mistral_Instruct_V1
    EXT_EXP_MOXIN_7B = Ext_Exp_7B_Moxin
    EXT_EXP_PHI_2_3B = Ext_Exp_3B_Phi_2

    # Cotraining w/ Unimodal Data

    EXP_VICUNA_NO_COTRAINING_7B = Exp_7B_Vicuna_No_Cotraining
    EXP_LLAMA2_NO_COTRAINING_7B = Exp_7B_Llama2_No_Cotraining

    # === Section 4.4 :: Scaling Properties - Train Time & Data ===
    EXP_1P25_EPOCHS = Exp_7B_1p25_Epochs
    EXP_1P5_EPOCHS = Exp_7B_1p5_Epochs
    EXP_2_EPOCHS = Exp_7B_2_Epochs
    EXP_3_EPOCHS = Exp_7B_3_Epochs

    EXP_LLAVA_LVIS4V = Exp_7B_LLaVa_LVIS4V
    EXP_LLAVA_LRV = Exp_7B_LLaVa_LRV
    EXP_LLAVA_LVIS4V_LRV = Exp_7B_LLaVa_LVIS4V_LRV

    # === Section 5 :: Prisms ===
    PRISM_CLIP_CONTROLLED_7B = Prism_7B_CLIP_Controlled
    PRISM_CLIP_CONTROLLED_13B = Prism_13B_CLIP_Controlled
    PRISM_CLIP_7B = Prism_7B_CLIP
    PRISM_CLIP_13B = Prism_13B_CLIP

    PRISM_SIGLIP_CONTROLLED_7B = Prism_7B_SigLIP_Controlled
    PRISM_SIGLIP_CONTROLLED_13B = Prism_13B_SigLIP_Controlled
    PRISM_SIGLIP_7B = Prism_7B_SigLIP
    PRISM_SIGLIP_13B = Prism_13B_SigLIP

    PRISM_DINOSIGLIP_CONTROLLED_7B = Prism_7B_DINOSigLIP_Controlled
    PRISM_DINOSIGLIP_CONTROLLED_13B = Prism_13B_DINOSigLIP_Controlled
    PRISM_DINOSIGLIP_7B = Prism_7B_DINOSigLIP
    PRISM_DINOSIGLIP_13B = Prism_13B_DINOSigLIP

    # === Inference Optimized :: 224px Prism Models ===
    PRISM_DINOSIGLIP_224PX_CONTROLLED_7B = Prism_7B_DINOSigLIP_224px_Controlled
    PRISM_DINOSIGLIP_224PX_7B = Prism_7B_DINOSigLIP_224px
    
    # Qwen
    PRISM_QWEN25_DINOSIGLIP_224PX_0_5B = Prism_Qwen25_0_5B_DINOSigLIP_224px
    PRISM_QWEN25_EXTRA_DINOSIGLIP_224PX_0_5B = Prism_Qwen25_0_5B_Extra_DINOSigLIP_224px
    PRISM_QWEN25_EXTRA_DINOSIGLIP_224PX_1_5B = Prism_Qwen25_1_5B_Extra_DINOSigLIP_224px

    PRISM_QWEN3_EXTRA_DINOSIGLIP_224PX_0_6B = Prism_Qwen3_0_6B_Extra_DINOSigLIP_224px

    # Moxin
    PRISM_MOXIN_DINOSIGLIP_224PX_7B = Prism_Moxin_7B_DINOSigLIP_224px

    @property
    def model_id(self) -> str:
        return self.value.model_id


# Register Models in Choice Registry
for model_variant in ModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)
