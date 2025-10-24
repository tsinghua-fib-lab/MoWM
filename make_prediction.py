import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import json

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL, UNetSpatioTemporalConditionModel
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

# from diffusers.models.attention_processor import AttnProcessor2_0, Attention
# from diffusers.models.attention import BasicTransformerBlock
# from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)
from einops import rearrange, repeat
import imageio
from video_models.pipeline import (
    MaskStableVideoDiffusionPipeline,
    TextStableVideoDiffusionPipeline,
)
import wandb
from decord import VideoReader, cpu
import decord


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def encode_text(
    texts,
    tokenizer,
    text_encoder,
    img_cond=None,
    img_cond_mask=None,
    image_encoder=None,
    position_encode=True,
    use_clip=True,
    args=None,
):
    max_length = args.clip_token_length
    with torch.no_grad():
        if use_clip:
            inputs = tokenizer(
                texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(text_encoder.device)
            outputs = text_encoder(**inputs)
            encoder_hidden_states = outputs.last_hidden_state  # (batch, 30, 512)
            if position_encode:
                embed_dim, pos_num = (
                    encoder_hidden_states.shape[-1],
                    encoder_hidden_states.shape[1],
                )
                pos = np.arange(pos_num, dtype=np.float64)

                position_encode = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
                position_encode = torch.tensor(
                    position_encode,
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                    requires_grad=False,
                )

                encoder_hidden_states += position_encode
            assert encoder_hidden_states.shape[-1] == 512

            if image_encoder is not None:
                assert img_cond is not None
                assert img_cond_mask is not None
                img_cond = img_cond.to(image_encoder.device)
                if len(img_cond.shape) == 5:
                    img_cond = img_cond.squeeze(1)

                img_hidden_states = image_encoder(img_cond).image_embeds
                img_hidden_states[img_cond_mask] = 0.0
                img_hidden_states = img_hidden_states.unsqueeze(1).expand(
                    -1, encoder_hidden_states.shape[1], -1
                )
                assert img_hidden_states.shape[-1] == 512
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, img_hidden_states], dim=-1
                )
                assert encoder_hidden_states.shape[-1] == 1024
            else:
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, encoder_hidden_states], dim=-1
                )

        else:
            inputs = tokenizer(
                texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                max_length=32,
            ).to(text_encoder.device)
            outputs = text_encoder(**inputs)
            encoder_hidden_states = outputs.last_hidden_state  # (batch, 30, 512)
            assert encoder_hidden_states.shape[1:] == (32, 1024)

    return encoder_hidden_states


def generate_video_chunk(
    pipeline,
    start_frame_tensor,
    text_token,
    img_cond,
    img_cond_mask,
    image_encoder,
    args,
):
    """
    生成一个视频片段.
    """
    with torch.no_grad():
        videos = MaskStableVideoDiffusionPipeline.__call__(
            pipeline,
            image=start_frame_tensor,
            text=text_token,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=16,
            decode_chunk_size=args.decode_chunk_size,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
        ).frames

    # 返回 PIL Image 格式的帧列表
    return videos[0]


def eval(
    pipeline,
    text,
    tokenizer,
    text_encoder,
    initial_image,
    img_cond,
    img_cond_mask,
    image_encoder,
    args,
    pretrained_model_path,
    preprocess,
    preprocess_clip,
    scene_name=None,
):

    rollout_steps = 10  # <--- 在这里设置你想要的 rollout 步数
    all_video_frames = []  # 用于存储所有生成的帧

    # 1. 对初始文本进行一次编码，后续重复使用
    with torch.no_grad():
        print("Encoding text prompt...")
        text_token = encode_text(
            text,
            tokenizer,
            text_encoder,
            img_cond=img_cond,
            img_cond_mask=img_cond_mask,
            image_encoder=image_encoder,
            position_encode=args.position_encode,
            args=args,
        )

    # 2. 设置初始条件
    current_start_frame_tensor = initial_image

    # 3. 开始自回归循环
    print(f"Starting autoregressive rollout for {rollout_steps} steps...")
    for i in tqdm(range(rollout_steps), desc="Rollout Steps"):
        print(f"Step {i+1}/{rollout_steps}")

        # 调用独立的生成函数
        generated_frames_pil = generate_video_chunk(
            pipeline,
            current_start_frame_tensor,
            text_token,
            img_cond,
            img_cond_mask,
            image_encoder,
            args,
        )

        # 将生成的帧（除了最后一帧）添加到总列表中
        # 最后一帧将作为下一次生成的起点
        all_video_frames.extend(generated_frames_pil[:-1])

        # 获取最后一帧作为下一次的输入
        last_frame_pil = generated_frames_pil[-1]

        # 如果这是最后一步，把最后一帧也加上
        if i == rollout_steps - 1:
            all_video_frames.append(last_frame_pil)
            break

        # 4. 预处理最后一帧，为下一次迭代做准备
        last_frame_np = np.array(last_frame_pil)
        vf_tensor = torch.from_numpy(np.array([last_frame_np])).permute(0, 3, 1, 2)

        # 使用与初始帧相同的预处理流程
        current_start_frame_tensor = preprocess(vf_tensor).to(pipeline.device)
        # （可选）你也可以在这里更新 img_cond，如果需要的话
        # img_cond = preprocess_clip(vf_tensor).to(pipeline.device).unsqueeze(1)

    # 5. 保存最终的长视频 -> 改为保存一系列 PNG 帧
    print("Rollout complete. Saving frames as PNG...")

    video_frames_np = [np.array(frame) for frame in all_video_frames]

    out_dir = (
        f"./CALVIN/calvin_debug_dataset/calvin_debug_dataset/visualize_svd/{scene_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    for i, frame in enumerate(all_video_frames):  # all_video_frames 里是 PIL.Image
        filename = os.path.join(out_dir, f"frame_{i:04d}.png")
        frame.save(filename)

    print(f"✅ Saved {len(all_video_frames)} frames to {out_dir}")

    return


def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            pretrained_model_path, torch_dtype=torch.float16
        )
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, pipeline.vae, pipeline.unet


def main_eval(
    pretrained_model_path: str,
    clip_model_path: str,
    args: Dict,
    seed: Optional[int] = None,
):
    if seed is not None:
        set_seed(seed)
    # Load scheduler, tokenizer and models.
    pipeline, _, _ = load_primary_models(pretrained_model_path, eval=True)
    device = torch.device("cuda")
    pipeline.to(device)
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path, use_fast=False)
    text_encoder.requires_grad_(False).to(device)

    from video_dataset.video_transforms import Resize_Preprocess, ToTensorVideo

    preprocess = T.Compose(
        [
            ToTensorVideo(),
            Resize_Preprocess((256, 256)),  # 288 512
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    image_encoder = None
    if True:
        # load image encoder
        from transformers import CLIPVisionModelWithProjection

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
        image_encoder.requires_grad_(False)
        image_encoder.to(device)

        preprocess_clip = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(
                    tuple([args.clip_img_size, args.clip_img_size])
                ),  # 224,224
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    inplace=True,
                ),
            ]
        )
        print("use image condition")

    scene_name = "pick_up_the_red_block_from_the_table"
    text_prompt = scene_name.replace("_", " ")
    image_path = (
        f"./CALVIN/calvin_debug_dataset/calvin_debug_dataset/visualize/{scene_name}.png"
    )
    start_frame = Image.open(image_path).convert("RGB")
    vf = [np.array(start_frame)]
    vf_tensor = torch.from_numpy(np.array(vf)).permute(0, 3, 1, 2)  # (1,3,256,256)
    image_for_svd = preprocess(vf_tensor).to(device)  # (1,3,256,256)
    img_cond = preprocess_clip(vf_tensor).to(device)  # (1,3,224,224)
    img_cond = img_cond.unsqueeze(1)  # (1,1,3,224,224)
    img_cond_mask = torch.tensor([True]).to(device)

    eval(
        pipeline,
        [text_prompt],
        tokenizer,
        text_encoder,
        image_for_svd,
        img_cond,
        img_cond_mask,
        image_encoder,
        args,
        pretrained_model_path,
        preprocess,
        preprocess_clip,
        scene_name=scene_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="video_conf/val_svd.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--video_model_path",
        type=str,
        default="./checkpoints/svd-robot-calvin-ft",
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default="./checkpoints/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--val_dataset_dir", type=str, default="video_dataset_instance/bridge"
    )
    parser.add_argument("--val_idx", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    val_args = args_dict.validation_args
    val_args.val_dataset_dir = args.val_dataset_dir
    val_args.num_inference_steps = args.num_inference_steps

    if args.val_idx is not None:
        idxs = args.val_idx.split("+")
        idxs = [int(idx) for idx in idxs]
        val_args.val_idx = idxs

    main_eval(
        pretrained_model_path=args.video_model_path,
        clip_model_path=args.clip_model_path,
        args=val_args,
    )
