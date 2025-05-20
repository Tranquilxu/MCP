import json
import math
import os
import datetime
from argparse import ArgumentParser
from pathlib import Path
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import step_lr_schedule, print_write, set_seed, processed_name, get_article, accuracy
from dataset_loader import load_datasets
from model_structure import TextPromptNetwork, VisionPromptNetwork, Adapter
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.cuda.amp import autocast, GradScaler

_tokenizer = _Tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    # number of context vectors
    parser.add_argument("--n-ctx", type=int, default=16)
    args = parser.parse_args()
    return args


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        print("Initializing a generic context")
        ctx_vectors = torch.empty(args.n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * args.n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {args.n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@torch.no_grad()
def test_model(model, test_loader, taglist):
    model.eval()

    num_classes = len(taglist)
    # inference
    final_logits = torch.empty(len(test_loader.dataset), num_classes)
    # final_logits = torch.empty(len(test_loader.dataset))
    targs = torch.empty(len(test_loader.dataset))
    pos = 0

    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        bs = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        output = model(imgs)[:, :num_classes].softmax(dim=-1)

        final_logits[pos:pos + bs, :] = output.cpu()
        targs[pos:pos + bs] = labels.cpu()
        pos += bs
    # evaluate and record
    top1, top5 = accuracy(final_logits, targs, topk=(1, 5))
    return top1, top5


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # "CIFAR100", "stanford_cars", "caltech-101", "food-101"
    datasets = "food-101"
    test_loader, info = load_datasets(
        dataset=datasets,
        pattern="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist = info["taglist"]
    num_classes = len(taglist)

    clip_model, _ = clip.load("ViT-L/14", device="cpu")
    for params in clip_model.parameters():
        params.requires_grad = False
    # (100, 768)

    model = CustomCLIP(args, taglist, clip_model)
    clip_model.to(device).eval()
    model.to(device)

    for params in model.parameters():
        params.requires_grad = False

    checkpoint_path = "./outputs_coop/tiny-imagenet-200/Train-2025-03-25/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["CoOp_prompt_learner"]

    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    model.prompt_learner.load_state_dict(state_dict, strict=False)


    top1, top5 = 0.0, 0.0
    top1, top5 = test_model(model, test_loader, taglist)
    print(top1[0])
