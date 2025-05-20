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
from collections import OrderedDict

_tokenizer = _Tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--batch-size", type=int, default=32)
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
        vis_dim = clip_model.visual.output_dim

        print("Initializing a generic context")
        ctx_vectors = torch.empty(args.n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * args.n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {args.n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        # self.meta_net.half()

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

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
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

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

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

        output = model(imgs).softmax(dim=-1)

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

    model = CustomCLIP(args, taglist, clip_model)
    clip_model.to(device).eval()
    model.to(device)

    for params in model.parameters():
        params.requires_grad = False

    checkpoint_path = "./outputs_cocoop/tiny-imagenet-200/Train-2025-03-25/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["CoCoOp_prompt_learner"]

    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    model.prompt_learner.load_state_dict(state_dict, strict=False)

    top1, top5 = 0.0, 0.0
    top1, top5 = test_model(model, test_loader, taglist)
    print(top1[0])
