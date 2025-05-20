import os
import datetime
from argparse import ArgumentParser
from pathlib import Path
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import step_lr_schedule, print_write, set_seed, accuracy
from utils.dataset_loader_for_others import divide_labeled_or_not, load_datasets
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.cuda.amp import autocast, GradScaler
import copy
from clip_text import clip
from collections import OrderedDict

_tokenizer = _Tokenizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--output-dir", type=str, default="./outputs_tcp")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="ViT-L/14")  # "ViT-L/14""ViT-B/16"
    # number of context vectors
    parser.add_argument("--n-ctx", type=int, default=16)
    args = parser.parse_args()
    return args


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, class_feature, weight, tokenized_prompts, flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            x = self.transformer(x)
        else:
            counter = 0
            outputs = self.transformer.resblocks([x, class_feature, weight, counter])
            x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
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

        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = "a photo of a {}."
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        vis_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 4, bias=True)),
                         ("relu", QuickGELU()),
                         ("linear2", nn.Linear(vis_dim // 4, 4 * ctx_dim, bias=True))
                         ]))
        classnames = [name.replace("_", " ") for name in classnames]
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = "end"
        self.prev_ctx = None
        self.ctx_dim = ctx_dim

    def forward(self):
        class_feature = self.meta_net(self.text_features)
        class_feature = class_feature.reshape(class_feature.shape[0], -1, self.ctx_dim)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompt, class_feature


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = 1.0

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features_old = self.ori_embedding
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, class_prompt = self.prompt_learner()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_encoder(prompts, class_prompt, self.weight, tokenized_prompts.detach())
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale.detach() * image_features.detach() @ text_features_norm.t()

        if self.prompt_learner.training:
            score = cos(text_features_norm, text_features_old)
            score = 1.0 - torch.mean(score)
            loss = F.cross_entropy(logits, label) + 8.0 * score
            return logits, loss
        else:
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
    datasets = "stanford_cars"
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

    checkpoint_path = "./outputs_tcp/tiny-imagenet-200/Train-2025-04-03/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["TCP_prompt_learner"]

    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    if "token_midfix" in state_dict:
        del state_dict["token_midfix"]

    model.prompt_learner.load_state_dict(state_dict, strict=False)
    with autocast():
        top1, top5 = test_model(model, test_loader, taglist)
    print(top1[0])
