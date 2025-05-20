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
    parser.add_argument("--dataset", type=str,
                        choices=(
                            "CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"),
                        default="caltech-101")
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--output-dir", type=str, default="./outputs_cross_dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    # number of context vectors
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--image-prompt-m", type=float, default=0.1)
    parser.add_argument("--text-prompt-m", type=float, default=0)
    parser.add_argument("--image-adapter-m", type=float, default=0)
    parser.add_argument("--text-adapter-m", type=float, default=0.2)
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

def get_text_prompts_embedding(classnames, args):
    template = "a photo of a {}."
    processed_names = [processed_name(name, rm_dot=True) for name in classnames]
    if "a {}" in template:
        prompts = [
            template.replace("a {}", f"{get_article(name)} {{}}", 1).format(pname)
            for name, pname in zip(taglist, processed_names)
        ]
    else:
        prompts = [template.format(pname) for pname in processed_names]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    tokenized_label_prompts_embedding = clip_model.token_embedding(tokenized_prompts)

    # Initializing class-specific contexts
    # init_txt_vector = torch.empty(num_classes, 77, device=device, requires_grad=False)
    # nn.init.normal_(init_txt_vector, std=0.02)
    text_prompt_prefix = " ".join(["X"] * args.n_ctx)
    text_prompt_vector = [text_prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
    text_prompt_vector = torch.cat([clip.tokenize(p) for p in text_prompt_vector]).to(device)
    text_prompt_embedding = clip_model.token_embedding(text_prompt_vector)

    return tokenized_label_prompts_embedding, text_prompt_embedding, tokenized_prompts

def text_prompt_fusion(label_embedding, text_embedding, args):
    fused_embedding = label_embedding.clone()
    fused_embedding[:, 1:args.n_ctx, :] = ((1 - args.text_prompt_m) * fused_embedding[:, 1:args.n_ctx, :]
                                           + args.text_prompt_m * text_embedding[:, 1:args.n_ctx, :])
    return fused_embedding


def get_llm_prompt_features(clip_model, classnames, datasets):
    llm_prompt_features = []
    with open(f"./llm_prompt/{datasets}.json", 'r') as file:
        llm_prompt_dict = json.load(file)
    llm_prompt_dict = {k.lower().replace("_", " "): v for k, v in llm_prompt_dict.items()}
    for single_key in classnames:
        single_class_prompts = llm_prompt_dict[single_key.lower().replace("_", " ")]
        x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
        with torch.no_grad():
            llm_features = clip_model.encode_text(x_tokenized.cuda())
        llm_features = llm_features / llm_features.norm(dim=-1, keepdim=True)
        llm_prompt_features.append(llm_features.mean(0).unsqueeze(0))
    llm_prompt_features = torch.cat(llm_prompt_features, dim=0)
    llm_prompt_features = llm_prompt_features / llm_prompt_features.norm(dim=-1, keepdim=True)
    return llm_prompt_features


@torch.no_grad()
def test_model(args, text_prompt_network, vision_prompt_network, text_adapter, vision_adapter,
               clip_model, text_encoder, test_loader, taglist,
               init_img_mat, init_text_prompt_embedding, tokenized_label_prompts_embedding, tokenized_prompts):
    clip_model.eval()
    text_prompt_network.eval()
    vision_prompt_network.eval()
    text_adapter.eval()
    vision_adapter.eval()

    num_classes = len(taglist)
    # inference
    final_logits = torch.empty(len(test_loader.dataset), num_classes)
    # final_logits = torch.empty(len(test_loader.dataset))
    targs = torch.empty(len(test_loader.dataset))
    pos = 0

    # (nun_cls,77,768)
    text_prompt_embedding = text_prompt_network(init_text_prompt_embedding)
    text_prompt = text_prompt_fusion(tokenized_label_prompts_embedding, text_prompt_embedding, args)
    # (100, 768)
    text_embedding = text_encoder(text_prompt, tokenized_prompts)
    text_features_a = text_adapter(text_embedding)
    text_features = args.text_adapter_m * text_features_a + (1 - args.text_adapter_m) * text_embedding
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        bs = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        vision_prompt_embedding = vision_prompt_network(init_img_mat).repeat(bs, 1, 1, 1)
        imgs = (1 - args.image_prompt_m) * imgs + args.image_prompt_m * vision_prompt_embedding
        imgs_embedding = clip_model.encode_image(imgs)
        imgs_features_a = vision_adapter(imgs_embedding)
        imgs_features = args.image_adapter_m * imgs_features_a + (1 - args.image_adapter_m) * imgs_embedding
        imgs_features = imgs_features / imgs_features.norm(dim=-1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * imgs_features @ text_features.t()
        output = logits_per_image.softmax(dim=-1)

        final_logits[pos:pos + bs, :] = output.cpu()
        targs[pos:pos + bs] = labels.cpu()
        pos += bs
    # evaluate and record
    top1, top5 = accuracy(final_logits, targs, topk=(1, 5))
    return top1, top5


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # set up output paths
    # output_dir = args.output_dir + "/" + args.method + "/" + "Test-" + str(datetime.datetime.now().date())
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    # summary_file = output_dir + "/" + "summary.txt"
    # with open(summary_file, "w", encoding="utf-8") as f:
    #     print_write(f, "****************")
    #     for key in (
    #             "dataset", "img_size",
    #             "output_dir", "batch_size", "num_workers"
    #     ):
    #         print_write(f, f"{key}: {getattr(args, key)}")
    #     print_write(f, "****************")
    #
    # train_loader, test_loader, taglist = get_datasets(args)

    # "CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"
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
    clip_model.to(device).eval()
    for params in clip_model.parameters():
        params.requires_grad = False
    text_encoder = TextEncoder(clip_model)
    # (100, 768)
    llm_prompt_features = get_llm_prompt_features(clip_model, taglist, datasets)

    with torch.no_grad():
        tokenized_label_prompts_embedding, init_text_prompt_embedding, tokenized_prompts = get_text_prompts_embedding(
            taglist, args)
    tokenized_label_prompts_embedding = tokenized_label_prompts_embedding.detach().requires_grad_(False)
    init_text_prompt_embedding = init_text_prompt_embedding.detach().requires_grad_(False)
    tokenized_prompts = tokenized_prompts.detach().requires_grad_(False)

    text_prompt_network = TextPromptNetwork().to(device)

    init_img_mat = torch.empty(1, 1, args.img_size, args.img_size, device=device, requires_grad=False)
    nn.init.normal_(init_img_mat, std=0.02)
    vision_prompt_network = VisionPromptNetwork().to(device)

    text_adapter = Adapter(c_in=768, reduction=4).to(device)
    vision_adapter = Adapter(c_in=768, reduction=4).to(device)

    for params in text_prompt_network.parameters():
        params.requires_grad = False
    for params in vision_prompt_network.parameters():
        params.requires_grad = False
    for params in text_adapter.parameters():
        params.requires_grad = False
    for params in vision_adapter.parameters():
        params.requires_grad = False

    checkpoint_path = "./outputs/tiny-imagenet-200/Train-2025-03-16/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    text_adapter.load_state_dict(checkpoint['text_adapter'])
    vision_adapter.load_state_dict(checkpoint['vision_adapter'])
    text_prompt_network.load_state_dict(checkpoint['text_prompt_network'])
    vision_prompt_network.load_state_dict(checkpoint['vision_prompt_network'])

    top1, top5 = 0.0, 0.0
    top1, top5 = test_model(args, text_prompt_network, vision_prompt_network, text_adapter, vision_adapter,
                            clip_model, text_encoder, test_loader, taglist,
                            init_img_mat, init_text_prompt_embedding,
                            tokenized_label_prompts_embedding, tokenized_prompts)
    print(top1[0], top5[0])

    # with open(summary_file, "a", encoding="utf-8") as f:
    #     print_write(f,
    #                 f"Epoch : {epoch + 1}  | top1 : {top1[0]}  | top5 : {top5[0]} | best_top1 : {best_top1}| "
    #                 f"best_top5 : {best_top5}"
    #                 )
