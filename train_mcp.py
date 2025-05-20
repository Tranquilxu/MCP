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
from dataset_loader import load_datasets_gened_clip
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
                        default="CIFAR100")
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--output-dir", type=str, default="./outputs_gened_clip")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    # number of context vectors
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--image-prompt-m", type=float, default=0.2)
    parser.add_argument("--text-prompt-m", type=float, default=0.3)
    parser.add_argument("--image-adapter-m", type=float, default=0.4)
    parser.add_argument("--text-adapter-m", type=float, default=0.5)
    parser.add_argument("--llm-weight", type=int, default=10000)
    parser.add_argument("--vision-weight", type=int, default=100)
    parser.add_argument("--max-alpha", type=float, default=0.3)
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


def get_datasets(args):
    train_loader, info = load_datasets_gened_clip(
        dataset=args.dataset,
        pattern="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist = info["taglist"]
    test_loader, _ = load_datasets_gened_clip(
        dataset=args.dataset,
        pattern="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    return train_loader, test_loader, taglist


# def construct_prompts(clip_model, taglist, args):
#     template = CUSTOM_TEMPLATES[args.dataset]
#     ctx_init = template.replace("_", " ").replace(".", "").format("").rstrip()
#     n_ctx = len(ctx_init.split(" "))
#     prompt = clip.tokenize(ctx_init).to(device)
#     with torch.no_grad():
#         embedding = clip_model.token_embedding(prompt).type(clip_model.dtype)
#     ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#
#     processed_names = [processed_name(name, rm_dot=True) for name in taglist]
#     if "a {}" in template:
#         prompts = [
#             template.replace("a {}", f"{get_article(name)} {{}}", 1).format(pname)
#             for name, pname in zip(taglist, processed_names)
#         ]
#     else:
#         prompts = [template.format(pname) for pname in processed_names]
#     tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
#     with torch.no_grad():
#         embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
#     token_prefix = embedding[:, :1, :]  # SOS
#     token_suffix = embedding[:, 1 + n_ctx:, :]  # CLS, EOS
#     if ctx_vectors.dim() == 2:
#         ctx_vectors = ctx_vectors.unsqueeze(0).expand(len(taglist), -1, -1)
#     prompts = torch.cat(
#         [
#             token_prefix,  # (n_cls, 1, dim)
#             ctx_vectors,  # (n_cls, n_ctx, dim)
#             token_suffix,  # (n_cls, *, dim)
#         ],
#         dim=1,
#     )
#     return prompts


def get_text_prompts_embedding(classnames, num_classes, args):
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

def get_intramodal_loss(logits_orig, logits_strong, logits_weak):
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
    # score_orig_strong = cos(logits_orig, logits_strong)
    # score_orig_weak = cos(logits_orig, logits_weak)
    # score_strong_weak = cos(logits_strong, logits_weak)
    # total_score = torch.mean(score_orig_strong) + torch.mean(score_orig_weak) + torch.mean(score_strong_weak)
    # intramodal_loss = 1.0 - total_score / 3

    B = logits_orig.size(0)
    # 将三个视图堆叠为 [3, B, D]
    triplet_emb = torch.stack([logits_orig, logits_strong, logits_weak], dim=0)  # [3, B, D]
    triplet_emb = F.normalize(triplet_emb, p=2, dim=-1)  # L2归一化
    # 计算所有视图对之间的相似度 [3, 3, B]
    sim_matrix = torch.einsum('kbd,lbd->klb', triplet_emb, triplet_emb)  # 爱因斯坦求和
    # 提取有效相似度对（排除自身比较）
    mask = ~torch.eye(3, dtype=torch.bool, device=logits_orig.device)  # 3x3掩码，排除对角线
    valid_sim = sim_matrix[mask].view(3, 3 - 1, B)  # [3, 2, B]
    # 计算每个样本的平均相似度 [B]
    avg_sim_per_sample = valid_sim.mean(dim=(0, 1))  # 对视图对和样本求平均
    # 最终损失
    intramodal_loss = 1.0 - avg_sim_per_sample.mean()
    return intramodal_loss


def vision_process(imgs, imgs_strong, imgs_weak, vision_prompt_embedding, clip_model, vision_adapter, text_features,
                   args):
    imgs = (1 - args.image_prompt_m) * imgs + args.image_prompt_m * vision_prompt_embedding
    imgs_strong = (1 - args.image_prompt_m) * imgs_strong + args.image_prompt_m * vision_prompt_embedding
    imgs_weak = (1 - args.image_prompt_m) * imgs_weak + args.image_prompt_m * vision_prompt_embedding
    imgs_embedding = clip_model.encode_image(imgs)
    imgs_strong_embedding = clip_model.encode_image(imgs_strong)
    imgs_weak_embedding = clip_model.encode_image(imgs_weak)

    imgs_features_a = vision_adapter(imgs_embedding)
    imgs_features = args.image_adapter_m * imgs_features_a + (1 - args.image_adapter_m) * imgs_embedding
    strong_features_a = vision_adapter(imgs_strong_embedding)
    strong_features = args.image_adapter_m * strong_features_a + (1 - args.image_adapter_m) * imgs_strong_embedding
    weak_features_a = vision_adapter(imgs_weak_embedding)
    weak_features = args.image_adapter_m * weak_features_a + (1 - args.image_adapter_m) * imgs_weak_embedding

    imgs_features = imgs_features / imgs_features.norm(dim=-1, keepdim=True)
    # imgs_embedding = imgs_embedding / imgs_embedding.norm(dim=-1, keepdim=True)
    strong_features = strong_features / strong_features.norm(dim=-1, keepdim=True)
    weak_features = weak_features / weak_features.norm(dim=-1, keepdim=True)

    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * imgs_features @ text_features.t()
    logits_strong = logit_scale * strong_features @ text_features.t()
    logits_weak = logit_scale * weak_features @ text_features.t()

    intramodal_loss = get_intramodal_loss(logits, logits_strong, logits_weak)
    return imgs_features, intramodal_loss


def text_prompt_fusion(label_embedding, text_embedding, args):
    fused_embedding = label_embedding.clone()
    fused_embedding[:, 1:args.n_ctx, :] = ((1 - args.text_prompt_m) * fused_embedding[:, 1:args.n_ctx, :]
                                           + args.text_prompt_m * text_embedding[:, 1:args.n_ctx, :])
    return fused_embedding


def get_llm_prompt_features(clip_model, classnames, args):
    llm_prompt_features = []
    with open(f"./llm_prompt/{args.dataset}.json", 'r') as file:
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


def RC_loss(logits_F, labels_F, output, num_classes):
    m_loss = nn.CrossEntropyLoss(reduction='none').to(device)
    loss = 0.0
    for i in range(num_classes):
        value = m_loss(logits_F, torch.ones(labels_F.size()).long().to(device) * i)
        # print("value", value.size())
        # print("output[:, 0]", output[:, i].size())
        loss = loss + torch.inner(value.float(), output[:, i].float())
        # print("loss", loss)
    loss_rf = m_loss(logits_F, labels_F)
    # print("labels_F", labels_F.size())
    loss_tf = 0
    for j in range(len(labels_F)):
        loss_tf = loss_tf + output[j, labels_F[j]].float() * loss_rf[j]
    # print("loss_tf", loss_tf)
    loss = loss - loss_tf
    return loss


def get_rc_loss(imgs, imgs_features, text_features, labels, labels_t, clip_model, epoch, llm_prompt):
    ce_loss = torch.nn.CrossEntropyLoss()

    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * imgs_features @ text_features.t()

    mask_T = torch.where(labels_t == 1)[0].to(device)
    mask_F = torch.where(labels_t == 0)[0].to(device)
    logits_T = torch.index_select(logits, dim=0, index=mask_T)
    logits_F = torch.index_select(logits, dim=0, index=mask_F)
    labels_T = torch.index_select(labels, dim=0, index=mask_T)
    labels_F = torch.index_select(labels, dim=0, index=mask_F)

    imgs_embedding = clip_model.encode_image(imgs).to(device)
    imgs_embedding = imgs_embedding / imgs_embedding.norm(dim=-1, keepdim=True)
    imgs_embedding_F = torch.index_select(imgs_embedding, dim=0, index=mask_F)
    output_zs = logit_scale * imgs_embedding_F @ llm_prompt.t()
    output_zs = output_zs.softmax(dim=-1)

    # tau = torch.ones(output.size()[1]).to(device) * 0.001
    # output = soft(output / tau)
    # tau = torch.ones(logits_F.size()[1]).to(device) * 100000
    # output_f = logits_F.detach()
    # output_f = soft(output_f / tau)

    output_our_F = logits_F.detach()
    output_our_F = output_our_F.softmax(dim=-1)

    # cosine dynamic weighting [0,max_alpha]
    # max_alpha = 0.5
    alpha = (args.max_alpha / 2) * (1 - math.cos(math.pi * epoch / args.epochs))
    output = (1 - alpha) * output_zs + alpha * output_our_F

    if len(logits_T) != 0:
        loss_T = ce_loss(logits_T, labels_T)
    else:
        loss_T = 0
    loss_F = RC_loss(logits_F, labels_F, output, text_features.size()[0])
    loss = loss_T + loss_F
    return loss


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
    output_dir = args.output_dir + "/" + args.dataset + "/" + "Train-" + str(datetime.datetime.now().date())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_file = output_dir + "/" + "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
                "dataset", "img_size",
                "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")

    train_loader, test_loader, taglist = get_datasets(args)
    num_classes = len(taglist)

    clip_model, _ = clip.load("ViT-L/14", device="cpu")
    clip_model.to(device).eval()
    for params in clip_model.parameters():
        params.requires_grad = False
    text_encoder = TextEncoder(clip_model)
    # (100, 768)
    llm_prompt_features = get_llm_prompt_features(clip_model, taglist, args)

    with torch.no_grad():
        tokenized_label_prompts_embedding, init_text_prompt_embedding, tokenized_prompts = get_text_prompts_embedding(
            taglist, num_classes, args)
    tokenized_label_prompts_embedding = tokenized_label_prompts_embedding.detach().requires_grad_(False)
    init_text_prompt_embedding = init_text_prompt_embedding.detach().requires_grad_(False)
    tokenized_prompts = tokenized_prompts.detach().requires_grad_(False)

    # text_prompts_token = construct_prompts(clip_model, taglist, args)

    text_prompt_network = TextPromptNetwork().to(device)

    # total = sum([param.nelement() for param in text_prompt_network.parameters()])
    # print('Number of parameter: ', total)

    init_img_mat = torch.empty(1, 1, args.img_size, args.img_size, device=device, requires_grad=False)
    nn.init.normal_(init_img_mat, std=0.02)
    vision_prompt_network = VisionPromptNetwork().to(device)

    text_adapter = Adapter(c_in=768, reduction=4).to(device)
    vision_adapter = Adapter(c_in=768, reduction=4).to(device)

    for params in text_prompt_network.parameters():
        params.requires_grad = True
    for params in vision_prompt_network.parameters():
        params.requires_grad = True
    for params in text_adapter.parameters():
        params.requires_grad = True
    for params in vision_adapter.parameters():
        params.requires_grad = True

    optimizer_text_prompt = torch.optim.AdamW(text_prompt_network.parameters(), lr=1e-3, weight_decay=0.8)
    optimizer_vision_prompt = torch.optim.AdamW(vision_prompt_network.parameters(), lr=1e-3, weight_decay=0.8)
    optimizer_text_adapter = torch.optim.AdamW(text_adapter.parameters(), lr=1e-3, weight_decay=0.05)
    optimizer_vision_adapter = torch.optim.AdamW(vision_adapter.parameters(), lr=1e-3, weight_decay=0.05)

    llm_mse = torch.nn.MSELoss()
    scaler = GradScaler()  # 混合精度

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    best_top5 = 0
    for epoch in range(args.epochs):
        step_lr_schedule(optimizer_text_prompt, epoch, init_lr=8e-2, min_lr=5e-4, decay_rate=0.01)
        step_lr_schedule(optimizer_vision_prompt, epoch, init_lr=8e-2, min_lr=5e-4, decay_rate=0.01)
        step_lr_schedule(optimizer_text_adapter, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        step_lr_schedule(optimizer_vision_adapter, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        torch.cuda.empty_cache()

        text_prompt_network.train()
        vision_prompt_network.train()
        text_adapter.train()
        vision_adapter.train()
        clip_model.eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for ((imgs, imgs_strong, imgs_weak), labels, labels_t) in pbar:
            imgs, imgs_strong, imgs_weak = imgs.to(device), imgs_strong.to(device), imgs_weak.to(device)
            labels, labels_t = labels.to(device), labels_t.to(device)

            optimizer_text_prompt.zero_grad()
            optimizer_vision_prompt.zero_grad()
            optimizer_text_adapter.zero_grad()
            optimizer_vision_adapter.zero_grad()

            with autocast():
                vision_prompt_embedding = vision_prompt_network(init_img_mat).repeat(imgs.size(0), 1, 1, 1)
                imgs_features, vision_intramodal_loss = vision_process(
                    imgs, imgs_strong, imgs_weak, vision_prompt_embedding, clip_model, vision_adapter,
                    llm_prompt_features, args
                )

                text_prompt_embedding = text_prompt_network(init_text_prompt_embedding)
                text_prompt = text_prompt_fusion(tokenized_label_prompts_embedding, text_prompt_embedding, args)
                text_embedding = text_encoder(text_prompt, tokenized_prompts)
                text_features_a = text_adapter(text_embedding)
                text_features = args.text_adapter_m * text_features_a + (1 - args.text_adapter_m) * text_embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                llm_loss = torch.tensor(0.0) if args.llm_weight == 0.0 else llm_mse(text_features,
                                                                                    llm_prompt_features)

                rc_loss = get_rc_loss(imgs, imgs_features, text_features, labels, labels_t, clip_model, epoch,
                                      llm_prompt_features)
                # rc_loss = rc_loss / 10
                # total_loss = (10 * vision_intramodal_loss + llm_loss + rc_loss) / 3

            # 处理vision_intramodal_loss，仅优化vision_prompt和vision_adapter
            set_requires_grad(text_prompt_network, False)
            set_requires_grad(text_adapter, False)
            set_requires_grad(vision_prompt_network, True)
            set_requires_grad(vision_adapter, True)
            scaler.scale(args.vision_weight * vision_intramodal_loss).backward(retain_graph=True)

            # 处理llm_loss，仅优化text_prompt和text_adapter
            set_requires_grad(text_prompt_network, True)
            set_requires_grad(text_adapter, True)
            set_requires_grad(vision_prompt_network, False)
            set_requires_grad(vision_adapter, False)
            scaler.scale(args.llm_weight * llm_loss).backward(retain_graph=True)

            # 处理rc_loss，优化所有四个模块
            set_requires_grad(text_prompt_network, True)
            set_requires_grad(text_adapter, True)
            set_requires_grad(vision_prompt_network, True)
            set_requires_grad(vision_adapter, True)
            scaler.scale(rc_loss).backward(retain_graph=False)

            scaler.step(optimizer_text_prompt)
            scaler.step(optimizer_vision_prompt)
            scaler.step(optimizer_text_adapter)
            scaler.step(optimizer_vision_adapter)
            scaler.update()

            pbar.set_postfix(
                {"Vision loss": f"{args.vision_weight * vision_intramodal_loss.item():.4f}",
                 "LLM loss": f"{args.llm_weight * llm_loss.item():.2f}", "RC loss": f"{rc_loss.item():.2f}"})

        top1, top5 = test_model(args, text_prompt_network, vision_prompt_network, text_adapter, vision_adapter,
                                clip_model, text_encoder, test_loader, taglist,
                                init_img_mat, init_text_prompt_embedding,
                                tokenized_label_prompts_embedding, tokenized_prompts)
        if top1[0] >= best_top1:
            best_top1 = top1[0]
            save_obj = {
                'text_adapter': text_adapter.state_dict(),
                'vision_adapter': vision_adapter.state_dict(),
                'text_prompt_network': text_prompt_network.state_dict(),
                'vision_prompt_network': vision_prompt_network.state_dict(),
                'epoch': epoch,
                'best_top1': best_top1,
                'top5': top5[0],
                'seed': args.seed
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))
        if top5[0] >= best_top5:
            best_top5 = top5[0]

        with open(summary_file, "a", encoding="utf-8") as f:
            print_write(f,
                        f"Epoch : {epoch + 1}  | top1 : {top1[0]}  | top5 : {top5[0]} | best_top1 : {best_top1}| "
                        f"best_top5 : {best_top5}"
                        )
