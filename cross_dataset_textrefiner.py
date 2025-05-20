from argparse import ArgumentParser
from clip_textrefiner import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import set_seed, accuracy
from dataset_loader import load_datasets
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
import copy

_tokenizer = _Tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    # number of context vectors
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument(
        "--balance", type=float, default=50, help="balance factor for semantic loss"
    )
    parser.add_argument(
        "--distill", type=float, default=20, help="balance factor for regularization loss"
    )
    parser.add_argument(
        "--memory-size", type=int, default=25, help="the size of memory"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="the factor to aggregate two feature"
    )
    args = parser.parse_args()
    return args


class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )

    def forward(self, input_feat):  # input_feat:[B, d] [B, N, d]

        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))

        return final_feat.squeeze(-1).squeeze(-1)


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
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    # zero-shot clip
    def encode_text(self, text):
        text = text.cuda()
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


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
        self.register_buffer("tokenized_prompts", tokenized_prompts)

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

        self.mid_image_trans = Feature_Trans_Module_two_layer(clip_model.ln_final.weight.shape[0],
                                                              clip_model.ln_final.weight.shape[0])
        self.mid_image_trans = self.mid_image_trans.cuda().float()
        # convert_weights(self.mid_image_trans)

    def forward(self, image):
        image_features, image_fine = self.image_encoder(image.type(self.dtype), all_layer_outputs=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        image_fine_list = []
        top_image_fine_list = []
        B, d = image_features.shape
        image_fine_features = [x[0] for x in image_fine]  # B,N,d
        image_fine_attns = [x[1] for x in image_fine]  # B,1,N
        layers = [-1]
        _, _, before_d = image_fine_features[0].shape
        loss = 0.0
        for i, layer in enumerate(layers):
            x = image_fine_features[layer]
            x = x.reshape(-1, before_d)

            # if self.training:
            x = self.mid_image_trans(x)
            x = x.reshape(B, -1, d)

            image_fine_feature = x
            image_fine_list.append(image_fine_feature)

            k = 5
            _, indices = torch.topk(image_fine_attns[layer], k=k, dim=-1)
            indices += 1
            indices = torch.cat((torch.zeros(B, 1, dtype=torch.int64).cuda(), indices), dim=1)
            top_image_fine_feature = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(B, k + 1, d))
            avg_image_feature = torch.mean(x, dim=1, keepdim=True)
            top_image_fine_feature = torch.cat((top_image_fine_feature, avg_image_feature), dim=1)

            top_image_fine_list.append(top_image_fine_feature.reshape(-1, d))

        if len(image_fine_list) > 0:
            image_fine_list = torch.cat(image_fine_list)
            top_image_fine_list = torch.cat(top_image_fine_list)

        return text_features, image_features, logit_scale, image_fine_list, top_image_fine_list


class Memory(nn.Module):
    def __init__(self, clip_model, feature_dim=768, memory_size=25, reduction=4, frozen_text_embedding=None, alpha=0.2,
                 momentum=0.8):
        super().__init__()
        self.device = clip_model.dtype
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.text_fine_cache = F.normalize(torch.rand(self.memory_size, feature_dim), dim=-1)
        self.text_fine_cache = self.text_fine_cache.to(self.device)
        self.text_fine_cache = self.text_fine_cache.cuda()
        self.alpha = alpha
        self.momentum = momentum

        if frozen_text_embedding is not None:
            self.frozen_text_embedding = frozen_text_embedding

        self.extractor = nn.Linear(2 * feature_dim, feature_dim, bias=False)
        self.extractor = self.extractor.to(self.device)
        self.extractor = self.extractor.cuda()

        self.writeTF = lambda x: x.clone()

    def forward(self, text_token=None, image_token=None):
        fine_feature = self.read(text_token)

        text_fine_feature = torch.cat((text_token, fine_feature), dim=-1)
        text_fine_feature = self.alpha * self.extractor(text_fine_feature) + text_token
        if self.training:
            _ = self.write(image_token)
            normalized_text_features = F.normalize(text_fine_feature, dim=-1)
            loss = F.l1_loss(normalized_text_features, text_token, reduction='mean')
        else:
            loss = 0.0
        return text_fine_feature, loss

    def get_score(self, query, mem):
        score = query @ mem.t()
        score_query = F.softmax(score, dim=0)
        score_mem = F.softmax(score, dim=1)
        return score_query, score_mem

    def read(self, x):
        base_features = F.normalize(x, dim=-1)
        C, d = x.shape
        if self.training:
            self.text_fine_cache = self.text_fine_cache.detach()
        _, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)
        fine_feature = softmax_score_cache @ self.text_fine_cache  # (N, d)

        return fine_feature

    def write(self, x):
        m, d = self.text_fine_cache.shape
        ratio = 0.2
        base_features = x.clone()
        base_features = self.writeTF(base_features)

        base_features = base_features.reshape(-1, d)  # (B * P, d)
        base_features = F.normalize(base_features, dim=-1)

        softmax_score_query, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)  # (B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)

        updated_cache = self.text_fine_cache.clone().detach()
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(
                    score * base_features[idx.squeeze(1)], dim=0)

        updated_cache = F.normalize(updated_cache, dim=-1)

        loss = 0.0
        self.text_fine_cache = updated_cache.to(self.device)
        return loss

    def diversityloss(self, mem):
        # it is same with orthonomal constraints.
        cos_sim = torch.matmul(mem, torch.t(mem))
        margin = 0
        cos_sim_pos = cos_sim - margin
        cos_sim_pos[cos_sim_pos < 0] = 0
        loss = (torch.sum(cos_sim_pos) - torch.trace(cos_sim_pos)) / (self.memory_size * (self.memory_size - 1))
        return loss


@torch.no_grad()
def test_model(model, test_loader, taglist, memory):
    model.eval()
    memory.eval()
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

        text_features, image_features, logit_scale, image_fine, _ = model(imgs)
        fine_text_features, _ = memory(text_features, image_fine)

        output = logit_scale * image_features @ fine_text_features.t()
        output = output.softmax(dim=-1)

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

    clip_model, _ = clip.load(args.backbone, device="cpu")
    # clip_model.float()

    for params in clip_model.parameters():
        params.requires_grad = False

    model = CustomCLIP(args, taglist, clip_model)

    print("Building memory")
    memory = Memory(clip_model, feature_dim=clip_model.ln_final.weight.shape[0],
                    memory_size=args.memory_size,
                    alpha=args.alpha)
    memory.eval()
    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" in name or 'mid_image_trans' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(False)

    print("check hyper-parameter balance: ", args.balance)
    print("check hyper-parameter distill: ", args.distill)
    print("check hyper-parameter alpha: ", memory.alpha)
    print("check hyper-parameter momentum: ", memory.momentum)

    clip_model.to(device).eval()
    model.to(device)

    # NOTE: only give prompt_learner to the optimizer
    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(memory)
    # optim = build_optimizer(trainable_list, cfg.OPTIM)
    # sched = build_lr_scheduler(optim, cfg.OPTIM)

    enabled = set()
    for name, param in trainable_list.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    print(f"Parameters count: {len(enabled)}")

    clip_model.to(device).eval()
    model.to(device).eval()

    checkpoint_path = "./outputs_textrefiner/tiny-imagenet-200/Train-2025-04-04/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint["textrefiner_prompt_learner"]

    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    if "tokenized_prompts" in state_dict:
        del state_dict["tokenized_prompts"]

    if "frozen_text_embedding" in state_dict:
        del state_dict["frozen_text_embedding"]

    model.prompt_learner.load_state_dict(state_dict, strict=False)

    state_dict_memory = checkpoint["memory_item"]

    # Ignore abstract layer
    keys = [key for key in state_dict_memory.keys() if 'mid_image_trans' in key]
    state_dict_memory = {k: v for k, v in state_dict_memory.items() if k not in keys}

    if "frozen_text_embedding" in state_dict_memory:
        del state_dict_memory["frozen_text_embedding"]

    # set strict=False
    memory.load_state_dict(state_dict_memory, strict=False)
    # if 'Mem' in name:
    #     self._models[name].text_fine_cache = checkpoint["memory_item"]

    top1, top5 = 0.0, 0.0
    top1, top5 = test_model(model, test_loader, taglist, memory)
    print(top1[0])
