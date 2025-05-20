import copy
import os
import datetime
from argparse import ArgumentParser
from pathlib import Path
from clip_maple import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import step_lr_schedule, print_write, set_seed, accuracy
from utils.dataset_loader_for_others import divide_labeled_or_not, load_datasets
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.cuda.amp import autocast, GradScaler

_tokenizer = _Tokenizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str,
                        choices=(
                            "CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"),
                        default="caltech-101")
    parser.add_argument("--img-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--output-dir", type=str, default="./outputs_maple")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    # number of context vectors
    parser.add_argument("--n-ctx", type=int, default=2)
    parser.add_argument("--prompt-depth", type=int, default=1)  # [1,12] 3 9
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
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        # ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.prompt_depth  # max=12, but will create 11 such shared prompts
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if ctx_init and (n_ctx) <= 4:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = n_ctx
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        #     prompt_prefix = ctx_init
        # else:
        #     # random initialization
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 1024)
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
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

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(
            self.ctx), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_datasets(args):
    labeled_dataset, unlabeled_dataset = divide_labeled_or_not(dataset=args.dataset, input_size=args.img_size)
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader, info = load_datasets(
        dataset=args.dataset,
        pattern="val",
        img_size=args.img_size,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers
    )
    taglist = info["taglist"]
    return labeled_loader, test_loader, taglist


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

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

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

    clip_model = load_clip_to_cpu(args)
    clip_model.float()

    for params in clip_model.parameters():
        params.requires_grad = False

    model = CustomCLIP(args, taglist, clip_model)
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    clip_model.to(device).eval()
    model.to(device)

    optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=1e-3, weight_decay=0.8)

    scaler = GradScaler()

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    best_top5 = 0
    for epoch in range(args.epochs):
        step_lr_schedule(optimizer, epoch, init_lr=8e-2, min_lr=5e-4, decay_rate=0.01)
        torch.cuda.empty_cache()
        model.train()
        clip_model.eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for (imgs, labels, _) in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = model(imgs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        top1, top5 = test_model(model, test_loader, taglist)
        if top1[0] >= best_top1:
            best_top1 = top1[0]
            save_obj = {
                'MaPLe_prompt_learner': model.prompt_learner.state_dict(),
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
