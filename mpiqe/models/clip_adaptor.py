import os

import torch
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip import clip
from torch.nn import functional as F
from torch.nn import Dropout
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

number_map = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
              6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
              12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
              16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
              20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
              60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'}

def get_number(i):
    if i in number_map:
        return number_map[i]
    else:
        tens = (i // 10) * 10
        ones = i % 10
        return number_map[tens]+"-"+number_map[ones]


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


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
        # print(x.shape)
        # print(tokenized_prompts.shape)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x + self.fc(x)
        return x


class ADA_CLIP(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.h = config.DATA.H_RESOLUTION // config.MODEL.VIT.PATCH_SIZE
        self.w = config.DATA.W_RESOLUTION // config.MODEL.VIT.PATCH_SIZE
        clip_model = load_clip_to_cpu(config, self.h, self.w)
        clip_model.to("cuda")
        # self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(config, num_classes, clip_model.dtype, clip_model.token_embedding)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.adaptor = Adapter(768, 4).to(clip_model.dtype)
        self.text_encoder = TextEncoder(clip_model)
        # self.adaptor = Adapter
        # self.logit_scale = clip_model.logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dtype = clip_model.dtype
        # self.adapter = Adapter(512, 4).to(clip_model.dtype)
        self.juery = 8
        bunch_layer = nn.TransformerDecoderLayer(
            d_model=512,
            dropout=0.0,
            nhead=8,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(512 * 4),
            norm_first=True,
        )
        self.bunch_decoder = nn.TransformerDecoder(bunch_layer, num_layers=1)
        self.bunch_embedding = nn.Parameter(torch.randn(1, 8, 512))
        self.heads = nn.Linear(512, 512, bias=False)
        trunc_normal_(self.bunch_embedding, std=0.02)

        # self.num_tokens = config.MODEL.NUM_TOKENS
        # self.prompt_dropout = Dropout(config.MODEL.DROPOUT)
        #
        # # if project the prompt embeddings
        # # if self.prompt_config.PROJECT > -1:
        # #     # only for prepend / add
        prompt_dim = 512
        # self.prompt_proj = nn.Linear(prompt_dim, 768)
        self.encoder_proj = nn.Linear(768, prompt_dim)
        # nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal_(self.encoder_proj.weight, a=0, mode='fan_out')
        # # else:
        # #     prompt_dim = config.hidden_size
        # #     self.prompt_proj = nn.Identity()
        #
        # initiate prompt:
        # patch_size = _pair(config.MODEL.VIT.PATCH_SIZE)
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        # #
        # self.prompt_embeddings = nn.Parameter(torch.zeros(
        #     1, self.num_tokens, prompt_dim))
        # # xavier_uniform initialization
        # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        # #
        # self.depth = config.DEPTH
        # self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #     self.depth, self.num_tokens, prompt_dim))
        # # xavier_uniform initialization
        # nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def image_encode(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.image_encoder.get_embedding(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i in range(12):
            # x = x + self.image_encoder.transformer.resblocks[i].attention(self.image_encoder.transformer.resblocks[i].ln_1(x)) \
            #     + self.adaptor(self.image_encoder.transformer.resblocks[i].ln_1(x))
            # x = x + self.image_encoder.transformer.resblocks[i].mlp(self.image_encoder.transformer.resblocks[i].ln_2(x)) \
            #     + self.adaptor(self.image_encoder.transformer.resblocks[i].ln_2(x))

            x = x + self.image_encoder.transformer.resblocks[i].attention(self.image_encoder.transformer.resblocks[i].ln_1(x))
            x = x + self.adaptor(self.image_encoder.transformer.resblocks[i].mlp(self.image_encoder.transformer.resblocks[i].ln_2(self.adaptor(x))))

        x = x.permute(1, 0, 2)
        x = self.image_encoder.ln_post(x)
        encoded = self.encoder_proj(x)
        return encoded



    def get_text_features(self, label):
        prompts, tokenized_prompts = self.prompt_learner(label)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_features(self, x):
        B = x.shape[0]
        embedding_prompt = self.image_encode(x)
        ref, image_features = embedding_prompt[:, :1, :], embedding_prompt[:, 1:, :]
        bunch_embedding = self.bunch_embedding.expand(B, -1, -1)
        ref = ref.view(B, 1, -1)
        ref = ref.expand(B, self.juery, -1)
        output_embedding = bunch_embedding + ref
        image_features = self.bunch_decoder(output_embedding, image_features)  # B * 6 * 384
        image_features = self.heads(image_features)
        image_features = image_features.mean(dim=1)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, x=None, label=None, get_image=False, get_text=False, get_logit=False, get_all=False, text_features=None):
        if get_text:
            text_features = self.get_text_features(label)
            return text_features

        if get_image:
            image_features = self.get_image_features(x)
            return image_features

        if get_all:
            text_features = self.get_text_features(label)
            image_features = self.get_image_features(x)
            return text_features, image_features

        if get_logit:
            image_features = self.get_image_features(x)
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_image = F.softmax(logits_per_image, dim=1)
            return logits_per_image


def load_clip_to_cpu(config, h, w):
    url = clip._MODELS[config.MODEL.BACKBONE]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h, w)
    return model


# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dtype, token_embedding):
#         super().__init__()
#         # ctx_init = "A photo of X X X X point."
#         ctx_init = "A photo with a quality score of X X X X."
#         ctx_dim = 512
#         # use given words to initialize context vectors
#         ctx_init = ctx_init.replace("_", " ")
#         n_ctx = 7
#
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#         n_cls_ctx = 4
#         cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(cls_vectors, std=0.02)
#         self.cls_ctx = nn.Parameter(cls_vectors)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
#         self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
#         self.num_class = num_class
#         self.n_cls_ctx = n_cls_ctx
#
#     def forward(self, label):
#         cls_ctx = self.cls_ctx[label]
#         b = label.shape[0]
#         prefix = self.token_prefix.expand(b, -1, -1)
#         suffix = self.token_suffix.expand(b, -1, -1)
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 cls_ctx,  # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts


class PromptLearner(nn.Module):
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = ""
        ctx_dim = 512
        n_ctx = config.TRAIN.COOP_N_CTX
        self.n_cls = num_class
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if config.TRAIN.COOP_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(num_class, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [get_number(i) for i in range(num_class)]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts.to('cuda')).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = config.TRAIN.COOP_CLASS_TOKEN_POSITION

    def forward(self, label):
        if self.ctx.dim() == 2:
            ctx = self.ctx
            ctx = ctx.unsqueeze(0).expand(len(label), -1, -1)
        else:
            ctx = self.ctx[label]

        prefix = self.token_prefix[label]
        suffix = self.token_suffix[label]

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(prompts.shape)
        tokenized_prompts = self.tokenized_prompts[label]
        return prompts, tokenized_prompts