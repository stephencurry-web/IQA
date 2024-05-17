from .vit import build_vit


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "purevit":
        model = build_vit(
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            pretrained=config.MODEL.VIT.PRETRAINED,
            pretrained_model_path=config.MODEL.VIT.PRETRAINED_MODEL_PATH,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
