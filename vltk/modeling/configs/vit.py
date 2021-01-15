pxls = config.data.img_max_size
patch = config.data.patch_size
vit_pretrained_dir = config.pathes.vit_pretrained_dir
vitfp = os.path.join(vit_pretrained_dir, "pytorch_model.bin")
vittorch = VisionTransformer(image_size=(pxls, pxls), patch_size=(patch, patch))
vittorch.load_state_dict(torch.load(vitfp))
for n, p in vittorch.named_parameters():
    if "embed" not in n:
        p.requires_grad = False


