from transformers import LxmertConfig, LxmertForQuestionAnswering


def lxmert_factory(command_name, model_config, run_config, dataset_config):
    if model_config.ckp_transformers:
        ckp_name = model_config.ckp_transformers
    if model_config.from_transformers:
        lxmert_config = LxmertConfig(
            num_qa_labels=model_config.known_labels,
            x_layers=model_config.x_layers,
            l_layers=model_config.l_layers,
            r_layers=model_config.r_layer,
            task_matched=model_config.msm,
            task_mask_lm=model_config.mlm,
            task_obj_predict=model_config.opm,
            visual_attr_loss=model_config.visual_attr_loss,
            visual_obj_loss=model_config.visual_obj_loss,
            visual_feat_loss=model_config.visual_feat_loss,
        )
    if command_name == "pretrain":
        raise NotImplementedError
    else:
        lxmert = LxmertForQuestionAnswering(lxmert_config)
    if ckp_name:
        lxmert.from_pretrained(ckp_name)
    return lxmert


NAME2MODEL = {
    "lxmert": lxmert_factory,
}
