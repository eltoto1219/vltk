from transformers import LxmertConfig, LxmertForQuestionAnswering


def lxmert_factory(command_name, model_config, run_config, dataset_config):
    if model_config.ckp_transformers:
        ckp_name = model_config.ckp_transformers
    if model_config.from_transformers:
        lxmert_config = LxmertConfig(
            num_qa_labels=model_config.known_labels,
            x_layers=model_config.x_layers,
            l_layers=model_config.l_layers,
            r_layers=model_config.r_layers,
            task_matched=run_config.task_matched,
            task_mask_lm=run_config.task_mask_lm,
            task_obj_predict=run_config.task_obj_predict,
            visual_attr_loss=run_config.visual_attr_loss,
            visual_obj_loss=run_config.visual_obj_loss,
            visual_feat_loss=run_config.visual_feat_loss,
        )
    if command_name == "pretrain":
        raise NotImplementedError
    else:
        lxmert = LxmertForQuestionAnswering(lxmert_config)
    if ckp_name:
        lxmert.from_pretrained(ckp_name)
    print("LXMERT FINAL NUM LABELS", lxmert.num_qa_labels)
    return lxmert


NAME2MODEL = {
    "lxmert": lxmert_factory,
}
