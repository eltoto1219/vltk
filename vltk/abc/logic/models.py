import torch
from vltk.modeling import Get as Mget
from vltk.modeling.configs import Get

# TODO change gpu config stuff

# return model to gpu dict


def _init_models(config, model_list):
    model_dict = {}
    model_configs = {}
    state_dict = None
    dev_list = config.models.models_to_devices
    if dev_list is not None:
        dev_map = {}
        for md in dev_list:
            dev_map[md[0]] = md[1]
    else:
        dev_map = None
    for x in model_list:
        if isinstance(x, tuple):
            name = x[0]
            model_class = x[1]
        elif isinstance(x, str):
            name = x
            model_class = Mget[name]

        model_config = getattr(config.models, name, None)
        if model_config is None:
            print(f"No Model Config For {name}", "returning class, not instance")
            model_dict[name] = model_class
            model_configs[name] = None
            if config.models.all_on_same_device:
                model_class.to(torch.device(config.gpu))
                # ADD TO MODEL2DEV DICT
            else:
                assert (
                    name in dev_map
                ), f"model {name} must be in dev_map, please see docs."
                dev = dev_map[name]
                model_class.to(torch.device(dev))
                # ADD TO MODEL2DEV DICT
            continue
        checkpoint = getattr(model_config, "checkpoint", None)
        print(f"instantiating {name} from {checkpoint}")
        # load from checkpoint if specificed

        if checkpoint is not None and hasattr(model_class, "from_pretrained"):
            # this is a huggingface model, so the config must be added appropriately
            model_instance = model_class.from_pretrained(
                checkpoint,
            )
            checkpoint = model_instance.state_dict()
            model_config = Get[name](**model_config.to_dict())
            model_instance = model_class(model_config)

        elif not hasattr(model_class, "from_pretrained"):
            model_instance = model_class(**model_config.to_dict())
            if checkpoint is not None:
                state_dict = torch.load(checkpoint)
        else:
            # model does not have checkpoint
            try:
                model_instance = model_class(model_config)
            except Exception:
                model_instance = model_class(**model_config.to_dict())

        if checkpoint is not None and state_dict is not None:
            model_instance.load_state_dict(state_dict, strict=False)

        # for question answering models, we will need to resize the number of labels
        # accordingly
        if hasattr(model_instance, "resize_num_qa_labels"):
            assert getattr(self, "label_to_id", None) is not None, "no label dict found"
            print(f"Number of Labels: {len(self.label_to_id)}")
            model_instance.resize_num_qa_labels(len(self.label_to_id))

        if self.config.models.all_on_same_device:
            model_instance.to(torch.device(self.config.gpu))
            setattr(self, f"{name}_dev", self.config.gpu)
        else:
            assert (
                dev_map is not None and name in dev_map
            ), f"model {name} must be in dev_map, please see docs."
            dev = dev_map[name]
            model_instance.to(torch.device(dev))
            setattr(self, f"{name}_dev", dev)

        model_dict[name] = model_instance
        model_configs[name] = model_config
        setattr(self, name, model_instance)

    self._model_dict = model_dict
    self._model_configs = model_configs
