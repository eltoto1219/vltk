import torch
from mllib import compat
from mllib.abc.imageset import Imageset
from mllib.configs import Config
from mllib.modeling.frcnn import FRCNN

TESTPATH = "/home/avmendoz/mllib/tests"


# user must only define forward in this function


class FRCNNSet(Imageset):
    # name will be overwritten with the name of the dataset when loaded from file
    name = "frcnn"

    def forward(filepath, image_preprocessor, model, **kwargs):

        pad_value = kwargs.get("pad_value", 0.0)
        min_size = kwargs.get("min_size", 800)
        max_size = kwargs.get("max_size", 800)
        pxl_mean = kwargs.get("pxl_mean", None)
        pxl_sdev = kwargs.get("pxl_sdev", None)
        device = kwargs.get("device", "cpu")

        image, sizes, scale_hw = image_preprocessor(
            filepath,
            min_size=min_size,
            max_size=max_size,
            mean=pxl_mean,
            sdev=pxl_sdev,
        )

        sizes = torch.tensor(list(sizes))
        scale_hw = torch.tensor(list(scale_hw))

        image, sizes, scale_hw = (
            image.to(torch.device(device)),
            sizes.to(torch.device(device)),
            scale_hw.to(torch.device(device)),
        )

        output_dict = model(
            images=image.unsqueeze(0),
            image_shapes=sizes.unsqueeze(0),
            scales_yx=scale_hw.unsqueeze(0),
            padding="max_detections",
            pad_value=pad_value,
            return_tensors="np",
            location="cpu",
        )

        return output_dict


if __name__ == "__main__":

    # frcnnconfig = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    # frcnnconfig.model.device = 0
    # frcnn = FRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnnconfig)
    config = Config().data

    # imageset = FRCNNSet.extract(
    #     dataset_name="coco2014",
    #     config=config,
    #     model=frcnn,
    #     image_preprocessor="img_to_tensor",
    #     features="frcnn",
    #     max_detections=config.max_detections,
    #     pos_dim=config.pos_dim,
    #     visual_dim=config.visual_dim,
    #     device=frcnnconfig.model.device,
    # )

    # vqa = VQAset.from_config(config, split="val")["val"]
    # arrow_path = VQAset.locations(
    #     config, split="trainval", imageset="coco2014", textset=VQAset.name
    # )["arrow"][0]
    # coco2014 = FRCNNSet.from_file(arrow_path)

    # imgid = next(iter(coco2014.img_to_row_map.keys()))

    # print("is aligned?", coco2014.check_imgid_alignment())
    # print(f"entry for {imgid}: ", coco2014.get_image(imgid).keys())
