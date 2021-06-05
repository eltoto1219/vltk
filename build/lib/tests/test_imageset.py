import datasets as ds
from vltk.abc.imageset import Imageset


class Coco2014(Imageset):
    name = "coco2014"

    @staticmethod
    def default_features(max_detections, pos_dim, visual_dim):
        return {
            "attr_ids": ds.Sequence(length=max_detections, feature=ds.Value("float32")),
            "attr_probs": ds.Sequence(
                length=max_detections, feature=ds.Value("float32")
            ),
        }

    def forward(filepath, image_preprocessor, model, **kwargs):
        pass


coco = Coco2014.extract(
    name="coco2014",
    img_config=None,
    splits=[],
    searchdirs="/playpen1/home/avmendoz/data",
)

coco = Coco2014.from_file(
    "/playpen1/home/avmendoz/data/coco2014/boxes/annotations.arrow"
)

print(coco)
