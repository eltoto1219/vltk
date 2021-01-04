from collections import OrderedDict

import datasets as ds


def frcnn(max_detections, pos_dim, visual_dim):
    return ds.Features(
        OrderedDict(
            {
                "attr_ids": ds.Sequence(
                    length=max_detections, feature=ds.Value("float32")
                ),
                "attr_probs": ds.Sequence(
                    length=max_detections, feature=ds.Value("float32")
                ),
                "boxes": ds.Array2D((max_detections, pos_dim), dtype="float32"),
                "normalized_boxes": ds.Array2D((max_detections, pos_dim), dtype="float32"),
                "img_id": ds.Value("string"),
                "obj_ids": ds.Sequence(
                    length=max_detections, feature=ds.Value("float32")
                ),
                "obj_probs": ds.Sequence(
                    length=max_detections, feature=ds.Value("float32")
                ),
                "roi_features": ds.Array2D(
                    (max_detections, visual_dim), dtype="float32"
                ),
                "sizes": ds.Sequence(length=2, feature=ds.Value("float32")),
                "preds_per_image": ds.Value(dtype="int32"),
            }
        )
    )

