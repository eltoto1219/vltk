import datasets as ds

__all__ = ["box", "segmentation", "area", "imgid", "ids", "boxtensor", "features"]

box = ds.Sequence(
    length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
)

segmentation = (
    ds.Sequence(length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))),
)

area = ds.Sequence(length=-1, feature=ds.Value("float32"))

imgid = ds.Value("string")
string = ds.Value("string")

ids = ds.Sequence(length=-1, feature=ds.Value("float32"))

boxtensor = lambda n: ds.Array2D((n, 4), dtype="float32")

features = lambda n, d: ds.Array2D((n, d), dtype="float32")
