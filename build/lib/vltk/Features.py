import datasets as ds

Box = ds.Sequence(
    length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
)

Polygons = (
    ds.Sequence(length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))),
)

Points = (ds.Sequence(length=-1, feature=ds.Value("float32")),)

FloatList = ds.Sequence(length=-1, feature=ds.Value("float32"))

Imgid = ds.Value("string")

String = ds.Value("string")

StringList = ds.Sequence(length=-1, feature=ds.Value("string"))

NestedStringList = ds.Sequence(ds.Sequence(length=-1, feature=ds.Value("string")))


Ids = ds.Sequence(length=-1, feature=ds.Value("float32"))


def Boxtensor(n):
    return ds.Array2D((n, 4), dtype="float32")


# something doesnt look right here (between 2d and 3d features)
def Features2D(d):
    return ds.Array2D((-1, d), dtype="float32")


def Features3D(n, d):
    return ds.Array2D((n, d), dtype="float32")
