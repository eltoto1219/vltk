import datasets as ds

Box = ds.Sequence(
    length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
)

Polygons = ds.Sequence(
    length=-1,
    feature=ds.Sequence(
        length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
    ),
)

# RLE = ds.Sequence(
#     length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
# )


Bool = ds.Value("bool")
BoolList = ds.Sequence(length=-1, feature=ds.Value("bool"))

RLE = ds.Sequence(length=-1, feature=ds.Value("float32"))

FloatList = ds.Sequence(length=-1, feature=ds.Value("float32"))

Imgid = ds.Value("string")

String = ds.Value("string")

StringList = ds.Sequence(length=-1, feature=ds.Value("string"))

NestedStringList = ds.Sequence(ds.Sequence(length=-1, feature=ds.Value("string")))

Int = ds.Value("int32")
IntList = ds.Sequence(length=-1, feature=ds.Value("int32"))
NestedIntList = ds.Sequence(
    length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("int32"))
)
Span = ds.Sequence(length=-1, feature=ds.Value("int32"))
Float = ds.Value("float32")


Ids = ds.Sequence(length=-1, feature=ds.Value("float32"))


def Boxtensor(n):
    return ds.Array2D((n, 4), dtype="float32")


# something doesnt look right here (between 2d and 3d features)
def Features2D(d):
    return ds.Array2D((-1, d), dtype="float32")


def Features3D(n, d):
    return ds.Array2D((n, d), dtype="float32")
