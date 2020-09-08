# finally-frcnn

``python3 -m pip install -e git+https://github.com/eltoto1219/finally-frcnn.git``

# example

``
import numpy as np

from frcnn import GeneralizedRCNN, Preprocess, load_config
``
load default config
``
default_config = load_config()
``
change value if needed, one only needs to print the config to see all of the current settings
``
default_config.roi_heads.nms_thresh_test = 0.2
``
load the model, make sure to put it in eval mode, training is not supported
``
model = GeneralizedRCNN.from_pretrained(config=default_config)
model.eval()
image_processor = Preprocess(default_config)
``
random image, can also just reference path, or a list of paths, or list of nd.arrays (H, W, C)
``
image = np.random.rand(800, 800, 3).astype("uint8")
images, sizes, scales_yx = image_processor(image)
output_dict = model(images, sizes, scales_yx=scales_yx)
``
