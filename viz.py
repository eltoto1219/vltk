import cv2
import matplotlib as mpl
import matplotlib.figure as mplfigure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from image_processor_frcnn import tensorize

# RGB:
_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
# fmt: on


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def draw_boxes(ax, boxes, font_size, scale):
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    sorted_idxs = np.argsort(-areas).tolist()
    # Re-order overlapped instances in descending order.
    boxes = boxes[sorted_idxs] if boxes is not None else None
    assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(len(boxes))]
    assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
    for i in range(len(boxes)):
        color = assigned_colors[i]
        draw_box(ax, boxes[i], font_size, scale, edge_color=color)
    return ax


def draw_box(
    ax, box_coord, font_size, scale, alpha=0.5, edge_color="g", line_style="-"
):
    x0, y0, x1, y1 = box_coord
    width = x1 - x0
    height = y1 - y0

    linewidth = max(font_size, 1)

    ax.add_patch(
        mpl.patches.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth * scale,
            alpha=alpha,
            linestyle=line_style,
        )
    )


def viz_image(img, boxes, scale=1.2):
    """
    Args:
        img (ndarray): an RGB image of shape (H, W, 3).
        scale (float): scale the input image
    """
    width, height = img.shape[1], img.shape[0]
    font_size = max(np.sqrt(height * width) // 90, 10 // scale)
    fig = mplfigure.Figure(frameon=False)
    dpi = fig.get_dpi()
    fig.set_size_inches(
        (width * scale + 1e-2) / dpi,
        (height * scale + 1e-2) / dpi,
    )

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.set_xlim(0.0, width)
    ax.set_ylim(height)

    ax = draw_boxes(ax, boxes, font_size, scale)

    save(img, fig, ax, height, width, "test_out.jpg")


def save(img, fig, ax, height, width, filepath):
    if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        # faster than matplotlib's imshow
        cv2.imwrite(filepath, get_image(img, fig, height, width)[:, :, ::-1])
    else:
        # support general formats (e.g. pdf)
        fig.savefig(filepath)


def show(img):
    pass


def get_image(img, fig, hwidth, hheight):
    canvas = FigureCanvasAgg(fig)
    # canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
    s, (width, height) = canvas.print_to_buffer()
    if (hwidth, hheight) != (width, height):
        img = cv2.resize(img, (width, height))

    # buf = io.BytesIO()  # works for cairo backend
    # canvas.print_rgba(buf)
    # width, height = self.width, self.height
    # s = buf.getvalue()

    buffer = np.frombuffer(s, dtype="uint8")

    # imshow is slow. blend manually (still quite slow)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)

    try:
        import numexpr as ne  # fuse them with numexpr

        visualized_image = ne.evaluate(
            "img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)"
        )
    except ImportError:
        alpha = alpha.astype("float32") / 255.0
        visualized_image = img * (1 - alpha) + rgb * alpha

    visualized_image = visualized_image.astype("uint8")

    return visualized_image
