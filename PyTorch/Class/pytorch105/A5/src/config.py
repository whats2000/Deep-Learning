VOC_CLASSES = (  # always index 0
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_IMG_MEAN = (123, 117, 104)  # RGB
VOC_IMG_STD = [58.395, 57.120, 57.375]  # Replace with your actual std values if different

COLORS = [
    [128, 0, 0],    # Maroon
    [255, 0, 0],    # Red
    [128, 128, 0],  # Olive
    [128, 64, 0],   # Brown
    [0, 128, 0],    # Green
    [0, 255, 0],    # Lime
    [0, 128, 128],  # Teal
    [0, 255, 255],  # Cyan
    [0, 0, 128],    # Navy
    [0, 0, 255],    # Blue
    [128, 0, 128],  # Purple
    [255, 0, 255],  # Magenta
    [128, 128, 128],# Gray
    [255, 255, 255],# White
    [64, 0, 0],     # Dark Maroon
    [255, 255, 0],  # Yellow
    [0, 128, 64],   # Greenish
    [0, 64, 128],   # Bluish
    [64, 0, 128],   # Purple-ish
    [128, 0, 64]    # Pinkish
]

# network expects a square input of this dimension
YOLO_IMG_DIM = 448
