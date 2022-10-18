# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --quiet

try:
    from detectron2.config import get_cfg  

except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git'])

finally:
    import numpy as np
    import cv2, torch, os, gdown

    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg



def build_cfg():

    cfg = get_cfg()

    config_name = "config.yml" # Using pre trained layout parser configs
    cfg.merge_from_file(config_name)


    cfg.MODEL.DEVICE = "cpu"

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.TEST.EVAL_PERIOD = 20 # Evaluate after N epochs

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # Default 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # in config file, it is written before weights

    if not os.path.exists('./model.pth'):
        url = "https://drive.google.com/uc?id=1S5LhdZiKS8dXqUeXYfDEOVkfBxVMuZsk"
        output = "./model.pth"
        gdown.download(url, output, quiet=True)

    cfg.MODEL.WEIGHTS = './model.pth' # layout parser Pre trained weights


    cfg.SOLVER.IMS_PER_BATCH = 4 # Batch size
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.SOLVER.MAX_ITER = 1000 # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (300, 800) # must be less than  MAX_ITER 
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 20  # Save weights after these many epochs

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    return cfg


class TorchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = build_cfg() # Own function in helpers.py to load weights
        self.model = build_model(cfg) # Build Model
        _ = DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)  # Load weights
        self.model = self.model.eval() # In evaluation mode
    
    def forward(self, INPUT, score_thresh:float = 0.6, nms:float = 0.3):
        if isinstance(INPUT, (np.ndarray, torch.Tensor)):
            INPUT = [{"image":INPUT}]

        with torch.no_grad():
            outputs = self.model(INPUT)[0]['instances']
        
        boxes, labels, scores = outputs.pred_boxes.tensor, outputs.pred_classes, outputs.scores.detach()

        nms_indices = apply_score_nms(boxes.numpy(), scores.numpy(), score_thresh = score_thresh, nms = nms)

        return boxes[nms_indices], labels[nms_indices], scores[nms_indices]


def apply_score_nms(dets:np.ndarray, scores:np.ndarray, score_thresh:float, nms:float) -> list:
    '''
    apply Min thresholding and NMS and return only the indices which fulfil the criteria
    args:
        dets: [N,4] array depecting bounding boxes coordinates
        scores: N elemment array describing the probability of detection
        score_thresh: probability threshold above which the detections will be considered
        nms: drop the area of overlapping between boxes if the common area is more than this number
    '''
    mask = np.where(scores >= score_thresh)[0]
    scores = scores[mask]
    dets = dets[mask]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms)[0]
        order = order[inds + 1]

    return keep



def draw_boxes( image, boxes, labels, classes, scores, COLORS):
    '''
    Accept BGR image and plot Bounding boxe, labels on top of it
    '''
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2)

        text = f"{classes[i]} - {str(round(scores[i], 3))}"
        cv2.putText(image, text, (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    lineType=cv2.LINE_AA)

    return image[:,:,::-1] # Return BGR Image


def renormalize_cam_in_bounding_boxes(image_float_np, grayscale_cam, boxes, labels, classes, scores, colors):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        images.append(img)
    
    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam)
    image_with_bounding_boxes = draw_boxes(eigencam_image_renormalized, boxes, labels, classes, scores, colors)
    return image_with_bounding_boxes


def fasterrcnn_reshape_transform(x):
    '''
    Override the code from the original repo due to the same issue as: 
    Solution for the Issue: https://github.com/jacobgil/pytorch-grad-cam/issues/278
    Pass in your Pooling Layers name: Last one in my model is "p6"
    '''
    # for key, value in x.items():print(key)
    layer_name = "p6"

    target_size = x[layer_name].size()[-2:]
    activations = []
    for key, value in x.items():
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(value),
                target_size,
                mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


def get_results(model, image, score_thresh, nms):
    '''
    Get results from the model given an image
    '''
    float_image = np.float32(image/255.) #  Normalized BGR image between 0-1

    tensor_image = torch.from_numpy(image.copy()).permute(2, 0, 1) # [B, channels, W, H]
    tensor_boxes, labels, scores = model(tensor_image, score_thresh, nms)

    boxes = np.int32(tensor_boxes.numpy())
    labels = [i for i in range(len(labels))]
    classes = [f"Q: {str(i)}" for i in labels]

    COLORS = np.random.uniform(0, 255, size=(len(set(labels)), 3)) # for plotting purpose

    return float_image, tensor_image, boxes, labels, classes, scores.numpy(), COLORS


def eigen_res(model, float_image, tensor_image, boxes, labels, classes, scores, colors):
    '''
    Run the EigenCam to get results
    '''
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)] # this not executed in EigenCam just init
    target_layers = [model.model.backbone]

    cam = EigenCAM(model,target_layers, use_cuda=torch.cuda.is_available(), reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(tensor_image, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :] # Take the first image in the batch

    cam_image = show_cam_on_image(float_image, grayscale_cam) 

    RGB_unnormalized_image = draw_boxes(cam_image, boxes, labels, classes, scores, colors) # without normalizing
    RGB_normalized_image = renormalize_cam_in_bounding_boxes(float_image, grayscale_cam,  boxes, labels, classes, scores, colors) # with Normalized Boxes

    return RGB_unnormalized_image, RGB_normalized_image






