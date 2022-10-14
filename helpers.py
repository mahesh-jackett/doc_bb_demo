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



def build_cfg(THRESH:float = 0.6):

    cfg = get_cfg()

    config_name = "config.yml" # Using pre trained layout parser configs
    cfg.merge_from_file(config_name)


    cfg.MODEL.DEVICE = "cpu"

    cfg.DATALOADER.NUM_WORKERS: 2
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

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH
    return cfg


class TorchModel(torch.nn.Module):
    def __init__(self, thresh:float = 0.6) -> None:
        super().__init__()
        cfg = build_cfg(thresh) # Own function in helpers.py to load weights
        self.model = build_model(cfg) # Build Model
        _ = DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)  # Load weights
        self.model = self.model.eval() # In evaluation mode
    
    def forward(self, INPUT):
        if isinstance(INPUT, (np.ndarray, torch.Tensor)):
            INPUT = [{"image":INPUT}]

        with torch.no_grad():
            outputs = self.model(INPUT)[0]['instances']
        
        boxes, labels, scores = outputs.pred_boxes.tensor, outputs.pred_classes, outputs.scores.detach()

        return boxes, labels, scores


def draw_boxes( image, boxes, labels, classes, COLORS):
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

        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    lineType=cv2.LINE_AA)

    return image[:,:,::-1] # Return BGR Image


def renormalize_cam_in_bounding_boxes(image_float_np, grayscale_cam, boxes, classes, labels, colors):
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
    image_with_bounding_boxes = draw_boxes(eigencam_image_renormalized, boxes, labels, classes, colors)
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


def get_results(model, image):
    '''
    Get results from the model given an image
    '''
    float_image = np.float32(image/255.) #  Normalized BGR image between 0-1

    tensor_image = torch.from_numpy(image.copy()).permute(2, 0, 1) # [B, channels, W, H]
    tensor_boxes, labels, scores = model(tensor_image)

    boxes = np.int32(tensor_boxes.numpy())
    labels = [i for i in range(len(labels))]
    classes = [f"Q: {str(i)}" for i in labels]

    COLORS = np.random.uniform(0, 255, size=(len(set(labels)), 3)) # for plotting purpose

    return float_image, tensor_image, boxes, labels, classes, COLORS


def eigen_res(model, float_image, tensor_image, labels, boxes, classes, colors):
    '''
    Run the EigenCam to get results
    '''
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)] # this not executed in EigenCam just init
    target_layers = [model.model.backbone]

    cam = EigenCAM(model,target_layers, use_cuda=torch.cuda.is_available(), reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(tensor_image, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :] # Take the first image in the batch

    cam_image = show_cam_on_image(float_image, grayscale_cam) 

    RGB_unnormalized_image = draw_boxes(cam_image, boxes, labels, classes, colors) # without normalizing
    RGB_normalized_image = renormalize_cam_in_bounding_boxes(float_image, grayscale_cam,  boxes, classes, labels, colors) # with Normalized Boxes

    return RGB_unnormalized_image, RGB_normalized_image






