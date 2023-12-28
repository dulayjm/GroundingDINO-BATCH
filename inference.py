# sagemaker functions

import base64
import cv2
from io import BytesIO
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import subprocess
import time
import torch
import warnings

print("WORKING DIR:", os.getcwd())


# do the example code lik in the other stuff ...

# os.system("python /opt/ml/code/setup.py build develop --user")
# os.system("pip install packaging==21.3")
warnings.filterwarnings("ignore")

print("MADE IT PAST SYSTEM CALLS")

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

print("MADE IT PAST IMPORTS")


# except subprocess.CalledProcessError:
#     print(f"Failed to install. Please install it manually.")
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USING DEVICE: ", device)

# init function
def load_models(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network) and the column info.
    """    
    model = load_model("/opt/ml/code/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_dir, "./weights/", cpu_only=False)

    print("MODEL IS ", model)

    return model


# call function
def predict(model, data):
    """
    model: fashion_clip is a transformers.pipeline
    data: json data like:
    inputs {
        image: [List[str]: list of base64 encoded images],
        candiates: [List[str]: inputs for clases]
    }
    """
    # if "application/json" in input_content_type:
    data = json.loads(data)
    inputs = data.pop("inputs", data)

    if isinstance(inputs['image'], list):
        print(f"Processing {len(inputs['image'])} images ...")

        input_images = []
        for base64_string in inputs['image']:
            image = Image.open(BytesIO(base64.b64decode(base64_string)))
            # TODO: convert to cv2 image ...
            # opencv_image = np.array(pillow_image)
            # Convert RGB to BGR (as OpenCV uses BGR format)
            # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            torch_image = pil_image_to_torch_tensor(image)
            input_images.append(torch_image)
            
            
        text_prompt = inputs['text']
            
        inport_torch_stack = torch.stack(input_images)

        boxes_filt, pred_phrases = get_grounding_output(
            model, inport_torch_stack, text_prompt, 
            0.35, 0.25, 
            cpu_only=False, token_spans=None)

        # output = model(images=input_images, candidate_labels=inputs["candiates"])
        print("boxes_filt", boxes_filt)
        print("pred_phrases", pred_phrases)

        return boxes_filt, pred_phrases
            
            
# helper functions 
def get_grounding_output(model, images, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    images = images.to(device)

    # here is where we are making chnages with those outputs ...
    
    # print(type(image))
    print(images.shape)
    # print(image[None].shape)

    # print(torch.eq(image, image[None]))

    # y = image.detach().clone() # method e
    # z = image.detach().clone() # method e

    # images = torch.stack([image, y, z])

    print("images shape", images.shape)
    
    # match dimension of inputs with combining for length of everything
    captions = [caption for i in range(len(images))]
    
    with torch.no_grad():
        outputs = model(images, captions=captions)
    logits = outputs["pred_logits"].sigmoid()  # (nq, 256)
    boxes = outputs["pred_boxes"]  # (nq, 4)

    print(logits.shape)
    print(boxes.shape)

    # for logit in logits, you can do the rest with it ...
    all_boxes_filt = []
    all_pred_phrases = []
    for i in range(len(logits)):
        logits_filt = logits[i].cpu().clone()
        boxes_filt = boxes[i].cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        
        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(captions[i])
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        
        boxes_filt = list(boxes_filt)

        all_boxes_filt.append(boxes_filt)
        all_pred_phrases.append(pred_phrases)

    print("all_boxes_filt", all_boxes_filt)
    print("all_pred_phrases", all_pred_phrases)
    return all_boxes_filt, all_pred_phrases

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    # checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    # load on GPU
    checkpoint = torch.load(model_checkpoint_path)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def pil_image_to_torch_tensor(image_pil):
    # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image
