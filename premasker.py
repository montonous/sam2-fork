from groundingdino.util.inference import load_model, load_image, predict, annotate
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2 as cv
import numpy as np


def xywh_2_xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    new_boxes = np.stack((x1, y1, x2, y2), axis=1)
    
    return new_boxes


class GSAM_entity_masker():
    
    def __init__(self, image_path:str, text:str) -> None:
        
        self.image_path = image_path
        self.text = text
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        
        self.out_dir = "/home/systest/Documents/repos/Milestone-Adaptive-Experience/agent/main_graph/data/out/"
        self.gdino_model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.sam_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        self.sam_model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_predictor = SAM2ImagePredictor(build_sam2(self.sam_model_cfg, self.sam_checkpoint))

        self.image_np, self.image_g = load_image(self.image_path)
        self.H, self.W, self.C = self.image_np.shape
        self.axis_size = np.array([self.W, self.H, self.W, self.H])
        
    def mask(self):
        # Function that looks for entities specified in text that appear in the image. Segments
        # and returns a path for a segmented image for premasking purposes
        self.boxes, logits, phrases = predict(model=self.gdino_model,
                                        image=self.image_g,
                                        caption=self.text,
                                        box_threshold=self.BOX_TRESHOLD,
                                        text_threshold=self.TEXT_TRESHOLD
                                    )
        boxes_xyxy = xywh_2_xyxy(self.boxes)
        self.boxes_pixel = boxes_xyxy * self.axis_size
        
        self.sam_predictor.set_image(image=self.image_np)
        self.masks, confs, low_res_masks = self.sam_predictor.predict(box=self.boxes_pixel, 
                                                            multimask_output=False)
        # ATM not saving masks, if we want to have several object with individual masks we need to act here
        dsp_image = self.image_np
        masked_img = np.zeros_like(dsp_image)
        for mask_ in self.masks:
            mask = mask_.squeeze(0)
            mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            masked_img = np.where(mask_3ch>0, dsp_image, masked_img)
        masked_img = cv.cvtColor(masked_img, cv.COLOR_BGR2RGB)

        #add full out dir path when finish testing 
        masked_img_path = "annotated_image.jpg"
        cv.imwrite(masked_img_path, masked_img)
        
        return masked_img_path

if __name__ == '__main__':
    image_path = "../../data/test_images/00286.jpg"
    text = "car . person . bus . door . box . cart ."
    masker = GSAM_entity_masker(image_path, text)
    new_path = masker.mask()
    print(new_path)