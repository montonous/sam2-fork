from modules.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate 
# from groundingdino.util.inference import load_model, load_image, predict, annotate
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2 as cv
import numpy as np
import hydra

FILTER_SIZE = (31, 31)

def xywh_2_xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    new_boxes = np.stack((x1, y1, x2, y2), axis=1)
    
    return new_boxes


class GSAM_entity_masker():
    
    def __init__(self, 
                 out_dir:str, 
                 gdino_model_cfg:str, 
                 gdino_checkpoint:str, 
                 sam_model_cfg:str, 
                 sam_checkpoint:str, 
                 sam_config_dir:str,
                 image_path:str, 
                 text:str
        ) -> None:
        
        # hydra.core.global_hydra.GlobalHydra.instance().clear()
        # hydra.initialize_config_module(sam_config_dir, version_base='1.2')

        self.image_path = image_path
        self.text = " . ".join(text)
        print(self.image_path)
        print("************************", text)
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        
        self.out_dir = out_dir
        self.gdino_model = load_model(gdino_model_cfg, gdino_checkpoint)
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
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
        print("################### masks: ", self.boxes_pixel)
        print("################### image: ", self.image_np.shape)
        if len(self.boxes_pixel) > 0:
            self.masks, confs, low_res_masks = self.sam_predictor.predict(box=self.boxes_pixel, 
                                                                multimask_output=False)
            # ATM not saving masks, if we want to have several object with individual masks we need to act here
            dsp_image = self.image_np
            masked_img = cv.GaussianBlur(dsp_image, FILTER_SIZE, 0)
            for mask_ in self.masks:
                if mask_.shape[0] == 1:
                    mask = mask_.squeeze(0)                
                mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                masked_img = np.where(mask_3ch>0, dsp_image, masked_img)
            masked_img = cv.cvtColor(masked_img, cv.COLOR_BGR2RGB)

            #add full out dir path when finish testing 
            masked_img_path = self.out_dir + "/" + "premasked_" + self.image_path.split("/")[-1]
            cv.imwrite(masked_img_path, masked_img)
            
            return masked_img_path
        else:
            return self.image_path

if __name__ == '__main__':
    image_path = "../../data/test_images/train_station_person_tracks_2_SDE.png"
    text = "car . person . woman"
    out_dir = "../../data/out/"
    gdino_checkpoint = "../GroundingDINO/weights/groundingdino_swint_ogc.pth"
    gdino_model_cfg = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    sam_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam_model_cfg = "sam2/sam2.1_hiera_l.yaml"
    sam_config_dir_path = "../sam2/"
    masker = GSAM_entity_masker(out_dir = out_dir, 
                                gdino_checkpoint=gdino_checkpoint,
                                gdino_model_cfg=gdino_model_cfg,
                                sam_checkpoint=sam_checkpoint,
                                sam_config_dir=sam_config_dir_path,
                                sam_model_cfg=sam_model_cfg,
                                image_path= image_path,
                                 text= text)
    new_path = masker.mask()
    print(new_path)