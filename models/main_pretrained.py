import os
import torch
import pretrained_fasterrcnn, pretrained_maskrcnn, pretrained_retinanet, pretrained_ssd

# Run pretrained model evaluation on COCO dataset 
pretrained_fasterrcnn.pretrained_evaluation()
pretrained_maskrcnn.pretrained_evaluation()
pretrained_retinanet.pretrained_evaluation()
pretrained_ssd.pretrained_evaluation()

# Run pretrained model evaluation on COCO dataset with blurred images 
pretrained_fasterrcnn.pretrained_evaluation(True)
pretrained_maskrcnn.pretrained_evaluation(True)
pretrained_retinanet.pretrained_evaluation(True)
pretrained_ssd.pretrained_evaluation(True)