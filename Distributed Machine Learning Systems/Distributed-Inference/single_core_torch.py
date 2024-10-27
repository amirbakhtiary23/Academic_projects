import math
from PIL import Image
import os
import torch
from torch import nn
import time
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
detectron = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
detectron.eval();
path="dataset/val2017/"
files=os.listdir(path)
timer=time.time()
#rank=torch.distributed.get_rank()
#world_size=local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
#print (rank,world_size)

for j in range(30):
	f=path+files[j]
	if "jpg" in f :
		img=transform(Image.open(f)).unsqueeze(0)
		output=detectron(img)
		probas = output['pred_logits'].softmax(-1)[0, :, :-1]
		keep = probas.max(-1).values > 0.9
		# convert boxes from [0; 1] to image scales
		#bboxes_scaled = rescale_bboxes(output['pred_boxes'][0, keep], im.size)
timer2=time.time()
print (f"inference on 30 images took {timer2-timer}")
print (f"{30/(timer2-timer)} fps")

