import math
from PIL import Image
import os,datetime
import torch
from torch import nn
import time
import queue,threading
import mpi4py.MPI as mpi
comm=mpi.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
print(rank,size)
import torchvision.transforms as T
master_port="32415"
torch.set_grad_enabled(False)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
path="dataset/val2017/"
files=os.listdir(path)    
class Distribute:
    def __init__(self,world_size,rank):
        
        self.rank=rank
        self.world_size=world_size
        self.path="dataset/val2017/"
        self.files=os.listdir(self.path)
        self.capture()
    def capture(self):
        """
        this method can be replaced with a real videocapture method
        """
        start_time=time.time()
        counter=0
        images=0
        for j in range(0,30+self.world_size-1):
            f=self.path+self.files[j]
            if "jpg" in f :
                if counter>30:
                    f=0
                    continue
                request=None
                
                request=comm.recv(request,tag=1)#tag 1 is request
                comm.send(f,dest=request,tag=1)           
                counter+=1
                images+=1
        end_time = time.time()
        print (f"PyTorch model inference on {images} images took {end_time-start_time} on {size-1} workers")
        print (f"{images/(end_time-start_time)} fps on {size-1} workers")
def run ():   
    #print (rank,"rank")
    if rank==1:
        dist=Distribute(size,rank)
    else :
        detectron = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)        
        detectron.eval();
        images=0
        while True:
            comm.send(rank,dest=1,tag=1)
            f=None
            f=comm.recv(f,tag=1,source=1)
            print (f,end=" , ")
            if f==0:
                print("finished") 
                break
            img=transform(Image.open(f)).unsqueeze(0)
            #print (f"rank {rank} image {images} {f}")
            output=detectron(img)
            images+=1
            probas = output['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.9
            #print ("done")
            #print (keep.shape)

        

run()
    

