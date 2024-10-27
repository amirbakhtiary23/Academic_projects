import math
import cv2
import os,datetime
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import mpi4py.MPI as mpi
comm=mpi.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
#print(rank,size)
path="dataset/val2017/"
interpreter = tflite.Interpreter(model_path="detr_fp32.tflite")
interpreter.allocate_tensors()
files=os.listdir(path)    
class Distribute:
    def __init__(self,world_size,rank):
        
        self.rank=rank
        self.world_size=world_size
        self.path="dataset/val2017/"
        self.files=os.listdir(self.path)
        self.capture()
    def capture(self):
        start_time=time.time()
        """
        this method can be replaced with a real videocapture method
        """
        counter=0
        for j in range(0,30+self.world_size-1):
            f=self.path+self.files[j]
            if "jpg" in f :
                if counter>29:
                    f=0
                    continue
                request=None
                
                request=comm.recv(request,tag=1)#tag 1 is request
                comm.send(f,dest=request,tag=1)           
                counter+=1
        end_time = time.time()
        print (f"inference on {counter} images took {end_time-start_time} on {size-1} workers ")
        print (f"{counter/(end_time-start_time)} fps on {size-1} workers mode 1")
        exit()
def run ():   
    #print (rank,"rank")
    if rank==1:
        dist=Distribute(size,rank)
    else :
        images=0
        start_time = time.time()
        while True:
            comm.send(rank,dest=1,tag=1)
            f=None
            f=comm.recv(f,tag=1,source=1)
            if f==0:
                print("finished") 
                break
            inp=cv2.imread(f)
            inp=cv2.resize(inp,(224,224))
            inp=inp.reshape(1,224,224,3).astype("float32")/255
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_tflite= interpreter.get_tensor(output_details[0]['index'])
            bounding_boxes = interpreter.get_tensor(output_details[1]['index'])
            images+=1
            #print (keep.shape)

        

run()
    

