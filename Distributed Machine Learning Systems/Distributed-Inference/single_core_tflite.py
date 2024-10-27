import tflite_runtime.interpreter as tflite
import cv2
import os
import time
import numpy as np
interpreter = tflite.Interpreter(model_path="detr_fp32.tflite")
interpreter.allocate_tensors()
path="dataset/val2017/"
files=os.listdir(path)
timer=time.time()
for j in range(30):
    
    f=path+files[j]
    if "jpg" in f :
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
timer2=time.time()
print (f"inference on 30 images took {timer2-timer}")
print (f"{30/(timer2-timer)} fps")