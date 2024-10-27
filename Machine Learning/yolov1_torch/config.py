import utils

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}
classes_num2 = {v: k for k, v in classes_num.items()}

# A list of configurations tuples
# 0) type: "M" for max pool , "A" for average pool , "GM" for global maxpool , "GA" for global Average pool
# 1) in_channel 
# 2) out_put channel 
# 3) strides 
# 4) kernel_size
# 5) padding
# 6) repeat
# (type,in,out,strides,kernel,pad,activation)

darknet_Backbone=[
("C",3,64,2,(7,7),3,"leaky_relu"),#1
("M",64,64,2,(2,2),0,"leaky_relu"),#2
("C",64,192,1,(3,3),1,"leaky_relu"),#3
("M",192,192,2,(2,2),0,"leaky_relu"),#4
("C",192,128,1,(1,1),0,"leaky_relu"),#5
("C",128,256,1,(3,3),1,"leaky_relu"),#6
("C",256,256,1,(1,1),0,"leaky_relu"),#7
("C",256,512,1,(3,3),1,"leaky_relu"),#8
("M",512,512,2,(2,2),0,"leaky_relu"),#9

("C",512,256,1,(1,1),0,"leaky_relu"),#10
("C",256,512,1,(3,3),1,"leaky_relu"),#11

("C",512,256,1,(1,1),0,"leaky_relu"),#12
("C",256,512,1,(3,3),1,"leaky_relu"),#13

("C",512,256,1,(1,1),0,"leaky_relu"),#14
("C",256,512,1,(3,3),1,"leaky_relu"),#15

("C",512,256,1,(1,1),0,"leaky_relu"),#16
("C",256,512,1,(3,3),1,"leaky_relu"),#17

("C",512,512,1,(1,1),0,"leaky_relu"),#18
("C",512,1024,1,(3,3),1,"leaky_relu"),#19

("M",1024,1024,2,(2,2),0,"leaky_relu"),#20

("C",1024,512,1,(1,1),0,"leaky_relu"),#21
("C",512,1024,1,(3,3),1,"leaky_relu"),#22

("C",1024,512,1,(1,1),0,"leaky_relu"),#23
("C",512,1024,1,(3,3),1,"leaky_relu"),#24
("C",1024,1024,1,(3,3),1,"leaky_relu"),#25

("C",1024,1024,2,(3,3),1,"leaky_relu"),#26

("C",1024,1024,1,(3,3),1,"leaky_relu"),#27
("C",1024,1024,1,(3,3),1,"leaky_relu"),#28
]

lambda_coord=5
lambda_obj=5
lambda_noobj=0.5
if __name__=="__main__":
    input_shape=(448,448,3)
    for index,layer in enumerate(darknet_Backbone):
        print (input_shape,index)
        input_shape=utils.calculate_output(layer,input_shape[:2])
    print (input_shape,index+1)