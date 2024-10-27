import torch
import numpy as np
import config
from torch import nn

#we use torch summary instead of python list so torch summary can access it
from torchsummary import summary

activations={"leaky_relu":nn.LeakyReLU}
class Backbone(nn.Module):
    def __init__(self,config,index=1):
        super(Backbone,self).__init__()
        self.layers=nn.ModuleList()
        self.config=config
        self.index=index
        self.__create_network()
    def __create_network(self):
        for layer in self.config[:self.index]:
            layer_type,in_channels,out_channels,strides,kernel_size,pad,activation=layer
            if layer_type.lower()=="c":
                current_layer=nn.Conv2d(in_channels,out_channels,
                                        kernel_size,strides,pad,dtype=torch.float32)
                current_activation=activations[activation]()
                self.layers.append(current_layer)
                self.layers.append(current_activation)
            elif layer_type.lower()=="m":
                current_layer=nn.MaxPool2d(kernel_size,strides,pad)
                self.layers.append(current_layer)
            
    def initialize_weights(self):
        for m in self.layers():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def print_shape(self,x):
        for index,layer in enumerate(self.layers):
            print (index+1)
            x=layer(x)
            print(f'Layer: {layer.__class__.__name__}, Output shape: {x.shape}')
        return x
    
class YoloV1(nn.Module):
    def __init__(self,backbone,batch_size=8):
        super(YoloV1,self).__init__()
        self.module_list=nn.ModuleList()
        self.batch_size=batch_size
        self.backbone=backbone
        self.__create_model()

    def initialize_weights(self):
        for m in self.module_list():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def __create_model(self):
        #self.backbone=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)#Backbone(config.darknet_Backbone)
        """self.flatten=nn.Flatten()
        self.output=nn.Linear(512*12*12,1470)
        self.module_list.append(self.backbone)
        self.module_list.append(self.flatten)
        self.module_list.append(self.output)"""

        self.box2=nn.Conv2d(512,10,(8,8),1,0,dtype=torch.float32)
        self.class_probabilty=nn.Conv2d(512,20,(8,8),1,0,dtype=torch.float32)
        self.sigmoid1=nn.Sigmoid()
        self.sigmoid2=nn.Sigmoid()
        self.class_softmax=nn.Softmax()
        #self.module_list.append(self.box1)
        self.module_list.append(self.box2)
        self.module_list.append(self.class_probabilty)
        self.module_list.append(self.sigmoid1)
        self.module_list.append(self.sigmoid2)
        self.module_list.append(self.class_softmax)
    
    def forward(self,x):
        x=self.backbone(x)
        #print(x.shape)
        #flatten=self.flatten(x)
        """box1=self.box1(x)
        box1=self.sigmoid1(box1)"""
        #output=self.output(flatten)
        box2=self.box2(x)
        box2=self.sigmoid2(box2)
        class_probability=self.class_probabilty(x)
        #class_probability=self.class_softmax(class_probability)
        
        return torch.cat([class_probability,box2],dim=1)# output.view(self.batch_size,30,7,7)
    
    def print_shape(self,x):
        x=self.backbone(x)
        flatten=self.flatten(x)
        #box1=self.sigmoid1(box1)
        output=self.output(flatten)
        #class_probability=self.class_softmax(class_probability)
        print(f'class_probability: {x.__class__.__name__}, Output shape: {x.shape}')
        print(f'class_probability: {flatten.__class__.__name__}, Output shape: {flatten.shape}')
        print(f'class_probability: {output.__class__.__name__}, Output shape: {output.shape}')
        #return torch.cat([class_probability,box1,box2],dim=1)
        return output.view(self.batch_size,30,7,7)
    
class ResNet18WithoutHead(nn.Module):
    def __init__(self, original_model):
        super(ResNet18WithoutHead, self).__init__()
        # Copy all layers except the final fully connected layer
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        # You might also want to include a pooling layer if needed
        #self.pool =nn.MaxPool2d((3,3),1,0)

    def forward(self, x):
        x = self.features(x)
        #x = self.pool(x)
        return x
if __name__=="__main__":
    for i in range(len(config.darknet_Backbone)):
        backbone=ResNet18WithoutHead(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False))
        model=YoloV1(backbone=backbone,batch_size=1)
        batch1 = torch.randn(1, 3, 448, 448)
        batch2 = torch.randn(1, 3, 448, 448)
        output1 = model(batch1)
        output2 = model(batch2)
        print (output2.shape)
        break

       
    


   