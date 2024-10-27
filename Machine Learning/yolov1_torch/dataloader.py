import torch
import os
import pickle
import cv2
class Dataloader(torch.utils.data.Dataset):
    def __init__(self,data,B=2,S=7,C=20,batch_size=8,path=None):
        #super(Dataloader,self).__init__()
        self.C=C
        self.B=B
        self.S=S
        self.batch_size=batch_size
        self.data=data
        self.path=path
    
    def __len__(self):
        return len(self.data)//8
    
    def __read(self,begin,finish):
        batch=self.data[begin:finish]
        batch_target=torch.zeros((self.batch_size,25,7,7),
                                 dtype=torch.float32)
        batch_data=torch.zeros((self.batch_size,3,448,448),
                               dtype=torch.float32)
        #print (batch)
        for index,item in enumerate(batch):
            #print (index,item)
            (path,boxes)=item
            image = cv2.imread(self.path+"/"+path)#reading image from the respective path
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#opencv uses BGR by default so the conversion is required
            image_h, image_w = image.shape[0:2]
            aspect_x= 448/image_w
            aspect_y=448/image_h
            image=cv2.resize(image,(448,448))
            image=image.transpose(2, 0, 1)
            image=image.astype("float32")-127.5
            image/=127.5
            batch_data[index]=torch.Tensor(image)
            for box in boxes:
                (xmin,ymin,xmax,ymax,cls_id)=box
                xmin = xmin*aspect_x
                ymin = ymin*aspect_y
                xmax = xmax*aspect_x
                ymax = ymax*aspect_y

                x = (xmin + xmax) / 2 
                y = (ymin + ymax) / 2 #/ 448
                w = (xmax - xmin) / 448
                h = (ymax - ymin) / 448
                loc = [x/64, y/64]
                loc_i = int(loc[1])
                loc_j = int(loc[0])
                y = loc[1] - loc_i
                x = loc[0] - loc_j
                #placing labels with respect to their grid cell
                if batch_target[index,20,loc_i, loc_j] == 0:
                    batch_target[index,cls_id,loc_i, loc_j] = 1
                    batch_target[index,21,loc_i, loc_j] = x
                    batch_target[index,22,loc_i, loc_j] =y
                    batch_target[index,23,loc_i, loc_j] =w
                    batch_target[index,24,loc_i, loc_j] =h
                    batch_target[index,20,loc_i, loc_j] = 1  # response
        return batch_data,batch_target
            

    def __getitem__(self,idx):
        begin=idx*self.batch_size
        finish=begin+self.batch_size
        #print (begin,finish)
        return self.__read(begin,finish)
        

    
    