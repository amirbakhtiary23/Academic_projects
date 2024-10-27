
import torch
import torch.nn as nn
import config
import numpy as np
import cv2
def IOU(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[:, 0:1] - boxes_preds[:, 2:3] / 2
        box1_y1 = boxes_preds[:, 1:2] - boxes_preds[:, 3:4] / 2
        box1_x2 = boxes_preds[:, 0:1] + boxes_preds[:, 2:3] / 2
        box1_y2 = boxes_preds[:, 1:2] + boxes_preds[:, 3:4] / 2
        box2_x1 = boxes_labels[:, 0:1] - boxes_labels[:, 2:3] / 2
        box2_y1 = boxes_labels[:, 1:2] - boxes_labels[:, 3:4] / 2
        box2_x2 = boxes_labels[:, 0:1] + boxes_labels[:, 2:3] / 2
        box2_y2 = boxes_labels[:, 1:2] + boxes_labels[:, 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[:, 0:1]
        box1_y1 = boxes_preds[:, 1:2]
        box1_x2 = boxes_preds[:, 2:3]
        box1_y2 = boxes_preds[:, 3:4]  # (N, 1)
        box2_x1 = boxes_labels[:, 0:1]
        box2_y1 = boxes_labels[:, 1:2]
        box2_x2 = boxes_labels[:, 2:3]
        box2_y2 = boxes_labels[:, 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
def calculate_output(parameters:tuple,input_shape:list):
    output_shape=[0,0,parameters[2]]
    input_w,input_h=input_shape
    output_shape[0]=int(((input_w-parameters[4][0]+(2*parameters[5]))/parameters[3])+1)
    output_shape[1]=int(((input_h-parameters[4][1]+(2*parameters[5]))/parameters[3])+1)
    return (output_shape)

class YoloLoss(nn.Module):
    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss,self).__init__()
        self.S=S
        self.B=B
        self.C=C
        self.L1=nn.MSELoss()
        self.L2=nn.BCELoss()
        self.L3=nn.MSELoss()
        self.L4=nn.CrossEntropyLoss()
    
    def forward(self,target,pred):
        exist_box=target[:,20:21]
        
        b1_target=target[:,21:25,...]
        b1_pred=pred[:,21:25,...]
        b2_pred=pred[:,26:,...]
        iou_b1=IOU(b1_pred,b1_target)
        iou_b2=IOU(b2_pred,b1_target)
        ious=torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        iou_maxes,best_box=torch.max(ious,dim=0)
        #print (iou_maxes)
        
        #coords_loss
        box_predictions=exist_box* ((best_box*b1_pred)+ ((1-best_box)*b2_pred))
        b1_target=exist_box*b1_target
        #w_h
        """box_predictions[:,2:4]=torch.sign(box_predictions[:,2:4])*torch.sqrt(
            torch.abs(box_predictions[:,2:4])+1e-6)
        b1_target[:,2:4]=torch.sqrt(b1_target[:,2:4])"""
        box_predictions = torch.cat([
            box_predictions[:, :2],
            torch.sign(box_predictions[:, 2:4]) * torch.sqrt((torch.abs(box_predictions[:, 2:4]) + 1e-6))*448,
            box_predictions[:, 4:]
        ], dim=1)

        b1_target = torch.cat([
            b1_target[:, :2],
            torch.sqrt(b1_target[:, 2:4])*448,
            b1_target[:, 4:]
        ], dim=1)
        box_loss=self.L3(b1_target/448,box_predictions/448)

         #print (b1_pred.shape)
        #object_loss
        object_prediction=best_box*pred[:,25:26]+((1-best_box)*pred[:,20:21])
        object_loss=self.L1(exist_box*object_prediction,exist_box)#object_prediction)
        
        #no_object_loss
        #no_object_loss=best_box*pred[:,25:26]+((1-best_box)*pred[:,20:21])
        no_object_loss=self.L1(1-exist_box,(1-exist_box)*object_prediction)
        
        #class_prob_loss
        class_prob_target=target[:,0:20,...]
        class_prob_pred=pred[:,0:20,...]
        class_prob_loss=self.L4(class_prob_pred,class_prob_target)
        loss = ((config.lambda_coord*box_loss)+
                (config.lambda_noobj*no_object_loss)+
                (config.lambda_obj*object_loss)+(class_prob_loss)

        )
        return loss
"""def get_boxes(target_tensor,predicted_tensor,batch_size=8):
    pass"""

def convert_prob_to_coord(tensor,S=7,size=448,batch_size=8,is_gt=False,thresh=0.7):
    scale=size//S
    boxes=[]
    for item in range(batch_size):
        boxes.append([])
        for x in range(7):
            for y in range(7):
                if is_gt :
                    box_id=tensor[item,20,x,y]
                    if box_id<thresh:
                        continue
                    xc,yc,w,h=tensor[item,21:,x,y]            
                    xc=int((y+xc)*scale)
                    yc=int((x+yc)*scale)
                    w=w*size
                    h=h*size
                    xmin=int(xc-w/2)
                    ymin=int(yc-h/2)
                    xmax=xmin+int(w)
                    ymax=ymin+int(h)
                    cls_id=np.argmax(tensor[item,:20,x,y].flatten())
                    boxes[-1].append([cls_id,xmin,ymin,xmax,ymax,box_id])
                else :

                    box_margin=(5 if tensor[item,20,x,y]<tensor[item,25,x,y] else 0)
                    box_id=tensor[item,20+box_margin,x,y]
                    if box_id<thresh:
                        continue
                    xc,yc,w,h=tensor[item,21+box_margin:25+box_margin,x,y]            
                    xc=int((y+xc)*scale)
                    yc=int((x+yc)*scale)
                    w=w*size
                    h=h*size
                    xmin=int(xc-(w/2))
                    ymin=int(yc-(h/2))
                    xmax=xmin+int(w)
                    ymax=ymin+int(h)
                    cls_id=np.argmax(tensor[item,:20,x,y].flatten())
                    boxes[-1].append([cls_id,xmin,ymin,xmax,ymax,box_id])
    return boxes





    
def nms(bboxes,S=7):
    if len(bboxes)==0:
            return np.array([])
    total_boxes=[]
    for cls_id in range(20):

        boxes=bboxes[bboxes[:,0]==cls_id]

        
        """xc = boxes[:, 1]
        yc = boxes[:, 2]
        w = boxes[:, 3]
        h = boxes[:, 4]
        x1=xc-(w/2)
        y1=yc-(h/2)
        x2=x1+w
        y2=y1+h"""
        x1 = boxes[:, 1]
        y1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]
        w=(x2-x1)//2
        h=(y2-y1)//2
        area=w*h
        scores = boxes[:, -1]
        idxs = np.argsort(scores)
        keep = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            # push S in filtered predictions list
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > 0.5)[0])))
        
        for item in boxes[keep]:
            total_boxes.append(item)
    return np.array(total_boxes)

def get_bboxes(
            target,pred,thresh=0.4,test=False ):
    all_pred_boxes = []
    all_true_boxes = []
    batch_size = target.shape[0]
    true_bboxes = convert_prob_to_coord(target,is_gt=True)

    bboxes = convert_prob_to_coord(pred)
    
    #print (len(bboxes))
    # make sure model is in eval before get bboxes
    train_idx = 0
    final_boxes_pred=[]
    final_boxes_target=[]
    """    for batch_idx, (x, labels) in enumerate(loader):
        

        batch_size = x.shape[0]
        true_bboxes = convert_prob_to_coord(target,is_gt=True)
        bboxes = convert_prob_to_coord(pred)"""

    for idx in range(batch_size):
        nms_boxes = nms(
            np.array(bboxes[idx]))
        #print (nms_boxes.shape,"shapeeeeeeeeeeee")

        #if batch_idx == 0 and idx == 0:
        #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
        #    print(nms_boxes)

        final_boxes_pred.append(nms_boxes)
        final_box_target=[]
        for box in true_bboxes[idx]:
            # many will get converted to 0 pred
            if box[-1] > thresh:
                final_box_target.append(box)

        final_boxes_target.append(final_box_target)

    #if test :print (final_boxes_pred[0]==final_boxes_pred[1])
    return final_boxes_target, final_boxes_pred

def plot(X,boxes):
    for item in range(len(X)):
        image=X[item]
        #print (image.shape)
        image=image.transpose( 1, 2,0)
        image=image*127.5
        image=image.astype("uint8")+127
        cv2.imwrite(f"test/test{item}.jpg",image)
        image=cv2.imread(f"test/test{item}.jpg")
        for box in boxes[item]:
            cls_id,xmin,ymin,xmax,ymax,box_id=box
            #print (cls_id,xmin,ymin,xmax,ymax,box_id)
            image=cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,0))
            image=cv2.putText(image, f"{config.classes_num2[int(cls_id)]} {box_id}", (int(xmin), int(ymin)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imwrite(f"test/test{item}.jpg",image)
if __name__=="__main__":
    x=torch.randn(1,30,7,7)
    y=torch.randn(1,30,7,7)
    loss=YoloLoss()
    print (loss(x,y))

    """
    0.0 121.0 2.0 198.0 67.0 0.7145469188690186
    0.0 -13.0 162.0 53.0 314.0 0.5543376207351685
    2.0 122.0 48.0 206.0 146.0 0.6310638785362244
    4.0 219.0 382.0 322.0 453.0 0.7683298587799072
    4.0 55.0 4.0 152.0 139.0 0.7342973947525024
    5.0 -35.0 92.0 25.0 212.0 0.6855494379997253
    6.0 -41.0 24.0 16.0 77.0 0.8713847994804382
    6.0 368.0 65.0 436.0 154.0 0.8099191188812256
    6.0 225.0 56.0 314.0 167.0 0.7363520860671997
    6.0 -28.0 56.0 67.0 144.0 0.7283639907836914
    8.0 325.0 354.0 412.0 426.0 0.8200342059135437
    8.0 58.0 340.0 150.0 417.0 0.7924020886421204
    8.0 0.0 285.0 57.0 426.0 0.5774170160293579
    8.0 407.0 144.0 452.0 260.0 0.5574166178703308
    8.0 20.0 128.0 127.0 319.0 0.5316221714019775
    9.0 118.0 179.0 238.0 349.0 0.4685239791870117
    10.0 97.0 369.0 210.0 448.0 0.8159927129745483
    10.0 296.0 76.0 388.0 222.0 0.6815003752708435
    10.0 375.0 265.0 428.0 364.0 0.6689079999923706
    10.0 56.0 84.0 213.0 244.0 0.6154950261116028
    13.0 330.0 34.0 400.0 129.0 0.7411494851112366
    13.0 138.0 386.0 277.0 462.0 0.7113425731658936
    13.0 -40.0 210.0 87.0 340.0 0.5537689328193665
    14.0 41.0 -44.0 106.0 45.0 0.8240142464637756
    14.0 277.0 -5.0 305.0 18.0 0.7908817529678345
    14.0 162.0 37.0 272.0 95.0 0.7498701214790344
    14.0 301.0 -71.0 347.0 131.0 0.7183155417442322
    14.0 219.0 109.0 354.0 238.0 0.6075708866119385
    14.0 215.0 255.0 360.0 406.0 0.599052369594574
    14.0 355.0 226.0 414.0 346.0 0.5809190273284912
    14.0 59.0 196.0 163.0 369.0 0.5535243153572083
    15.0 6.0 397.0 56.0 459.0 0.7904970645904541
    16.0 363.0 99.0 417.0 228.0 0.6680915951728821
    17.0 127.0 30.0 283.0 131.0 0.639417290687561
    19.0 302.0 258.0 421.0 398.0 0.6401125192642212
    19.0 57.0 103.0 142.0 224.0 0.6260393261909485
    """