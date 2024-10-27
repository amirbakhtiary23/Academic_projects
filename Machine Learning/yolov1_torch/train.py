import torch
import dataloader
import utils
import tqdm
import pickle
import model as models
import mAP
import numpy as np
train_dict={}
def train():
    with open ("/media/amir/4C8C65E67913301E/Datasets/VOCdevkit/2007_test.pkl","rb") as f:
        Test_set=pickle.load(f)
    with open ("/media/amir/4C8C65E67913301E/Datasets/VOCdevkit/2007_val.pkl","rb") as f:
        Val_set=pickle.load(f)
    with open ("/media/amir/4C8C65E67913301E/Datasets/VOCdevkit/2007_train.pkl","rb") as f:
        Train_set=pickle.load(f)

    Train_set=Train_set+Val_set
    Dataloader=dataloader.Dataloader(Train_set,path="/media/amir/4C8C65E67913301E/Datasets/")
    TestLoader=dataloader.Dataloader(Test_set,path="/media/amir/4C8C65E67913301E/Datasets/")
    backbone=models.ResNet18WithoutHead(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False))
    
    model=models.YoloV1(backbone)
    #model.load_state_dict(torch.load("checkpoint.pth"))
    model.to(0)
    loss_fn = utils.YoloLoss().to(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs=200
    size = len(Dataloader)
    train_losses=[]
    test_losses=[]
    train_accuracies=[]
    test_accuracies=[]
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):
        if (epoch+1)%10 == 0:  # For example, change LR at epoch 30
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.9
        maps=[]
        loop=tqdm.tqdm(range(size),leave=True)
        epoch_loss=0
        model.train()
        for batch in loop:
            X, y=Dataloader.__getitem__(batch)
            if len(X)==0:
                break
            
            #X=torch.stack(X)
            X, y = X.to(0), y.to(0)
            
            # Compute prediction error
            pred = model(X)
            
            loss = loss_fn( y,pred)
            epoch_loss+=loss
            # Backpropagation
            #print (loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            boxes_target,boxes_pred=utils.get_bboxes(y.detach().cpu().numpy() ,pred.detach().cpu().numpy() )#used for computing map
            maps.append(mAP.mean_average_precision(boxes_target,boxes_pred))
            loop.set_description_str(f"Train_loss : {(epoch_loss/(batch+1)):>7f}, mAP : {np.mean(maps):>7f} Epoch:{epoch+1}/{epochs}")
        
        torch.save(model.state_dict(), "checkpoint.pth")
        model.eval()
        maps_test=[]
        #losses=[]
        test_loss=0
        for batch in range(len(TestLoader)):
            X, y=Dataloader.__getitem__(batch)
            if len(X)==0:
                break
            #X=torch.stack(X)
            X, y = X.to(0), y.to(0)
            # Compute prediction error
            pred = model(X)
            
 
            loss = loss_fn( y,pred)
            test_loss+=loss.item()

            boxes_target,boxes_pred=utils.get_bboxes(y.detach().cpu().numpy() ,pred.detach().cpu().numpy(),test=True )#used for computing map
            maps_test.append(mAP.mean_average_precision(boxes_target,boxes_pred))
            #loop.set_description_str(f"Train_loss : {(epoch_loss/(batch+1)):>7f}, mAP : {np.mean(maps):>7f} Epoch:{epoch+1}")
        train_dict[epoch]=[(epoch_loss/(batch+1)),np.mean(maps),test_loss/(batch+1),np.mean(maps_test)]
        print (f"test : {(epoch_loss/(batch+1)):>7f}, mAP : {np.mean(maps):>7f} ")
        with open ("log.pkl","wb") as file:
            pickle.dump(train_dict,file)
       
        X=X.detach().cpu().numpy()
        utils.plot(X,boxes_pred)
        if batch % 100 == 0:
            loss, current = epoch_loss/(batch+1), (batch + 1) 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] epoch {epoch}")
        accuracy_test,loss_test = compute_test()
        accuracy = compute_accuracy()
        train_losses.append(epoch_loss/(batch+1))
        test_losses.append(loss_test)
        train_accuracies.append(accuracy)
        test_accuracies.append(accuracy_test)
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy_train: {accuracy:.4f}, Accuracy_test: {accuracy_test:.4f} ,Loss_test: {loss_test:.4f}   ==================>")
if __name__=="__main__":
    train()