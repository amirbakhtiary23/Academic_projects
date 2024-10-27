#CA2 Q1 Part A, Amir Bakhtiary , 810101114
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import datetime
from torch import nn
import torch.nn.functional as F
import os
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
train_batch_size = 128
test_batch_size = 128
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5),(0.5),)])

def setup(backend):
    #os.environ["MASTER_ADDR"] = 'localhost'
    #os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend)#, rank=rank, world_size=world_size, timeout=timeout)

def load_data_torchrun():
    train_set = FashionMNIST("/home/dmls/bakhtiary/fmnist/", download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              sampler=train_sampler, 
                                              batch_size=train_batch_size, 
                                              shuffle=False, 
                                              persistent_workers=True,
                                              num_workers=1, 
                                              pin_memory=True)
    test_set =FashionMNIST("/home/dmls/bakhtiary/fmnist/", download=True, train=False, transform=transform)  
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                             sampler=test_sampler, 
                                             batch_size=test_batch_size, 
                                             shuffle=False, 
                                             persistent_workers=True,
                                             num_workers=1, 
                                             pin_memory=True)
    return train_loader, test_loader




class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
def train(backend,path,save_snapshot_parameter):
    setup( backend)
    

    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    trainloader, testloader = load_data_torchrun()
    #model = CNN()
    model = CNN().to(rank)
    device=rank
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    starting_epoch=load_snap(path,ddp_model)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    predictions_list = []
    labels_list = []

    start_time = time.time()
    for epoch in range(starting_epoch,10):
        if rank==0 and epoch==save_snapshot_parameter:
            save_snap(path,ddp_model,epoch)
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cuda_mem = torch.cuda.max_memory_allocated(device=device)
            count += 1
            if not (count % 50):
                total = 0
                correct = 0
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)
                    outputs = model(images)
                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)
                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if not (count % 500):
                    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
    end_time = time.time()
    print("Rank: {}, Training Time: {}".format(rank, end_time - start_time))
    print("Rank: {}, Max Memory Allocated: {} MB".format(rank, cuda_mem / (1024 ** 2)))
def load_snap(path,model):
    try:
        dic=torch.load(path)
        model.load_state_dict(dic["MODEL_STATE"])    
        epochs_runed=dic["EPOCHS_RUN"]
        return epochs_runed
    except : 
        return 0
def save_snap(path,model,epoch):
    ckp={}
    ckp["MODEL_STATE"]=model.module.state_dict()
    ckp["EPOCHS_RUN"]=epoch
    torch.save(ckp,path)
if __name__ == "__main__":
    start_time = time.time()
    snapshot_path=os.curdir+"/snapshot.pt"
    save_snapshot_parameter=5
    backend = 'nccl'
    timeout = datetime.timedelta(seconds=10)
    start_time = time.time()
    #mp.spawn(train, nprocs=world_size, args=(world_size, master_port, backend, timeout), join=True)
    train(backend,snapshot_path,save_snapshot_parameter)
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))
