import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

x_test=np.load("data/test_x.npy")
x_train=np.load("data/train_x.npy")
y_test=np.load("data/test_y.npy")
y_train=np.load("data/train_y.npy")
def normalize(data,mx,mn):
    data=data.astype("float32")
    data=(data-mn)/(mx-mn)
    return data
min_=np.min(x_train)
max_=np.max(x_train)
x_test=normalize(x_test,max_,min_)
x_train=normalize(x_train,max_,min_)

print (x_test.shape,"x_test")
print (x_train.shape,"x_train")
print (y_train.shape,"y_train")
print (y_test.shape,"y_test")

num_labels=np.unique(y_test).shape[0]
class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 768)
        self.bn2=nn.BatchNorm1d(768)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(768, 384)
        self.bn3=nn.BatchNorm1d(384)
        self.relu3 = nn.ReLU()
        self.cls=nn.Linear(384,20)
        self.softmax=nn.Softmax()
    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.softmax(self.cls(x))
        return x
class Data(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}

        return sample
def setup():
    # initialize the process group
    dist.init_process_group(backend="gloo")#, rank=rank, world_size=world_size)
def train(snapshot_path,save_snapshot_parameter):
    setup()
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    predictions_list = []
    labels_list = []
    rank=dist.get_rank()
    world_size    = int(os.environ["WORLD_SIZE"])
    print (rank)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10
    train_dataset = Data(x_train, y_train)
    test_dataset = Data(x_test, y_test)

    timer=time.time()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                               pin_memory=True,persistent_workers=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               sampler=test_sampler,
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                               pin_memory=True,persistent_workers=True)
    #int(os.environ["SLURM_CPUS_PER_TASK"])
    model = Model()
    model=DDP(model)
    starting_epoch=load_snap(snapshot_path,model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(starting_epoch,num_epochs):
        if rank==0 and epoch==save_snapshot_parameter:
            save_snap(snapshot_path,model,epoch)
        for i, sample in enumerate(train_loader):

            data, labels = sample['data'], sample['label']
            outputs = model(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f} , rank : {rank}')
    # Test 
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sample in test_loader:
            data, labels = sample['data'], sample['label']
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}% on epoch {epoch+1} on rank {rank}')
    print("Rank: {}, Training Time: {}".format(rank, time.time() - timer))
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
    start_time = time.time()
    train(snapshot_path,save_snapshot_parameter)
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))
