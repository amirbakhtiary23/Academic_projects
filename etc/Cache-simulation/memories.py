from math import log
from random import choice
from copy import deepcopy
from controller import CCU    
global replacement_policy;
replacement_policy="wb";
class memorySim():
    
    def __init__(self,addresszie):
        self.addresssize=addresszie;
        self.createMemory();
        
    def createMemory(self):
        self.memory_array=[];
        memory_size=pow(2,self.addresssize);
        for j in range (memory_size):
            self.memory_array.append(self.dataGenerator());
            
    def dataGenerator(self):
        data='';
        for i in range(8):
            data=data+choice(("0","1"));
        return data;
    
    def write(self,index,data):
        self.memory_array[index]=data;
        
    def read(self,index):
        return self.memory_array[index];
    
        
    
class cacheMemorySim():

    def __init__(self,cache_size,block_size,kway=1):
        
        self.cache_size=cache_size;
        self.block_size=block_size;
        self.number_of_blocks=int(cache_size/block_size);
        self.kway=kway;
        self.memory=[];
        self.createMemory();
        
    def createMemory(self):
              
        if self.kway>0: 
            self.number_of_sets=int(self.number_of_blocks/self.kway);
        
        else : 
            self.number_of_sets=1;
        number_of_blocks_per_set=int(self.number_of_blocks/self.number_of_sets)
        
        for i in range (self.number_of_sets):
            self.memory.append([]);
            
        for i in self.memory:
            for j in range(number_of_blocks_per_set):
                i.append([]);
                
        for i in self.memory:
            for j in i:
                for f in range(self.block_size):
                    j.append("0"*8);
        
        

"""adds=range(0,64)
memory=memorySim(8);
ccu=CCU(8,64,4,replacement_policy,1);
cache=cacheMemorySim(64,4,1);
#cache.memory[0][1][2]="00011010"
#ccu.directory[0][0]='0100001'
#ccu.directory[0][1]='0010100'
#ccu.directory[0][1]='0000100';
#print(cache.memory);

print ("directory :")
#print (memory.memory_array);
ccu.directory[12][0]="10010";
print (ccu.directory);
ccu.lfu[0][0]=-1;
print (ccu.lfu);
#print (ccu.directory)
for i in [84,85,86,87]:
    print (memory.memory_array[i]);
ccu.wrOperation(243,cache,memory,"w","11001100");
print(ccu.directory);

for i in adds:
    x=ccu.addressMaker(i)
    y=ccu.offset_setNumberExtractor(i)
    print ("Address "+str(i)+" : "+x+" "+" tag : "+ccu.tagExtractor(i)+" b# :" 
           +str(y[0])+" offset : "+str(y[1]));"""
    


            
        
        
        
    
