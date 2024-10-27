import controller
import memories
from sys import argv
from random import randrange,choice

hit_time=2;
miss_time=50;
help="""
    doc:
    arg 1= memory address size in bits, help for help
    arg 2=cache size in bytes
    arg 3=block size in bytes
    arg 4= replacement policies : wb=writeback wt=writethrough 
    arg 5= k in k-way ; for fully associative pass 0
    arg 6= number of addresses
    
    """
if argv[0]=="help":
    print (help);
    exit();
def data_generator():
    x='';
    for i in range(8):
        x=x+choice(("0","1"))
    return x
def address_generator(range_,iter):
    addresses=[];
    Range=pow(2,8)#argv[1])
    for i in range(iter):
        addresses.append([randrange(64),choice(("w","r")),data_generator()]);
    return addresses;
iter=10000;
address_size=16#int(argv[0]);
cache_size=64#int(argv[1]);
block_size=4#int(argv[2]);
policy="wt"#argv[3];
k=1#int(argv[4])
if __name__=="__main__":
    ccu=controller.CCU(address_size,cache_size,block_size,policy,k);
    cache=memories.cacheMemorySim(cache_size,block_size,k);
    memory=memories.memorySim(address_size)
    ads=address_generator(address_size,iter);
    
    for i in ads:
        ccu.wrOperation(i[0],cache,memory,i[1],i[2]);
    print (ccu.lfu)
    print ("total number of misses : "+ str(ccu.miss_counter));
    print ("total number of hits : "+ str(ccu.hit_counter));
    
    avrage_access_time=(ccu.hit_counter*1)/iter;
    print (avrage_access_time)
    avrage_access_time=(avrage_access_time*hit_time)+(1-avrage_access_time)*miss_time;
    print ("Average access time:"+str(avrage_access_time));
    print(ccu.directory)

