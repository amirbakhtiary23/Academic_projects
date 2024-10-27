from math import log
from random import choice
from copy import deepcopy

class CCU():
    
    def __init__(self,addresssize,cachesize,blocksize,policy:str,
                 kway=1,validbit=1,modifiedbit=1,replacementbit=1):
        self.miss_counter=0;
        self.hit_counter=0;
        self.policy=policy;
        self.addresssize=addresssize;
        self.cachesize=cachesize;
        self.blocksize=blocksize;
        self.kway=kway;
        self.validbit=validbit;
        self.modifiedbit=modifiedbit;
        self.replacementbit=replacementbit;
        self.directory=[];
        self.lfu=[];
        self.createDirectory();
        
    def createDirectory(self):
        self.cach_address_lenght=int(log(self.cachesize,2));
        self.block_address_lenght=int(log(self.blocksize,2));#offset
        
        if (self.kway>0):
            self.k_lenght=int(log(self.kway,2));
            
        else : 
            self.k_lenght=self.cach_address_lenght-self.block_address_lenght;
            
        self.tag_lenght=self.addresssize-self.cach_address_lenght+self.k_lenght;
        entry_lenght=self.validbit+self.modifiedbit+self.replacementbit+self.tag_lenght;
        
        if self.kway>0: 
            self.number_of_sets=pow(2,
                                                self.cach_address_lenght-self.block_address_lenght-self.k_lenght);
        
        else : 
            self.number_of_sets=1;
            
        print ("Memory size in bytes : 2^"+str(self.addresssize));
        print ("Cache size in bytes : 2^"+str(self.cach_address_lenght));
        print ("Block size in bytes : 2^"+str(self.block_address_lenght));
        print ("Number of sets in cache : " +str(self.number_of_sets));
        print ("Number of blocks in each set : "+str(int(pow(2,self.k_lenght))));
        print ("Tag field lenght : "+str(self.tag_lenght));
        
        for i in range (self.number_of_sets):
            self.directory.append([]);
            self.lfu.append([]);
        for i in self.directory:
            for j in range(pow(2,self.k_lenght)):
                i.append("0"*entry_lenght);
        for i in self.lfu:
            for j in range(pow(2,self.k_lenght)):
                i.append(0);
                
    def addressMaker(self,address):
        

        bin_address=bin(address)[2:];
        full_address="0"*(self.addresssize-len(bin_address))+bin_address;
        #tag=full_address[0:self.tag_lenght];
        return full_address;
    
    
    def tagExtractor(self,address):

        return self.addressMaker(address)[0:self.tag_lenght];
    
    
    def set_offsetNumberExtractor(self,address):


        cache_Address=self.addressMaker(address)[self.tag_lenght:];
        
        if self.number_of_sets>1:
            return (int(cache_Address[:int(log(self.number_of_sets,2))],2),
                    int(cache_Address[int(log(self.number_of_sets,2)):],2),
                    cache_Address[:int(log(self.number_of_sets,2))],
                    cache_Address[int(log(self.number_of_sets,2)):]);
            
        else :
            return (0,int(cache_Address,2),"",
                    cache_Address);
        
    def wrOperation(self,address,cacheMemory,mainMemory,command:str,input_data=''):

        flag=0;
        hit_varriable=0;
        valid=0;
        tag=self.tagExtractor(address);
        set_offset=self.set_offsetNumberExtractor(address);
        set_number=set_offset[0];
        offset=set_offset[1];
        binary_set_number=set_offset[2]
        binary_offset=set_offset[3]
        #print ("full Address " + tag+binary_set_number+binary_offset)
        destination_address=tag+binary_set_number+binary_offset
        #print (tag)
        #print (set_number)
        #print (offset)
        #print ("checking " +destination_address)
        #print (tag);
        
        for i in range(len(self.directory[set_number])):
            
            j=i;   
            if self.directory[set_number][i][0:self.tag_lenght]==tag :
                hit_varriable=1;
                if self.directory[set_number][i][-3]=="1":#checking if valid bit is 1
                        #valid bit tanha baraye ebtedaye kar estefade mishavad va 0 bodan an neshan dahandeye compulsory miss ast.
                    self.lfu[set_number][i]+=1;
                    valid=1;
                    
                flag=1;
                data=str(cacheMemory.memory[set_number][i][offset]);
                break
                
            if flag:
                break
            
        #print ("hit var : " +str(hit_varriable))
        if (hit_varriable==1 & valid==1) : 
            if command == "r" :print ("Address "+str(address)+" : hit. Data : "+data);
            else : 
                cacheMemory.memory[set_number][i][offset]=input_data;
                self.lfu[set_number][i]+=1;
                if self.policy=="wt":mainMemory.memory_array[address]=input_data;
            self.hit_counter+=1;
            self.directory[set_number][i]=self.directory[set_number][i][:-2]+"1"+self.directory[set_number][i][-1];
            hit_varriable=0;
            return;
        
        elif (hit_varriable==1 and  valid==0) :
            #print(valid)
            print ("Address "+str(address)+" : miss, replacing...");
            self.miss_counter+=1;
            
        else : 
            print ("Address "+str(address)+" :  miss, replacing...");
            self.miss_counter+=1;
        index=self.replace(address,set_number,offset,tag,binary_set_number,binary_offset,cacheMemory,mainMemory);
        if command=="w":
            cacheMemory.memory[index[0]][index[1]][offset]=input_data;
            self.lfu[index[0]][index[1]]+=1;
            if self.policy=="wt": mainMemory.memory_array[address]=input_data;
            self.directory[index[0]][index[1]]=self.directory[index[0]][index[1]][:-2]+"1"+self.directory[index[0]][index[1]][-1];
            print ("Addresses "+str(address)+" written in memory and cache set#"+str(index[0])+" b#"+str(index[1]))
        else:
            self.lfu[index[0]][index[1]]+=1;
            print ("Data : ",str(cacheMemory.memory[index[0]][index[1]][offset]))
        
    def replace(self,address:int,set_number:int,offset:int,tag:str,binary_set_add:str,binary_off:str,cacheMemory,mainMemory):
        if self.policy=="wt":
            return self.wtReplacement(address,set_number,offset,tag,binary_set_add,binary_off,cacheMemory,mainMemory);
            
        if self.policy == "wb":
            return self.wbReplacement(address,set_number,offset,tag,binary_set_add,binary_off,cacheMemory,mainMemory);
            
    
    def wtReplacement(self,address,set_number,offset,tag,binary_set_add,binary_offset_add,cacheMemory,mainMemory):
        print ("wtReplacement")
        word_number=offset
        offsets=[]
        block=[];
        memory_addresses=[];
        binary_address=tag+binary_set_add;
        mem_address=binary_address+binary_offset_add;
        #print ("full address : "+mem_address);
        
        
        for i in range(pow(2,len(binary_offset_add))):
            offset=bin(i)[2:];
            offset=((len(binary_offset_add)-len(offset))*"0")+offset;#digit extend
            #offsets.append(((self.block_address_lenght-len(offset))*"0")+offset)
            #print ("offset is "+offset)
            offset=binary_address+offset;
            memory_addresses.append(int(offset,2));
        #for j in memory_addresses:
        #    block.append(mainMemory.memory_array[j]);
        #print (memory_addresses);
        for i in memory_addresses:
            block.append(mainMemory.memory_array[i])
        #print ("set : "+str(set_number))
        block_to_be_replaced=self.lfu[set_number].index(min(self.lfu[set_number]));
        #print ("block" + str(block_to_be_replaced))
        cacheMemory.memory[set_number][block_to_be_replaced]=deepcopy(block);
        self.directory[set_number][block_to_be_replaced]=tag+"1"+"0"+"0";
        self.lfu[set_number][block_to_be_replaced]=0;
        #print (set_number,block_to_be_replaced);
        return set_number,block_to_be_replaced,word_number,memory_addresses;
    
    
    
    def wbReplacement(self,address,set_number,offset,tag,binary_set_add,binary_offset_add,cacheMemory,mainMemory):
       
        print ("wbReplacement")
        word_number=offset;
        offsets=[];
        block=[];
        offsets_for_replace=[]
        memory_addresses=[];
        binary_address=tag+binary_set_add;
        mem_address=binary_address+binary_offset_add;
        
        print ("Address "+str(address)+" ==> "+mem_address);
        
        
        for i in range(pow(2,len(binary_offset_add))):
            offset=bin(i)[2:];
            offset=((len(binary_offset_add)-len(offset))*"0")+offset;#digit extend
            #offsets.append(((self.block_address_lenght-len(offset))*"0")+offset)
            #print ("offset is "+offset)
            offset=binary_address+offset;
            memory_addresses.append(int(offset,2));
        #for j in memory_addresses:
        #    block.append(mainMemory.memory_array[j]);
        #print (memory_addresses);
        for i in memory_addresses:
            block.append(mainMemory.memory_array[i])
        #print ("set : "+str(set_number))
        block_to_be_replaced=self.lfu[set_number].index(min(self.lfu[set_number]));
        temp=deepcopy(cacheMemory.memory[set_number][block_to_be_replaced]);
        #print (block_to_be_replaced)
        
        if self.directory[set_number][block_to_be_replaced][-2]=="1":
            #print ("writing in memory")
            if self.kway>0:
                block_to_be_replaced_tag=self.directory[set_number][block_to_be_replaced][:-3]
                block_to_be_replaced_binary_set_number=(self.k_lenght-len(bin(set_number)[2:]))*"0"
                block_to_be_replaced_binary_set_number= block_to_be_replaced_binary_set_number+bin(set_number)[2:]
                offset_bits=int(log(self.blocksize,2));
                for i in range(self.blocksize):
                    bin_address=bin(i)[2:]
                    bin_address=((offset_bits-len(bin_address))*"0")+bin_address
                    offsets_for_replace.append(int(block_to_be_replaced_tag+block_to_be_replaced_binary_set_number+bin_address,2));
                    
                    
                
            else:
                block_to_be_replaced_tag=self.directory[set_number][block_to_be_replaced][:-3]
                #print (block_to_be_replaced_tag)
                offset_bits=int(log(self.blocksize,2));
                for i in range(self.blocksize):
                    bin_address=bin(i)[2:];
                    bin_address=((offset_bits-len(bin_address))*"0")+bin_address;
                    offsets_for_replace.append(int(block_to_be_replaced_tag+bin_address,2));
                    
                    
            for j in range(self.blocksize):
                #print (offsets_for_replace[j]);
                mainMemory.memory_array[offsets_for_replace[j]]=temp[j];
                print ("writing Address "+str(offsets_for_replace[j])+" ==> " +mainMemory.memory_array[offsets_for_replace[j]]+" to main memory")


                
                
        print ("block" + str(block_to_be_replaced))
        cacheMemory.memory[set_number][block_to_be_replaced]=deepcopy(block);
        self.directory[set_number][block_to_be_replaced]=tag+"1"+"0"+"0";
        print("tag is : "+tag+"1"+"0"+"0")
        self.lfu[set_number][block_to_be_replaced]=0;
        #print (set_number,block_to_be_replaced,word_number,memory_addresses);
        return set_number,block_to_be_replaced,word_number,memory_addresses;