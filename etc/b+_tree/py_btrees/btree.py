import bisect
from typing import Any, List, Optional, Tuple, Union, Dict, Generic, TypeVar, cast, NewType
from py_btrees.disk import DISK, Address
from py_btrees.btree_node import BTreeNode, KT, VT, get_node
from math import ceil
"""
----------------------- Starter code for your B-Tree -----------------------

Helpful Tips (You will need these):
1. Your tree should be composed of BTreeNode objects, where each node has:
    - the disk block address of its parent node
    - the disk block addresses of its children nodes (if non-leaf)
    - the data items inside (if leaf)
    - a flag indicating whether it is a leaf

------------- THE ONLY DATA STORED IN THE `BTree` OBJECT SHOULD BE THE `M` & `L` VALUES AND THE ADDRESS OF THE ROOT NODE -------------
-------------              THIS IS BECAUSE THE POINT IS TO STORE THE ENTIRE TREE ON DISK AT ALL TIMES                    -------------

2. Create helper methods:
    - get a node's parent with DISK.read(parent_address)
    - get a node's children with DISK.read(child_address)
    - write a node back to disk with DISK.write(self)
    - check the health of your tree (makes debugging a piece of cake)
        - go through the entire tree recursively and check that children point to their parents, etc.
        - now call this method after every insertion in your testing and you will find out where things are going wrong
3. Don't fall for these common bugs:
    - Forgetting to update a node's parent address when its parent splits
        - Remember that when a node splits, some of its children no longer have the same parent
    - Forgetting that the leaf and the root are edge cases
    - FORGETTING TO WRITE BACK TO THE DISK AFTER MODIFYING / CREATING A NODE
    - Forgetting to test odd / even M values
    - Forgetting to update the KEYS of a node who just gained a child
    - Forgetting to redistribute keys or children of a node who just split
    - Nesting nodes inside of each other instead of using disk addresses to reference them
        - This may seem to work but will fail our grader's stress tests
4. USE THE DEBUGGER
5. USE ASSERT STATEMENTS AS MUCH AS POSSIBLE
    - e.g. `assert node.parent != None or node == self.root` <- if this fails, something is very wrong

"""


class BTree:
    def __init__(self, M: int, L: int):
        """
        Initialize a new BTree.
        You do not need to edit this method, nor should you.
        """
        self.root_addr: Address = DISK.new() # Remember, this is the ADDRESS of the root node
        # DO NOT RENAME THE ROOT MEMBER -- LEAVE IT AS self.root_addr
        DISK.write(self.root_addr, BTreeNode(self.root_addr, None, None, True))
        self.M = M # M will fall in the range 2 to 99999
        self.L = L # L will fall in the range 1 to 99999

    def insert(self, key: KT, value: VT) -> None:
        """
        Insert the key-value pair into your tree.
        It will probably be useful to have an internal
        _find_node() method that searches for the node
        that should be our parent (or finds the leaf
        if the key is already present).

        Overwrite old values if the key exists in the BTree.

        Make sure to write back all changes to the disk!
        """
        x=self._find_node(key)

        # Check if the key is already present in the leaf node
        x.insert_data(key,value)
        x.write_back()
        self.split(x)

    def _find_node(self,key):
        x=get_node(self.root_addr)
        index =x.find_idx( key)
        while not x.is_leaf:
            index =x.find_idx( key)
            try:
                if x.keys[index]==key:
                    index+=1
            except : 
                pass
            x=x.get_child(index)
            
        return x
    
    def split (self,x):
        if x.is_leaf:#len(x.keys)>self.M -1 :
            #if self.M%2==1:
            if len(x.keys)>self.L:
                median_index = ceil((self.L)/2)
                median = x.keys[median_index]
                temp1=x.keys[0:median_index] 
                #temp1_data=x.data[0:median]
                temp2= x.keys[median_index:]
                #temp2_data=x.data[median:]
                if x.parent_addr is None:#Meaning that we only had one node
                    temp1_data=x.data[0:median_index]
                    temp2_data=x.data[median_index:]
                    
                    new_node1_addr=DISK.new()
                    new_node2_addr=DISK.new()
                    new_node1= BTreeNode(new_node1_addr, x.my_addr, 0, True)
                    new_node2=BTreeNode(new_node2_addr, x.my_addr, 1, True)
                    new_node1.keys=temp1
                    new_node1.data=temp1_data
                    new_node2.keys=temp2
                    new_node2.data=temp2_data
                    x.keys=[median]
                    x.children_addrs=[new_node1_addr,new_node2_addr]
                    x.data.clear()
                    x.is_leaf=False
                    DISK.write(new_node1_addr, new_node1)
                    DISK.write(new_node2_addr, new_node2)
                    x.write_back()

                else : #When we have more than one node 
                    median_index = ceil((self.L)/2)
                    median = x.keys[median_index]
                    temp1=x.keys[0:median_index] 
                    temp1_data=x.data[0:median_index]
                    temp2= x.keys[median_index:]
                    temp2_data=x.data[median_index:]
                    #temp1_children_addr=x.children_addrs[:median]
                    #temp2_children_addr=x.children_addrs[median:]
                    new_node1_addr=DISK.new()
                    new_node2_addr=DISK.new()
                    new_node1= BTreeNode(new_node1_addr, x.parent_addr, 0, True)
                    new_node2=BTreeNode(new_node2_addr, x.parent_addr, 1, True)
                    new_node1.keys=temp1
                    new_node1.data=temp1_data
                    new_node2.keys=temp2
                    new_node2.data=temp2_data
                    
                    x.data.clear()
                    parent=get_node(x.parent_addr)
                    idx=parent.insert_key(median,new_node1_addr,new_node2_addr)
                    new_node1.index_in_parent=idx
                    new_node2.index_in_parent=idx+1
                    parent.write_back()
                    x.is_leaf=False
                    x.write_back()
                    DISK.write(new_node1_addr, new_node1)
                    DISK.write(new_node2_addr, new_node2)
                    self.split(parent)
        else : 
            if len(x.keys)>self.M:
                median_index = ceil((self.M-1)/2)
                median = x.keys[median_index]
                temp1=x.keys[0:median_index] 
                #temp1_data=x.data[0:median]
                temp2= x.keys[median_index+1:]
                #temp2_data=x.data[median:]
                if x.parent_addr is None:#Meaning that we only had one node
                    temp1_children_addr=x.children_addrs[:median_index+1]
                    temp2_children_addr=x.children_addrs[median_index+1:]
                    new_node1_addr=DISK.new()
                    new_node2_addr=DISK.new()
                    new_node1= BTreeNode(new_node1_addr, x.my_addr, 0, False)
                    new_node2=BTreeNode(new_node2_addr, x.my_addr, 1, False)
                    new_node1.keys=temp1
                    #new_node1.data=temp1_data
                    new_node1.children_addrs=temp1_children_addr
                    new_node2.children_addrs=temp2_children_addr
                    new_node2.keys=temp2
                    #new_node2.data=temp2_data
                    x.keys=[median]
                    x.children_addrs=[new_node1_addr,new_node2_addr]
                    x.data.clear()
                    DISK.write(new_node1_addr, new_node1)
                    DISK.write(new_node2_addr, new_node2)
                    x.write_back()
                else : #When we have more than one node 
                    median_index = ceil((self.L-1)/2)
                    median = x.keys[median_index]
                    temp1=x.keys[0:median_index] 
                    #temp1_data=x.data[0:median]
                    temp2= x.keys[median_index+1:]
                    temp1_children_addr=x.children_addrs[:median_index+1]
                    temp2_children_addr=x.children_addrs[median_index+1:]
                    #temp1_children_addr=x.children_addrs[:median]
                    #temp2_children_addr=x.children_addrs[median:]
                    new_node1_addr=DISK.new()
                    new_node2_addr=DISK.new()
                    new_node1= BTreeNode(new_node1_addr, x.parent_addr, None, False)
                    new_node2=BTreeNode(new_node2_addr, x.parent_addr, None, False)
                    new_node1.keys=temp1
                    #new_node1.data=temp1_data
                    new_node2.keys=temp2
                    new_node1.children_addrs=temp1_children_addr
                    new_node2.children_addrs=temp2_children_addr
                    #new_node2.data=temp2_data
                    DISK.write(new_node1_addr, new_node1)
                    DISK.write(new_node2_addr, new_node2)
                    parent=get_node(x.parent_addr)
                    idx=parent.insert_key(median,new_node1_addr,new_node2_addr)
                    new_node1.index_in_parent=idx
                    new_node2.index_in_parent=idx+1
                    parent.write_back()
                    x.write_back()
                    self.split(parent)



        #DISK.write(temp_adress, BTreeNode(temp_adress, None, None, True))
    def find(self, key: KT) -> Optional[VT]:
        """
        Find a key and return the value associated with it.
        If it is not in the BTree, return None.

        This should be implemented with a logarithmic search
        in the node.keys array, not a linear search. Look at the
        BTreeNode.find_idx() method for an example of using
        the builtin bisect library to search for a number in 
        a sorted array in logarithmic time.
        """
        x=self._find_node(key)
        if key in x.keys:
            return x.find_data(key)

    def delete(self, key: KT) -> None:
        if not self.find(key):
            return None
        x=get_node(self.root_addr)
        flag=False
        index =x.find_idx( key)
        while not x.is_leaf:
            index =x.find_idx( key)
            try:
                if x.keys[index]==key:
                    
                    index+=1
            except : 
                
                pass
            x=x.get_child(index)
        index =x.find_idx( key)
        del x.keys[index]
        del x.data[index] 
        x.write_back()

