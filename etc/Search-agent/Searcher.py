import random
from copy import deepcopy
#allowed moves: up , upleft , upright
size= input()
sols = {}
n = int (size.split()[0])
m = int (size.split()[1])

initial_location=[0,random.randint(0,m-1)] # entekhab yek addad random baraye shoroe jostojo
grid = [] # tarif yek list khali baraye ijad jadval

for i in range(n):
    inp = input ()
    grid.append([])
    for j in inp.split():
        grid[i].append(int(j))

grid.reverse()

NumberOfNodes = (pow(3,n)-1)/2


class MakeTree():
    def __init__(self,grid,initial):
        self.grid=deepcopy(grid)
        self.tree=[]# derakht tohi
        self.initial=initial
        self.p1=initial[0]
        self.p2=initial[1]
        self.tree.append([self.p1,self.p2,grid[self.p1][self.p2]])
        self.make()
        
    def make(self):
        for i in self.tree:
            if len(self.tree)==NumberOfNodes: break #
            if i==None:
                self.tree.append(None)
                continue
            if i[0]<n-1:
                self.tree.append([i[0]+1,i[1],self.grid[i[0]+1][i[1]]])
            else : continue
            if i[1]!=0:
                self.tree.append([i[0]+1,i[1]-1,self.grid[i[0]+1][i[1]-1]])
            else :
                self.tree.append(None)
            if i[1]<m-1:
                self.tree.append([i[0]+1,i[1]+1,self.grid[i[0]+1][i[1]+1]])
            else :
                self.tree.append(None)
    def returnTree(self):
        return self.tree
                         
 
class Search():
    def __init__(self,tree):
        self.Tree=deepcopy(tree)
        self.Tree=[0]+self.Tree
        self.values=[]
        self.lenght=len(self.Tree)
        self.calculate()
    def calculate(self):
        for i in reversed(range(1,self.lenght)):
            if self.Tree[i]==None:
                continue
            index=i
            point=0
            moves=[]
            while True:
                if index==0:break
                if self.Tree[index]==None: break
                point = self.Tree[index][2]+point
                moves.append(self.Tree[index][1])
                
                if self.Tree[index][0]==n :
                    break
                
                if index%3==0:
                    nxt=int(index/3)
                elif (index+1)%3==0:
                    nxt=int((index+1)/3)
                elif (index-1)%3==0:
                    nxt=int((index-1)/3)
                index=nxt
            sols[point]=moves
            
        
#print (grid)
print (initial_location[1]+1)
Make = MakeTree(grid,initial_location)
Tree=Make.returnTree()

search=Search(Tree)
max_point=max(sols.keys())
print (max_point)
pnts=[]
for i in sols[max_point]:
    pnts.append(str(i+1))
print (" ".join(pnts))
       

