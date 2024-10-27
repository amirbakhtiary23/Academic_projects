import random
global point
from copy import deepcopy
point=[0,0]

seq1=[2,4]
seq2=list(range(0,4)) 
global matrix
matrix=[]

for i in range(4): 
    matrix.append([0]*4)
def evaluation():
    indexes=[]
    for i in range(4):
        for j in range(4) :
            if matrix[i][j]==2048:
                return "won"
            if matrix[i][j]==0:
                indexes.append([i,j])
    #print (indexes)
    if len (indexes)==0:
        print (matrix)

        x=deepcopy(ml(matrix,0))
        x1=deepcopy(mu(matrix,0))
        x2=deepcopy(md(matrix,0))
        x3=deepcopy(mr(matrix,0))
        if x==matrix and  x1==matrix and x2==matrix and x3==matrix:

            print ("game over")
            return ("go")
        else :
            return "go_on"
    else :
        addtile(indexes)
        return "go_on"

def addtile(indexes):

    choice=random.choice(indexes)
    x1=choice[0]
    x2=choice[1]
    print (choice)
    rnd=random.choice(seq1)
    print ("rnd is : %d" %rnd)
    matrix[x1][x2]=rnd

def mu(matrixx,v):
    for i in range(1,4):
        for j in range(4):
            i1=i
            if matrixx[i][j]==0 and i<3:
                i1=i+1
                if matrixx[i1][j]==0 and i1<3:
                    i1=i1+1
            if (matrixx[i-1][j]==matrixx[i1][j]):
                matrixx[i-1][j]=matrixx[i-1][j]*2
                if v :
                    point[0] = matrixx [i-1][j]+point[0]

                matrixx[i1][j]=0
    for f in range (4):
        for i in reversed(range(1,4)):
            for j in range(4):
                if (matrixx[i-1][j]==0):
                    matrixx[i-1][j]=matrixx[i][j]
                    matrixx[i][j]=0
    return matrixx
def md(matrixx,v):
    for i in reversed(range(0,3)):
        for j in range(4):
            i1=i

            if matrixx[i][j]==0 and i<0:
                i1=i-1
                if matrixx[i1][j]==0 and i1<0:
                    i1=i1-1
            if (matrixx[i+1][j]==matrixx[i1][j]):
                matrixx[i+1][j]=matrixx[i+1][j]*2
                if v : point [0]=matrixx [i+1][j]+point[0]
                matrixx[i1][j]=0
    for f in range(4):
        for i in range(0,3):
            for j in range(4):
                if (matrixx[i+1][j]==0):
                    matrixx[i+1][j]=matrixx[i][j]
                    matrixx[i][j]=0
    return matrixx

def mr(matrixx,v):

    for i in range(0,4):
        for j in reversed(range(0,3)):
            j1=j
            if matrixx[i][j]==0 and j>0:
                j1=j1-1
                if matrixx[i][j1]==0:
                    j1=j1-1
            if (matrixx[i][j+1]==matrixx[i][j1]):
                matrixx[i][j+1]=matrixx[i][j+1]*2
                if v : point[0] =  matrixx [i][j+1]+point[0]

                matrixx[i][j1]=0
    for f in range(4):
        for i in range(0,4):
            for j in range(0,3):
                if (matrixx[i][j+1]==0):
                    matrixx[i][j+1]=matrixx[i][j]
                    matrixx[i][j]=0
    return matrixx
def ml(matrixx,v):
    for i in range(0,4):
        for j in range(1,4):
            j1=j
            if matrixx [i][j]==0 and j<2:
                j1=j+1
                if matrixx[i][j1]==0:
                    j1=j1+1
            if (matrixx[i][j-1]==matrixx[i][j1]):
                matrixx[i][j-1]=matrixx[i][j-1]*2
                if v : point[0] =matrixx [i][j-1]+point[0]
                matrixx[i][j1]=0

    for f in range(4):
        for i in range(0,4):
            for j in range(1,4):
                if (matrixx[i][j-1]==0):
                    matrixx[i][j-1]=matrixx[i][j]
                    matrixx[i][j]=0
    return matrixx



class decision():
    def __init__(self,matr):
        self.matrix=deepcopy(matr)
        self.matrix1=deepcopy(matr)
        self.matrix2=deepcopy(matr)
        self.matrix3=deepcopy(matr)
        self.move=["mu(matrix,1)","md(matrix,1)","mr(matrix,1)","ml(matrix,1)"]

        self.choose=[0,0,0,0]
        self.calculValue()
        self.ret()
    def calculValue(self):
        mx1=deepcopy(mu(self.matrix,0))
        mx2=deepcopy(md(self.matrix1,0))
        mx3=deepcopy(mr(self.matrix2,0))
        mx4=deepcopy(ml(self.matrix3,0))
        #for i in range(4):
        #    for j in range():pass

        self.choose[0]=self.test(mx1)
        self.choose[1]=self.test(mx2)
        self.choose[2]=self.test(mx3)
        self.choose[3]=self.test(mx4)
        #print (self.choose)"""
    """
    def addtile(self,entry):
        emptyTiles=[]
        for i in entry:
            for j in i: """
    def test(self,ent):
        maximum=0
        mx_1=deepcopy(ent)
        mx_2=deepcopy(ent)
        cords=[]
        for i in range(4):
            for j in range(4):
                if ent[i][j]==0:
                    cords.append([i,j])

        for i in cords:
            mx_1[i[0]][i[1]]=2
            y1=self.evaluator(mu(mx_1,0))
            y2=self.evaluator(md(mx_1,0))
            y3=self.evaluator(mr(mx_1,0))
            y4=self.evaluator(ml(mx_1,0))
            mx_1[i[0]][i[1]]=0
            maximum=max(y1,y2,y3,y4,maximum)
        for i in cords:
            mx_2[i[0]]  [i[1]]=4
            y5=self.evaluator(mu(mx_1,0))
            y6=self.evaluator(md(mx_1,0))
            y7=self.evaluator(mr(mx_1,0))
            y8=self.evaluator(ml(mx_1,0))
            mx_2[i[0]][i[1]]=0
            maximum=max(y5,y6,y7,y8,maximum)
        cords=[]
        return maximum





    def evaluator(self,entery):
        tiles=0
        for i in range(4):
            for j in range(4):
                if entery[i][j]==0:
                    tiles+=1
        return tiles
    def ret(self):
        maxx=self.choose.index(max(self.choose))
        if maxx+1<len(self.choose):
            if self.choose[maxx]==self.choose[maxx+1]:

                return self.move[random.randint(maxx,maxx+1)]
        return (self.move[maxx])





x=0
evaluation ()
evaluation ()
#op=["mu(matrix,1)","md(matrix,1)","ml(matrix,1)","mr(matrix,1)"]
for i in matrix:
    print (i)
while True:
    x=x+1
    y=decision(matrix)
    y=y.ret()

    print ("================== move number "+str(x))
    print ("move " + str (y))
    matrix=eval(y)
    for i in matrix:
        print (i)
    print ("point : " +str(point[0]))
    #point[1]=point[1]+point[0]
    r=evaluation()
    del (y)
    if r=="won" or r=="go":
        print (r)
        break
    else : print (r)
