#!/usr/bin/env python 
#encoding=utf-8

class CQueue:
    '''round-robin queue'''
    def __init__(self,MaxSize):
        self.MaxSize = MaxSize
        self.values = [None]*self.MaxSize
        self.rear = 0
        self.front = 0

    def isFull(self):
        return (self.rear+1)%self.MaxSize == self.front

    def isEmpty(self):
        return self.front == self.rear

    def delete(self):
        val = None
        if not self.isEmpty():
            val = self.values[self.front]
            self.front = (self.front+1) % self.MaxSize
        return val

    def append(self,value):
        if self.isFull():
            self.delete()
        self.values[self.rear]=value
        self.rear = (self.rear+1)%self.MaxSize

    def getAll(self):
        res = [None]*self.length()
        i, idx = self.front, 0
        while i != self.rear:
            # print("check",i,idx)
            res[idx]=self.values[i]
            i, idx = (i+1)%self.MaxSize, idx+1
        return res

    def display(self):
        print("==================queue info===============")
        for k in self.__dict__:
            print("%s=%s"%(k,self.__dict__[k]))

    def length(self):
        return (self.rear + self.MaxSize - self.front)%self.MaxSize

    def getHead():
        if self.isEmpty():
            return None
        return self.values[self.front]

    def getRear(self):
        if self.isEmpty():
            return None
        return self.values[(self.rear -1 + self.MaxSize)%self.MaxSize]

    def reset(self):
        self.rear = 0
        self.front = 0
        for idx in range(len(self.values)):
            self.values[idx] = None

def get_k_b(X,Y):
    A,B,C,D,n = 0,0,0,0,0
    for k in range(len(X)):
        x, y = X[k], Y[k]
        A += x*x
        B += x
        C += x*y
        D += y
        n += 1
    slope = (C*n-B*D) / (A*n - B*B)
    trunc = (A*D - C*B) / (A*n - B*B)
    return slope, trunc

# test-queue
'''
if __name__=="__main__":
    q = CQueue(7)
    q.append(3)
    print("here 1:len=%d|values=%s"%(q.length(),q.getAll()))
    q.display()
    q.append(4)
    q.append(10)
    q.append(12)
    q.append(13)
    q.append(17)
    print("here 2:len=%d|values=%s"%(q.length(),q.getAll()))
    q.display()
    q.append(21)
    print("here 3:len=%d|values=%s"%(q.length(),q.getAll()))
    q.display()
    q.append(36)
    print("here 4:len=%d|values=%s"%(q.length(),q.getAll()))
    q.display()
    q.delete()
    print("here 5:len=%d|values=%s"%(q.length(),q.getAll()))
    q.display()
'''