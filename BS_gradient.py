import numpy as np

X = np.array([i for i in range(100)], dtype=np.float32)
Y = np.array([2 * i for i in range(100)], dtype=np.float32)

w = 0.0

TEST=1020

def forward(x): # makes the predictions
    return w * x


def loss(y, y_pred): # calculates the loss 
    return ((y_pred - y) ** 2).mean()


def gradient(x, y, y_pred): #calculates the gradient
    return np.dot(2 * x, y_pred - y).mean()


print(f"Prediction before training{forward(TEST):.3f}")

learning_rate = 0.001
iterations = 100

""""This part gets us the first value of the weight"""
Y_pred = forward(X)
l = loss(Y, Y_pred)
# print(l)
dw = gradient(X, Y, Y_pred)
w -= learning_rate * dw
w1 = w


""" Code to check change in sign of consecutive gradient signs"""
epchs=100
while(epchs):
    epchs-=1
    Y_pred = forward(X)
    l = loss(Y, Y_pred)
    dw = gradient(X, Y, Y_pred)
    w -= learning_rate * dw
    w2 = w
    if(w2>0 and w1<0):
        break;
    elif(w2<0 and w1>0):
        break
    else:
        w1=w2

"""This part uses the binary search to narrow down the search space . At this time the learning rate is 0.001 """

wmid=(w1+w2)/2
print(w1,'   ',w2,"  ",wmid)
i=0
for epoch in range(iterations):
  w=wmid
  i=i+1
  Y_pred = forward(X)
  l = loss(Y, Y_pred)

  if(abs(forward(TEST)-2040)<10): # this iteration stops when the error is within a given range. HERE IT IS 10 
    break
  if(wmid>0):
    w1=wmid
  elif(wmid<0):
    w2=wmid

  wmid=(w1+w2)/2
  print(w1,'   ',w2,"  ",wmid)
    
    
"""After the binary search the normal gradient descent method is applied"""    

learning_rate = 0.000001
for epoch in range(10):
  Y_pred = forward(X)
  l = loss(Y, Y_pred)
  dw = gradient(X, Y, Y_pred)
  w -= learning_rate * dw
  print(f"{forward(1020):.3f}")
