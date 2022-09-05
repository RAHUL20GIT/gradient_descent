import numpy as np

X = np.array([i for i in range(100)], dtype=np.float32)
Y = np.array([2 * i for i in range(100)], dtype=np.float32)

w = 0.0

TEST=1020

def forward(x):
    return w * x


def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


print(f"Prediction before training{forward(TEST):.3f}")

learning_rate = 0.001
iterations = 100

Y_pred = forward(X)
l = loss(Y, Y_pred)
# print(l)
dw = gradient(X, Y, Y_pred)
w -= learning_rate * dw
w1 = w

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

# the first

wmid=(w1+w2)/2
print(w1,'   ',w2,"  ",wmid)
i=0
for epoch in range(iterations):
  w=wmid
  i=i+1
  Y_pred = forward(X)
  l = loss(Y, Y_pred)

  if(abs(forward(TEST)-2040)<10):
    break
  if(wmid>0):
    w1=wmid
  elif(wmid<0):
    w2=wmid

  wmid=(w1+w2)/2
  print(w1,'   ',w2,"  ",wmid)

learning_rate = 0.000001
for epoch in range(10):
  Y_pred = forward(X)
  l = loss(Y, Y_pred)
  dw = gradient(X, Y, Y_pred)
  w -= learning_rate * dw
  print(f"{forward(1020):.3f}")
# model prediction