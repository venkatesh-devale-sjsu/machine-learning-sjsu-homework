
# coding: utf-8

# In[3]:


import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False 

#generate a data set of 20. 
#for simplicity, 10 in the first quadrant, another 10 in the third quadrant 
X1 = []
Y1 = []
X2 = []
Y2 = []

for i in range(10):
    X1.append(random.uniform(0,1))
    Y1.append(random.uniform(0,1))
    X2.append(random.uniform(-1,0))
    Y2.append(random.uniform(-1,0))
    
#label the data
data1 = [np.array([1,X1[i],Y1[i],1]) for i in range(10)]
data2 = [np.array([1,X2[i],Y2[i],-1]) for i in range(10)]
data = data1 + data2


# In[4]:


#Part a) of the Problem 1.4 from the HW - 1
plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'ro')
x = np.linspace(-1,1)
plt.plot(x, -x, color='black', linestyle='-')
#plt.axis([-1, 1, -1, 1])
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()


# In[28]:


#Perceptron Learning Algorithm
class Perceptron(object):
    def __init__(self, data):
        self.W = np.zeros(len(data[0:3]))
        self.update = 0
    
    def predict(self, x):
        activation = np.dot(self.W.T,x)
        return np.sign(activation)
    
    def fit(self, data):
        count = 0
        X = np.array(data)[:,0:3]
        d = np.array(data)[:, 3:4]
        while True:
            self.update = 0
            for i in range(len(data)):
                predicted_value_y = self.predict(X[i])
                expected_value = d[i]
                if expected_value * predicted_value_y <=0:
                    self.W = self.W + expected_value * X[i]
                    count += 1
                    self.update += 1
                    break
            if self.update == 0:
                break
        print("Number of iterations for converging:",count)


# In[29]:


#Part b where count is the number of updates before converging

#Initializing Perceptron Learning Algorithm
perceptron = Perceptron(data)

#Running Perceptron Learning Algorithm
perceptron.fit(data)

#Printing converged weight vector with count
print(perceptron.W)


# In[30]:


#Part b plotting target function f and hypothesis g of the Problem 1.4 from the HW - 1
plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'ro')

plt.plot(x, -x, color='green', linestyle='-')
plt.plot(x, ((-perceptron.W[0]-perceptron.W[1]*x))/perceptron.W[2], color='black', linestyle='-') #plotting hypothesis
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()


# In[31]:


#Problem 1.4 c)
#generate a data set of 20. 
#for simplicity, 10 in the first quadrant, another 10 in the third quadrant 
X1 = []
Y1 = []
X2 = []
Y2 = []

for i in range(10):
    X1.append(random.uniform(0,1))
    Y1.append(random.uniform(0,1))
    X2.append(random.uniform(-1,0))
    Y2.append(random.uniform(-1,0))
    
#label the data
data1 = [np.array([1,X1[i],Y1[i],1]) for i in range(10)]
data2 = [np.array([1,X2[i],Y2[i],-1]) for i in range(10)]
data = data1 + data2


# In[32]:


#Running PLA for sub question 'c'
perceptron = Perceptron(data)
perceptron.fit(data)
print("Final weight vector:",perceptron.W)


# In[33]:


#Plotting for sub question 'c'
plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'ro')
x = np.linspace(-1,1)
plt.plot(x, -x, color='green', linestyle='-')
plt.plot(x, ((-perceptron.W[1]*x))/perceptron.W[2], color='black', linestyle='-')
#plt.axis([-1, 1, -1, 1])
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()


# In[37]:


#Problem 1.4 d)
#generate a data set of 100. 
#for simplicity, 50 in the first quadrant, another 10 in the third quadrant 

X1 = []
Y1 = []
X2 = []
Y2 = []

for i in range(50):
    X1.append(random.uniform(0,1))
    Y1.append(random.uniform(0,1))
    X2.append(random.uniform(-1,0))
    Y2.append(random.uniform(-1,0))
    
#label the data
data1 = [np.array([1,X1[i],Y1[i],1]) for i in range(50)]
data2 = [np.array([1,X2[i],Y2[i],-1]) for i in range(50)]
data = data1 + data2


# In[38]:


perceptron = Perceptron(data)
perceptron.fit(data)
print("Final weight vector:",perceptron.W)
print(perceptron.W[1])
print(perceptron.W[2])
plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'ro')
x = np.linspace(-1,1)
plt.plot(x, -x, color='green', linestyle='-')
plt.plot(x, ((-perceptron.W[1]*x))/perceptron.W[2], color='black', linestyle='-')
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()


# In[39]:


#Problem 1.4 d)
#generate a data set of 1000. 
#for simplicity, 500 in the first quadrant, another 10 in the third quadrant 

X1 = []
Y1 = []
X2 = []
Y2 = []

for i in range(500):
    X1.append(random.uniform(0,1))
    Y1.append(random.uniform(0,1))
    X2.append(random.uniform(-1,0))
    Y2.append(random.uniform(-1,0))
    
#label the data
data1 = [np.array([1,X1[i],Y1[i],1]) for i in range(500)]
data2 = [np.array([1,X2[i],Y2[i],-1]) for i in range(500)]
data = data1 + data2


# In[40]:


perceptron = Perceptron(data)
perceptron.fit(data)
print("Final weight vector:",perceptron.W)
print(perceptron.W[1])
print(perceptron.W[2])
plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'ro')
x = np.linspace(-1,1)
plt.plot(x, -x, color='green', linestyle='-')
plt.plot(x, ((-perceptron.W[1]*x))/perceptron.W[2], color='black', linestyle='-')
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()


# In[ ]:


#As part of comparison we can see as we already have well separated data, we get the weight vector optimized each time 
#for that many samples and the hypothesis is away from target function f but classifying the entire data set as needed


# In[36]:


#Problem 1.5 getting the training and test data set
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False 

#generate a data set of 100. 
#for simplicity, 50 in the first quadrant, another 50 in the third quadrant 
train_X1 = []
train_Y1 = []
train_X2 = []
train_Y2 = []

for i in range(50):
    train_X1.append(random.uniform(0,1))
    train_Y1.append(random.uniform(0,1))
    train_X2.append(random.uniform(-1,0))
    train_Y2.append(random.uniform(-1,0))
    
#label the data
train_data1 = [np.array([1,train_X1[i],train_Y1[i],1]) for i in range(50)]
train_data2 = [np.array([1,train_X2[i],train_Y2[i],-1]) for i in range(50)]
train_data = train_data1 + train_data2


# In[18]:


#generate a test data set of 20. 
#for simplicity, 10 in the first quadrant, another 10 in the third quadrant 
test_X1 = []
test_Y1 = []
test_X2 = []
test_Y2 = []

for i in range(5000):
    test_X1.append(random.uniform(0,1))
    test_Y1.append(random.uniform(0,1))
    test_X2.append(random.uniform(-1,0))
    test_Y2.append(random.uniform(-1,0))
    
#label the data
test_data1 = [np.array([1,test_X1[i],test_Y1[i],1]) for i in range(5000)]
test_data2 = [np.array([1,test_X2[i],test_Y2[i],-1]) for i in range(5000)]
test_data = test_data1 + test_data2


# In[55]:


#Problem 1.5 variation of Adaline perceptron learning algorithm
class Perceptron(object):
    def __init__(self, data, learning_rate=100):
        self.W = np.zeros(len(data[0:3]))
        self.epochs = 1000
        self.update = 0
        self.learning_rate = learning_rate
    
    def predict(self, x):
        activation = np.dot(self.W.T,x)
        return np.sign(activation)
    
    def getErrorRate(self, test_data):
        X = np.array(test_data)[:,0:3]
        d = np.array(test_data)[:, 3:4]
        errorCount = 0
        for i in range(len(test_data)):
            predicted_value = self.predict(X[i])
            expected_value = d[i]
            if expected_value != predicted_value:
                errorCount += 1
        return (errorCount)/len(test_data)*100
                
    
    
    def fit(self, data):
        count = 0
        X = np.array(data)[:,0:3]
        d = np.array(data)[:, 3:4]
        while self.update < 1000:
            #self.update = 0
            for i in range(len(data)):
                predicted_value_y = self.predict(X[i])
                expected_value = d[i]
                if expected_value * predicted_value_y <=1:
                    self.W = self.W + self.learning_rate*(expected_value - predicted_value_y) * X[i]
                    #print(self.W)
                    self.update += 1

        #print("Number of iterations for converging:",count)
       


# In[61]:


#Running the algorithm

#Problem 1.5 a)

perceptron = Perceptron(train_data)
perceptron.fit(train_data)
print("Final weight vector:",perceptron.W)
print("Error rate on test data with learning rate {0}:{1}".format(100,perceptron.getErrorRate(test_data)))

#Problem 1.5 b)
perceptron = Perceptron(train_data, 1)
perceptron.fit(train_data)
print("Final weight vector:",perceptron.W)
print("Error rate on test data with learning rate {0}:{1}".format(1,perceptron.getErrorRate(test_data)))

#Problem 1.5 c)
perceptron = Perceptron(train_data, 0.01)
perceptron.fit(train_data)
print("Final weight vector:",perceptron.W)
print("Error rate on test data with learning rate {0}:{1}".format(0.01,perceptron.getErrorRate(test_data)))

#Problem 1.5 d)
perceptron = Perceptron(train_data, 0.0001)
perceptron.fit(train_data)
print("Final weight vector:",perceptron.W)
print("Error rate on test data with learning rate {0}:{1}".format(0.001,perceptron.getErrorRate(test_data)))

#There is no difference here due the nature of the data as already confirmed with professor for my case


# In[62]:


#Plotting on training data
plt.plot(train_X1, train_Y1, 'bo')
plt.plot(train_X2, train_Y2, 'ro')
#x = np.linspace(-1,1)
#plt.plot(x, -x, color='green', linestyle='-')
plt.plot(x, (-perceptron.W[0]-perceptron.W[1]*x)/perceptron.W[2], color='black', linestyle='-')
#plt.axis([-1, 1, -1, 1])
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()


# In[63]:


#Plotting on testing data
plt.plot(test_X1, test_Y1, 'bo')
plt.plot(test_X2, test_Y2, 'ro')
#x = np.linspace(-1,1)
#plt.plot(x, -x, color='green', linestyle='-')
plt.plot(x, (-perceptron.W[0]-perceptron.W[1]*x)/perceptron.W[2], color='black', linestyle='-')
#plt.axis([-1, 1, -1, 1])
plt.xlabel("X1");
plt.ylabel("X2");
plt.show()

