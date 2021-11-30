import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

x, y = spiral_data(samples=100, classes=3)

class layer_dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights=0.10*np.random.randn(n_inputs,n_neurons)
		self.biases= np.zeros((1,n_neurons))
	def forward(self, inputs):
		self.output= np.dot(inputs,self.weights) 

class activation_relu:
	def forward(self,inputs):
		self.output=np.maximum(0,inputs)

class softmax_layer:
	def forward(self,inputs):
		exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
		self.output=exp_values/np.sum(exp_values, axis=1, keepdims= True)

class Loss:
	def calculate(self,output,y):
		sample_loss=self.forward(output,y)
		data_loss= np.mean(sample_loss)
		return data_loss

class categorical_loss(Loss):
	def forward(self,y_pred,y_true):
		samples=len(y_pred)
		y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
		if len(y_true.shape) == 1:
			correct_values=y_pred_clipped[range(samples),y_true]
		elif len(y_true.shape) == 2:
			correct_values=np.sum(y_pred_clipped*y_true,axis=1)
		log_values=-np.log(correct_values)
		return log_values




layer1=layer_dense(2,3)
activation1=activation_relu()

layer2=layer_dense(3,3)
activation2=softmax_layer()


layer1.forward(x)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)

loss_function= categorical_loss()
loss=loss_function.calculate(activation2.output,y)
print("loss/n",loss)



