import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

num_input_nodes = 3
num_h1_nodes = 4
num_op_nodes = 2
learning_rate = 0.01

def make_network():


  h1_layer = {"weights": np.random.rand(num_input_nodes, num_h1_nodes),
              "biases": np.random.rand(num_h1_nodes),
              "deltas":np.zeros(num_h1_nodes),
              "activation":np.ones(num_h1_nodes),
              "changes":np.zeros((num_input_nodes,num_h1_nodes))}
  op_layer = {"weights": np.random.rand(num_h1_nodes, num_op_nodes),
              "biases": np.random.rand(num_op_nodes),
              "deltas":np.zeros(num_op_nodes),
              "activation":np.ones(num_op_nodes),
              "changes": np.zeros((num_h1_nodes,num_op_nodes))}
  # op_layer = {"weights": np.random.rand(num_h2_nodes, num_op_nodes), "biases": np.random.rand(num_op_nodes),"influence":np.zeros(num_op_nodes),"output":np.zeros(num_op_nodes)}

  return [h1_layer, op_layer]

network = make_network()


def summation(layer_input,layer):
  return np.matmul(layer_input,layer["weights"])+layer["biases"]


def sigmoid(neuron_sum):
  return 1.0/(1.0+np.exp(neuron_sum))


def derivative(output):
  return output*(1.0-output)

def forward_propagate(x):
  h1 = sigmoid(summation(x,network[0]))
  network[0]["activation"] = h1
  # print h1
  op = sigmoid(summation(h1,network[1]))
  network[1]["activation"] = op

  # print network
  # print derivative(h1)
  # print derivative(op)
  # print op
  # print op
  return op

def back_propagate(x,y):
  op_error = (network[1]["activation"]-y)
  network[1]["deltas"] = op_error*derivative(network[1]["activation"])

  # print network[1]["deltas"]

  # for j in range(num_h1_nodes):
  #   h1_error = 0.0
  #   for k in range(num_op_nodes):
  #     h1_error += network[1]["deltas"][k]*network[1]["weights"][j][k]
  #   network[0]["deltas"][j] = derivative(network[0]["activation"][j])*h1_error

  h1_error = network[1]["deltas"].dot(network[1]["weights"].T)

  network[0]["deltas"] = h1_error*derivative(network[0]["activation"])

  # print network[0]["deltas"]
  # print network[0]["activation"].T,(network[1]["deltas"])
  # network[1]["weights"] += network[0]["activation"].T.dot(network[1]["deltas"])
  # network[0]["weights"] += np.array(x).T.dot(network[0]["deltas"])


  #
  for j in range(num_h1_nodes):
    for k in range(num_op_nodes):
      change = network[1]["deltas"][k]*network[0]["activation"][j]
      network[1]["weights"][j][k] += learning_rate*change



  for i in range(num_input_nodes):
    for j in range(num_h1_nodes):
      change = network[0]["deltas"][j]*x[i]
      # print "Change",change
      network[0]["weights"][i][j] -= learning_rate*change





  # return network

def one_hot(y):
  oh = [[1,0],[0,1]]
  return oh[y]


def test_network(test_data):
  error = 0.0
  for index,row in test_data.iterrows():
    x = row[["x","y","z"]].values
    y = one_hot(int(row["class"]))
    y_pred = forward_propagate(x)
    error += np.sum(0.5*((np.array(y)-y_pred)**2))
  return error


def display_weights():
  print "h1 weights"
  print network[0]["weights"]
  print "h1 biases"
  print network[0]["biases"]
  print
  print "output weights"
  print network[1]["weights"]
  print "output biases"
  print network[1]["biases"]


def train_network():
  x = [3,4,5]
  y = [0,1]

  num_epochs = 100000
  for epoch in range(num_epochs):
    if epoch%(num_epochs/10)==0:
      print "Epoch:",epoch
      # display_weights()
    train_output = forward_propagate(x)
    back_propagate(x, y)
    if epoch%(num_epochs/10)==0:
      # display_weights()
      print "\nOutput:",train_output
      print "-" * 30
      print





  # train_df = pd.read_csv("3d_points.csv")
  #
  # num_epochs = 1000
  # for epoch in range(num_epochs):
  #   train_data, test_data = train_test_split(train_df, test_size=0.2)
  #   for index,row in train_data.iterrows():
  #     x = row[["x","y","z"]].values
  #     y = one_hot(int(row["class"]))
  #     # print x
  #     # print y
  #     # exit()
  #     forward_propagate(x)
  #     back_propagate(x,y)
  #
  #   if epoch%(num_epochs/10)==0:
  #     epoch_error = test_network(test_data)
  #     print epoch_error

train_network()




