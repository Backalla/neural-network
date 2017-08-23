import tensorflow as tf
from matplotlib import pyplot as plt

num_h1_nodes = 3
num_h2_nodes = 2
num_classes = 2

X = tf.placeholder(tf.float32,[None,2])
Y = tf.placeholder(tf.float32)

def plot_scatter(epoch,slope,intercept):
  plt.figure(epoch)
  data = get_data()
  x=[]
  y=[]
  colors = {1:'r',2:"b"}
  for point in data["x"]:
    x.append(point[0])
    y.append(point[1])
  c = [colors[a] for a in data["y"]]
  plt.scatter(x,y,c=c)

  m = float(slope[1])/float(slope[0])
  print "y = {}x+{}".format(m,float(intercept))
  y_values = [m * i + float(intercept) for i in x]
  # print y_values
  plt.plot(x, y_values, 'b')
  plt.savefig("epoch{}.png".format(epoch))
def get_data():
  data = []
  x_list = []
  y_list = []
  with open("data.txt") as f:
    lines = f.read().strip().split("\n")
    for line in lines:
      line = [float(x) for x in line.split(",")]
      x,y = [line[0],line[1]],int(line[2])
      x_list.append(x)
      y_list.append(y)
  return {"x":x_list,"y":y_list}

def define_network(data):
  h1 = {"weights":tf.Variable(tf.random_normal([2,num_h1_nodes])),"biases":tf.Variable(tf.random_normal([num_h1_nodes]))}
  h2 = {"weights":tf.Variable(tf.random_normal([num_h1_nodes,num_h2_nodes])),"biases":tf.Variable(tf.random_normal([num_h2_nodes]))}
  op = {"weights":tf.Variable(tf.random_normal([num_h2_nodes,1])),"biases":tf.Variable(tf.random_normal([1]))}

  l1 = tf.add(tf.matmul(data, h1["weights"]), h1["biases"])
  l1 = tf.nn.sigmoid(l1)

  l2 = tf.add(tf.matmul(l1, h2["weights"]), h2["biases"])
  l2 = tf.nn.sigmoid(l2)

  output = tf.add(tf.matmul(l2, op["weights"]), op["biases"])

  return output

def define_perceptron(data):
  op = {"weights":tf.Variable(tf.zeros([2,1])),"biases":tf.Variable(tf.zeros([1]))}

  output = tf.add(tf.matmul(data, op["weights"]), op["biases"])
  output = tf.nn.sigmoid(output)

  return output,op

def train_network():
  data = get_data()
  x = data["x"]
  y = data["y"]
  # Y_pred = define_network(x)
  Y_pred,perceptron = define_perceptron(x)
  cost = tf.reduce_mean(tf.squared_difference(Y_pred,Y))

  optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

  num_epochs = 100
  perceptron_w, perceptron_b = None,None
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
      _,c,perceptron_w,perceptron_b = sess.run([optimiser,cost,perceptron["weights"],perceptron["biases"]],feed_dict={X:x,Y:y})
      print "Loss: {}, W: {}, b: {}".format(c,perceptron_w,perceptron_b)
      try:
        if epoch%10==0:
          plot_scatter(epoch,perceptron_w,perceptron_b)
      except:
        pass



train_network()
