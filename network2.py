#This code is derived from the book "Neural Networks and Deep Learning" by Michael Nielsen
#You can access the code on https://github.com/mnielsen/neural-networks-and-deep-learning.git

class Network(object):

  def __init__(self, sizes, cost=CrossEntropyCost):
    #sizes=list of sizes for the respective layers, e.g. [1,2,3] for a network with 3 layers which have 1, 2 and 3 neurons respectively
    #cost=choice of the cost function to use
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.default_weight_initializer()
    #new approach to weight initialization
    self.cost=cost
    
  def default_weight_initializer(self):
    #weight initialization as Gaussian random variables with mean 0 and standard deviation 1/sqrt(n of input neurons to the neuron)
    #biases initialization as Gaussian random variables with mean 0 and standard deviation 1
    self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
    #initialize biases beginning from second layer
    self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
