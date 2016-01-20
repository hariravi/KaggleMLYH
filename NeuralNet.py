
"""
Multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep (This is now implemented).

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from LogisticRegression import LogisticRegression, load_data
import math
import ast
import DataLoader
import copy
import pickle
import MLutil

# Hidden layer of a neural network
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        # Note in future, can also use existing W/b values, i.e. in the case of recursive pretraining
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            # Recommended Sigmoid Initialization
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # Perform activation on input * W
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class (Neural network)

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, num_layers):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type num_layers: int
        :param num_layers: number of hidden layers, the number of layers we
        stack ... stacking > 1 makes the architecture deep

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        temp_str = "self.hiddenLayer"
        prev_str = temp_str[:]

        # Sanity check, should not have a 0 layer neural net
        assert(num_layers >= 1)
        self.params = []
        self.L1 = 0.
        self.L2_sqr = 0.
        # First layer
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.params += self.hiddenLayer.params
        self.L1 += abs(self.hiddenLayer.W).sum()
        self.L2_sqr += (self.hiddenLayer.W ** 2).sum()

        # Add additional hidden_layers
        for i in range(num_layers-1):
            new_str = temp_str + str(i)
            exec (new_str + '= HiddenLayer(rng=rng, input={0}.output, n_in=n_hidden, n_out=n_hidden,activation=T.tanh)'.format(prev_str))
            param_str = new_str + '.params'
            self.params += eval(param_str)
            self.L1 += abs(eval(new_str + '.W')).sum()
            self.L2_sqr += (eval(new_str + '.W') ** 2).sum()
            prev_str = new_str[:]


        # The logistic regression layer gets as input the hidden units
        # of the hidden 
        self.logRegressionLayer = LogisticRegression(
            input=eval("{0}.output".format(prev_str)),
            n_in=n_hidden,
            n_out=n_out
        )

        # Update these class variables
        self.params += self.logRegressionLayer.params
        self.L1 += abs(self.logRegressionLayer.W).sum()
        self.L2_sqr += (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # Digit predictions
        self.y_pred = self.logRegressionLayer.y_pred

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        # self.params = self.hiddenLayer.params + self.hiddenLayer2.params 
        # + self.hiddenLayer3.params + self.logRegressionLayer.params
        # keep track of model input
        self.input = input

# Initially just do L2 regularization
# L1 penalty is none
def validation_analysis(learning_rate=.001, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             dataset='mnist_train.csv', batch_size=20, n_hidden=300, num_layers = 2, mlp_in = 784, mlp_out = 10):
    """
    Main loop for the mlp

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to appropriate dataset

    :type batch_size: int
    :param batch_size: number of entries per mini-batch

    :type n_hidden: int
    :param n_hidden: number of entries per hidden layer

    :type num_layers: int
    :param num_layers: number of hidden layers

    :type mlp_in: int 
    :param mlp_in: dimension of data loaded from dataset

    :type mlp_out: int
    :param mlp_out: number of classes in dataset


   """
    # Load in the data   
    datasets = DataLoader.load_kaggle_mnist(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = math.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = math.floor(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = math.floor(test_set_x.get_value(borrow=True).shape[0] / batch_size)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # n data entries, m features per entry ==> n by m matrix of data
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    # Seed the random number generator
    rng = numpy.random.RandomState(1234)

    # construct the MLP class (the neural net)
    classifier = MLP(
        rng=rng,
        input=x,
        n_in = mlp_in,
        n_hidden=n_hidden,
        n_out=mlp_out,
        num_layers=num_layers
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #old_classifier = copy.deepcopy(classifier)
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]

                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    pickle.dump(classifier, open("best_model.p", "wb"))

                # Otherwise, reduce the learning rate (need to fix this/do this in a better way)
                '''
                else:
                    classifier = old_classifier
                    learning_rate = learning_rate / 2
                    print("Learning rate halved ... the new learning rate is {0}".format(learning_rate))
                '''
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] +
        ' ran for %.2fm' % ((end_time - start_time) / 60.))

    #pickle.dump(classifier, open("best_model.p", "wb"))

# Neural network predictions, use MLutil functions
def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    # load the saved model
    classifier = pickle.load(open("best_model.p", "rb"))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset = 'mnist_train.csv'
    datasets = DataLoader.load_kaggle_mnist(dataset)

    test_set_x, test_set_y = datasets[2]
    print(type(test_set_x))
    print(type(test_set_y))
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()


    predicted_values = predict_model(test_set_x[20:30])
    print("Sample Neural Prediction")
    print ("Predicted values for the first 20 examples in test set:")
    print(predicted_values)
    print ("The actual values are")
    print(test_set_y[20:30])

def predict_main(classifier_pickle):
    print("This functions is being called")
    datasets = DataLoader.load_kaggle_mnist("mnist_train.csv")
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()

    predictor = MLutil.Predictor(classifier_pickle, 'DNN')
    predicted_values = predictor.make_prediction(test_set_x)

    predAnalysis = MLutil.PredictionAccuracies(predicted_values, test_set_y)
    print(predAnalysis.get_misclass_rate())
    print(predAnalysis.get_indicies_misclassifications())

    pickle.dump(predAnalysis.get_indicies_misclassifications(), open("neural_indicies.p", "wb"))
    return predAnalysis.get_indicies_misclassifications()

if __name__ == '__main__':
    #validation_analysis()
    predict_main("best_model.p")
