from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
import theano, sys
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
import numpy

BATCH_SIZE = 100  # default batch size
L2_LAMBDA = 1.    # default L2 regularization parameter
INIT_LR = 0.01    # initial learning rate, try making it larger


def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.


def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
                dtype=theano.config.floatX)
        return x * mask
    return x


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


class Linear(object):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W_values *= 4  # This works for sigmoid activated networks!
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b

    def __repr__(self):
        return "Linear"


class SigmoidLayer(Linear):
    """ Sigmoid activation layer (sigmoid(W.X + b)) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        super(SigmoidLayer, self).__init__(rng, input, n_in, n_out, W, b)
        self.pre_activation = self.output
        self.output = T.nnet.sigmoid(self.pre_activation)


class ReLU(Linear):
    """ Rectified Linear Unit activation layer (max(0, W.X + b)) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        super(ReLU, self).__init__(rng, input, n_in, n_out, W, b)
        self.pre_activation = self.output
        self.output = relu_f(self.pre_activation)


class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, x, y=None, batch_size=BATCH_SIZE, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)

    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                if self.y != None:
                    yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],
                           self.y[i*self.batch_size:(i+1)*self.batch_size])
                else:
                    yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                if self.y != None:
                    yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
                           self.y[i*self.batch_size:(i+1)*self.batch_size])
                else:
                    yield (self.x[i*self.batch_size:(i+1)*self.batch_size])


class LogisticRegression:
    """ _Multi-class_ Logistic Regression """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W != None:
            self.W = W
        else:
            self.W = build_shared_zeros((n_in, n_out), 'W')
        if b != None:
            self.b = b
        else:
            self.b = build_shared_zeros((n_out,), 'b')
        self.input = input
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.y_pred
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood(y)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("!!! 'y' should have the same shape as 'self.y_pred'",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))


class NeuralNet(object):
    """ Neural network (not regularized, without dropout) """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=40*3,
                 layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024, 1024],
                 n_outs=62*3,
                 rho=0.95, eps=1.E-6,
                 momentum=0.9, step_adapt_alpha=1.E-4,
                 debugprint=False):
        """
        Basic Neural Net class
        """
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self._rho = rho  # ``momentum'' for adadelta (and discount/decay for RMSprop)
        self._eps = eps  # epsilon for adadelta (and for RMSprop)
        self._momentum = momentum  # for RMSProp
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        self._avggrads = []  # for RMSprop in the Alex Graves' variant
        self._stepadapts = []  # for RMSprop with step adaptations
        self._stepadapt_alpha = step_adapt_alpha

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
            this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])
            self._avggrads.extend([build_shared_zeros(t.shape.eval(),
                'avggrad') for t in this_layer.params])
            self._stepadapts.extend([shared(value=numpy.ones(t.shape.eval(),
                dtype=theano.config.floatX),
                name='stepadapt', borrow=True) for t in this_layer.params])
            self.layers.append(this_layer)
            layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
        self.cost = self.layers[-1].training_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)

        self.errors = self.layers[-1].errors(self.y)
        self.y_pred = self.layers[-1].y_pred
        self.p_y_given_x = self.layers[-1].p_y_given_x

    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str)))


    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate
        gparams = T.grad(self.mean_cost, self.params)  # all the gradients
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y),
                                           theano.Param(learning_rate)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate
        gparams = T.grad(self.mean_cost, self.params)  # all the gradients
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        gparams = T.grad(self.mean_cost, self.params)
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            updates[accudelta] = (self._rho * accudelta
                                  + (1 - self._rho) * dx * dx)
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y})

        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref

    def predict_(self, given_set):
        batch_x = T.fmatrix('batch_x')
        pred = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.y_pred,
                                givens={self.x: batch_x})
        def predictf():
            return [pred(batch_x) for batch_x in given_set]

        return predictf

    def predict_proba_(self, given_set):
        batch_x = T.fmatrix('batch_x')
        pred_prob = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.p_y_given_x,
                                givens={self.x: batch_x})
        def predict_probf():
            return [pred_prob(batch_x) for batch_x in given_set]

        return predict_probf


class RegularizedNet(NeuralNet):
    """ Neural net with L1 and L2 regularization """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=100,
                 layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024],
                 n_outs=2,
                 rho=0.95, eps=1.E-6,
                 L1_reg=0.1,
                 L2_reg=0.1,
                 debugprint=False):
        """
        A deep neural net with possible L1 and/or L2 regularization.
        """
        super(RegularizedNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint)

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        L1 = shared(0.)
        for param in self.params:
            L1 += T.sum(abs(param))
        if L1_reg > 0.:
            self.cost = self.cost + L1_reg * L1
        L2 = shared(0.)
        for param in self.params:
            L2 += T.sum(param ** 2)
        if L2_reg > 0.:
            self.cost = self.cost + L2_reg * L2


class DropoutNet(NeuralNet):
    """ Neural net with dropout (see Hinton's et al. paper) """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=40*3,
                 layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[4000, 4000, 4000, 4000],
                 dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
                 n_outs=62 * 3,
                 rho=0.98, eps=1.E-6,
                 debugprint=False):
        """
        A dropout-regularized neural net.
        """
        super(DropoutNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint)

        self.dropout_rates = dropout_rates
        dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []

        for layer, layer_type, n_in, n_out, dr in zip(self.layers,
                layers_types, self.layers_ins, self.layers_outs,
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything
                                           # from the last layer !!!
            if dr:
                this_layer = layer_type(rng=numpy_rng,
                        input=dropout_layer_input, n_in=n_in, n_out=n_out,
                        W=layer.W * 1. / (1. - dr),
                        b=layer.b * 1. / (1. - dr))
                # N.B. dropout with dr==1 does not dropanything!!
                this_layer.output = dropout(numpy_rng, this_layer.output, dr)
            else:
                this_layer = layer_type(rng=numpy_rng,
                        input=dropout_layer_input, n_in=n_in, n_out=n_out,
                        W=layer.W, b=layer.b)

            assert hasattr(this_layer, 'output')
            self.dropout_layers.append(this_layer)
            dropout_layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        # these are the dropout costs
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        self.cost = self.dropout_layers[-1].training_cost(self.y)

        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        return super(DropoutNet, self).__repr__() + "\n"\
                + "dropout rates: " + str(self.dropout_rates)


def add_fit_score_predict_proba(class_to_chg):
    """ Mutates a class to add the fit() and score() functions to a NeuralNet.
    """
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=20, early_stopping=True, split_ratio=0.1, # TODO 100+ epochs
            method='adadelta', verbose=False, plot=False):
        """
        TODO
        """
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        if method == 'sgd':
            train_fn = self.get_SGD_trainer()
        elif method == 'adagrad':
            train_fn = self.get_adagrad_trainer()
        elif method == 'adadelta':
            train_fn = self.get_adadelta_trainer()
        elif method == 'rmsprop':
            train_fn = self.get_rmsprop_trainer(with_step_adapt=True,
                    nesterov=False)
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf
        epoch = 0
        # TODO early stopping (not just cross val, also stop training)
        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []

        init_lr = INIT_LR
        if method == 'rmsprop':
            init_lr = 1.E-6  # TODO REMOVE HACK
        n_seen = 0
        while epoch < max_epochs:
            #lr = init_lr / (1 + init_lr * L2_LAMBDA * math.log(1+n_seen))
            #lr = init_lr / math.sqrt(1 + init_lr * L2_LAMBDA * n_seen/BATCH_SIZE) # try these
            lr = init_lr
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for x, y in train_set_iterator:
                if method == 'sgd' or method == 'adagrad' or method == 'rmsprop':
                    #avg_cost = train_fn(x, y, lr=1.E-2)
                    avg_cost = train_fn(x, y, lr=lr)
                elif method == 'adadelta':
                    avg_cost = train_fn(x, y)
                elif method == 'rmsprop':
                    avg_cost = train_fn(x, y, lr=lr)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())
                print('  epoch %i took %f seconds' %
                      (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                      (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                      (epoch, mean_train_errors))
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)
            dev_errors = numpy.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                          (epoch, dev_errors))
            epoch += 1
            n_seen += x_train.shape[0]
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param

    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return numpy.mean(scoref())
     
    def predict(self, x):
        iterator = DatasetMiniBatchIterator(x)
        predictf = self.predict_(iterator)
        return numpy.concatenate(predictf(), axis=0)

    def predict_proba(self, x):
        iterator = DatasetMiniBatchIterator(x)
        predictpbf = self.predict_proba_(iterator)
        return numpy.concatenate(predictpbf(), axis=0)

    class_to_chg.fit = MethodType(fit, None, class_to_chg)
    class_to_chg.score = MethodType(score, None, class_to_chg)
    class_to_chg.predict = MethodType(predict, None, class_to_chg)
    class_to_chg.predict_proba = MethodType(predict_proba, None, class_to_chg)


DEEP = True
ONEHOTENCODING = True

def model(X_train, y_train, X_test):
    add_fit_score_predict_proba(DropoutNet)
    add_fit_score_predict_proba(RegularizedNet)
    DEEP = True
    if DEEP:
        numpy_rng = numpy.random.RandomState(42)
        #dnn = DropoutNet(numpy_rng=numpy_rng, n_ins=X_train.shape[1],
        #    layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
        #    layers_sizes=[200, 200, 200],
        #    dropout_rates=[0.2, 0.5, 0.5, 0.5],
            #layers_types=[LogisticRegression],
            #layers_sizes=[],
            #dropout_rates=[0.0],
        #    n_outs=2,
        #    debugprint=0)
        dnn = RegularizedNet(numpy_rng=numpy_rng, n_ins=X_train.shape[1],
            layers_types=[LogisticRegression],
            layers_sizes=[],
            n_outs=2,
            debugprint=0)
        #clf = Pipeline([('imputer', Imputer()),
        #    ('dnn', dnn)])
        #clf.fit(X_train, y_train)
        dnn.fit(X_train, y_train, max_epochs=50)
        y_pred = dnn.predict(X_test)
        y_score = dnn.predict_proba(X_test)
    else:
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        y_score = gnb.predict_proba(X_test)
    return y_pred, y_score

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    df = pd.read_csv('train.csv')
    y_train = np.array(df['TARGET'].values, dtype='int32')
    X_train = np.array(df.drop('TARGET', axis=1).values, dtype='float32')
    #X_train = df.drop('TARGET', axis=1).values
    #X_train = np.nan_to_num(X_train)
    X_train = Imputer().fit_transform(X_train)

    # WEIGHTING
    X_train = np.concatenate([X_train, X_train[y_train==1]], axis=0)
    y_train = np.concatenate([y_train, y_train[y_train==1]], axis=0)
    from sklearn import utils
    print X_train.shape
    print y_train.shape
    X_train, y_train = utils.shuffle(X_train, y_train)

    # ONE HOT ENCODING
    if ONEHOTENCODING:
        from sklearn.preprocessing import OneHotEncoder
        categ_inds = filter(lambda (_, k): k.isupper(), enumerate(df.keys()))
        ohe = OneHotEncoder(categorical_features=zip(*categ_inds)[0], sparse=False)
        X_train = np.asarray(ohe.fit_transform(X_train), dtype='float32')
        print X_train.shape

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
            test_size=0.2)
    y_pred, y_score = model(X_train, y_train, X_test)
    print y_pred.shape
    print y_score.shape
    print y_test.shape
    ys = [y_score[i, j] for i, j in enumerate(y_test)]

    from sklearn.metrics import roc_auc_score, accuracy_score
    print "AUC:", roc_auc_score(y_test, ys)
    print "acc:", accuracy_score(y_test, y_pred)

    #rf = RandomForestClassifier(n_estimators=200)
    #clf = Pipeline([('imputer', Imputer()), ('rf', rf)])
    #X_train = clf.fit_transform(X_train, y_train)




