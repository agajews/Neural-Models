from lasagne.updates import adagrad
from lasagne.layers import InputLayer
from lasagne.layers import get_output, get_all_params, get_all_param_values, \
    set_all_param_values
from lasagne.objectives import squared_error, aggregate
from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers.helper import count_params

import theano
import theano.tensor as T

import numpy as np

import pickle

from neural_models.lib import iterate_minibatches


class Model(object):

    def __init__(self, hyperparams=None, param_filename=None):

        self.set_hyperparams(hyperparams)

        self.set_param_filename(param_filename)

        np.random.seed(42)

    def set_param_filename(self, param_filename):

        if param_filename is not None:
            self.param_filename = param_filename

        else:
            self.param_filename = self.get_default_param_filename()

    def get_default_param_filename(self):

        return 'params/model.p'

    def set_hyperparams(self, hyperparams):

        if hyperparams is not None:
            self.load_hyperparams(hyperparams)

        else:
            self.load_default_hyperparams()

    def load_hyperparams(self, hyperparams):

        raise NotImplementedError()

    def load_default_hyperparams(self):

        raise NotImplementedError()

    def create_model(self, input_spread, output_spread):

        raise NotImplementedError()

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        return None

    def build_train_loss(self, train_output, target_values):

        l2_penalty = regularize_layer_params(self.layers, l2) * self.l2_reg_weight
        loss = T.nnet.categorical_crossentropy(
                train_output, target_values).mean()
        loss += l2_penalty

        return loss

    def build_test_loss(self, test_output, target_values):

        test_loss = T.nnet.categorical_crossentropy(
                test_output, target_values).mean()

        return test_loss

    def build_test_acc(self, test_output, target_values):

        test_acc = T.mean(T.eq(
                T.argmax(test_output, axis=1),
                T.argmax(target_values, axis=1)),
            dtype=theano.config.floatX)

        return test_acc

    def update_params(self, loss, all_params):

        return adagrad(loss, all_params, self.learning_rate)

    def build_train_updates(self, loss):

        all_params = get_all_params(self.net, trainable=True)
        updates = self.update_params(loss, all_params)

        return updates

    def get_input_vars(self, input_layers):

        input_vars = []

        for layer in input_layers:
            if isinstance(layer, InputLayer):
                input_vars.append(layer.input_var)

        return input_vars

    def get_target_input_var(self):

        return T.imatrix('target_output')

    def build_train_fn(self):

        target_values = self.get_target_input_var()

        train_output = get_output(self.net)
        loss = self.build_train_loss(train_output, target_values)
        updates = self.build_train_updates(loss)

        train_fn = theano.function(
                self.input_vars + [target_values],
                loss, updates=updates, allow_input_downcast=True)

        return train_fn

    def build_test_fn(self):

        target_values = self.get_target_input_var()

        test_output = get_output(self.net, deterministic=True)
        test_loss = self.build_test_loss(test_output, target_values)
        test_acc = self.build_test_acc(test_output, target_values)

        test_fn = theano.function(
                self.input_vars + [target_values],
                [test_loss, test_acc], allow_input_downcast=True)

        return test_fn

    def compute_train_metrics(self, train_Xs, train_y):

        train_loss = 0
        train_batches = 0

        for batch in iterate_minibatches(
                *train_Xs, train_y, batch_size=self.batch_size):
            train_loss += self.train_fn(*batch)
            train_batches += 1

        train_loss /= train_batches

        return train_loss

    def compute_val_metrics(self, val_Xs, val_y):

        val_loss = 0
        val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(
                *val_Xs, val_y, batch_size=self.batch_size):
            [loss, acc] = self.test_fn(*batch)
            val_loss += loss
            val_acc += acc
            val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        return val_loss, val_acc

    def display_train_metrics(self, train_metrics, epoch=None):

        disp = ''

        if epoch is not None:
            disp += 'Epoch: %s | ' % epoch

        disp += 'Train loss: %s' % str(train_metrics)

        print(disp)

    def display_val_metrics(self, val_metrics):

        val_loss, val_acc = val_metrics

        print('Val Loss: ' + str(val_loss) + ' | Val Acc: ' + str(val_acc))

    def save_params(self):

        params = get_all_param_values(self.layers)
        pickle.dump(params, open(self.param_filename, 'wb'))

    def load_params(self):

        params = pickle.load(open(self.param_filename, 'rb'))
        set_all_param_values(self.layers, params)

    def perform_epoch(self,
            train_Xs, train_y, val_Xs, val_y,
            val=True,
            epoch=None):

        train_metrics = self.compute_train_metrics(train_Xs, train_y)

        if self.verbose:
            self.display_train_metrics(train_metrics, epoch=epoch)

        if self.save:
            self.save_params()

        if val:
            val_metrics = self.compute_val_metrics(val_Xs, val_y)

            if self.verbose:
                self.display_val_metrics(val_metrics)

        return train_metrics, val_metrics

    def create_model_with_supp(self, supp_model_params):

        if supp_model_params is not None:
            input_layers = self.create_model(**supp_model_params)
        else:
            input_layers = self.create_model()

        return input_layers

    def display_info(self,
            train_Xs, val_Xs,
            train_y, val_y):

        print('Train Input shapes:')
        self.print_shapes(train_Xs)

        print('Val Input shapes:')
        self.print_shapes(val_Xs)

        num_params = count_params(self.layers)
        print('Num params: %d' % num_params)

    def print_shapes(self, arrays):

        for array in arrays:
            print(array.shape)

    def compile_net(self, train_Xs, val_Xs, train_y, val_y):

        if self.verbose:
            print('Compiling net')

        self.layers = []

        supp_model_params = self.get_supp_model_params(
                train_Xs, train_y, val_Xs, val_y)

        input_layers = self.create_model_with_supp(supp_model_params)

        self.input_vars = self.get_input_vars(input_layers)

    def compile_net_notrain(self):

        self.compile_net(None, None, None, None)

    def train_model(self, train_Xs, val_Xs, train_y, val_y,
            val=True):

        if self.verbose:
            self.display_info(train_Xs, val_Xs, train_y, val_y)

        if self.verbose:
            print('Beginning training')

        for epoch in range(self.num_epochs):
            train_metrics, val_metrics = self.perform_epoch(
                    train_Xs, train_y, val_Xs, val_y,
                    val, epoch)

        if self.save:
            self.save_params()

        if val:
            return val_metrics

    def pretrain_compile(self, *data):

        self.compile_net(*data)

        self.train_fn = self.build_train_fn()

        self.test_fn = self.build_test_fn()

    def get_data(self):

        raise NotImplementedError()

    def load_train_with_data_config(self):

        self.verbose = True
        self.save = True
        self.epoch_save = False

    def train_with_data(self):

        self.load_train_with_data_config()

        data = self.get_data()

        self.pretrain_compile(*data)

        self.train_model(*data, val=True)

    def get_megabatches(self):

        raise NotImplementedError()

    def load_train_with_megabatches_config(self):

        self.verbose = False
        self.save = True
        self.epoch_save = False

    def train_with_megabatches(self):

        self.load_train_with_megabatches_config()

        data = self.get_data()

        self.pretrain_compile(*data)

        for megabatch in self.get_megabatches():

            self.train_model(*megabatch, val=True)

    def load_test_hyperparams_config(self):

        self.verbose = False
        self.save = False
        self.epoch_save = False

    def test_hyperparams(self, **hyperparams):

        self.load_test_hyperparams_config()

        self.set_hyperparams(hyperparams)

        data = self.get_data()

        self.pretrain_compile(*data)

        val_loss, val_acc = self.train_model(
                *data, val=True)

        return val_acc


class RegressionModel(Model):

    def build_train_loss(self, train_output, target_values):

        l2_penalty = regularize_layer_params(self.layers, l2) * self.l2_reg_weight
        loss = self.msq_err(train_output, target_values)
        loss += l2_penalty

        return loss

    def msq_err(self, train_output, target_values):

        loss = squared_error(train_output, target_values)
        loss = aggregate(loss, mode='mean')

        return loss

    def build_test_loss(self, test_output, target_values):

        test_loss = self.msq_err(test_output, target_values)

        return test_loss

    def build_test_acc(self, test_output, target_values):

        test_acc = T.mean(abs(test_output - target_values))

        return test_acc

    def get_target_input_var(self):

        return T.ivector('target output')
