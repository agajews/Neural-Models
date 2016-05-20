from lasagne import init
from lasagne.layers import InputLayer, LSTMLayer
from lasagne.layers import DropoutLayer, SliceLayer, DenseLayer
from lasagne.nonlinearities import tanh, softmax

import theano.tensor as T

import numpy as np

from neural_models.data.phys_weather.station_data import gen_station_data

from neural_models.lib import split_val, iterate_minibatches

from neural_models.models import Model


class WeatherModel(Model):

    def build_test_acc(self, test_output, target_values):

        y_hat = T.argmax(test_output, axis=1)
        y = T.argmax(target_values, axis=1)

        dist = abs(y_hat - y)
        unique_vals, unique_counts = T.extra_ops.Unique(
                dist, return_counts=True)

        return unique_vals, unique_counts

    def update_acc_dist(self, acc_dist, unique_vals, unique_counts):

        for val, count in zip(unique_vals, unique_counts):
            acc_dist[val] += count

        return acc_dist

    def compute_val_metrics(self, test_fn, val_Xs, val_y):

        num_examples, output_spread = val_y.shape

        val_loss = 0
        acc_dist = np.zeros(num_examples, output_spread)
        val_batches = 0

        for batch in iterate_minibatches(*val_Xs, val_y):
            [loss, acc] = test_fn(*batch)
            acc_dist = self.update_acc_dist(acc_dist, *acc)
            val_loss += loss
            val_batches += 1

        val_loss /= val_batches
        acc_dist /= val_batches

        return val_loss, acc_dist

    def display_val_metrics(self, val_metrics):

        val_loss, acc_dist = val_metrics

        print('Val Loss: ' + str(val_loss))
        print('Val Acc Dist: ' + str(acc_dist))


class StationModel(WeatherModel):

    def get_default_param_filename(self):

        return 'params/phys_weather/station_model.p'

    def load_hyperparams(self, hyperparams):

        self.num_hidden = int(hyperparams['num_hidden'])
        self.num_epochs = int(hyperparams['num_epochs'])
        self.batch_size = int(hyperparams['batch_size'])
        self.dropout_val = float(hyperparams['dropout_val'])
        self.learning_rate = float(hyperparams['learning_rate'])
        self.grad_clip = float(hyperparams['grad_clip'])
        self.l2_reg_weight = float(hyperparams['l2_reg_weight'])

    def load_default_hyperparams(self):

        self.num_hidden = 250
        self.num_epochs = 31
        self.batch_size = 365
        self.dropout_val = 0.0
        self.learning_rate = 0.055
        self.grad_clip = 927
        self.l2_reg_weight = 0.0007

    def create_lstm_stack(self, net):

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        return net

    def create_model(self, input_spread, output_spread):

        net = InputLayer(shape=(None, None, input_spread))

        for _ in range(2):
            net = self.create_lstm_stack(net)

        net = SliceLayer(net, -1, 1)

        net = DenseLayer(
                net,
                num_units=output_spread,
                W=init.Normal(),
                nonlinearity=softmax)

        return net

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        temp_spread = len(train_Xs[0][0, 0, :])

        supp_model_params = {}
        supp_model_params['input_spread'] = temp_spread
        supp_model_params['output_spread'] = temp_spread

        return supp_model_params

    def get_data(self):

        [
                min_train_X, min_train_y,
                min_test_X, min_test_y,
                _, _, _, _
        ] = gen_station_data()
        train_X, val_X, train_y, val_y = split_val(min_train_X, min_train_y)
        train_Xs, val_Xs = [train_X], [val_X]

        return train_Xs, val_Xs, train_y, val_y
