from lasagne import init
from lasagne.layers import InputLayer, LSTMLayer, DropoutLayer
from lasagne.layers import SliceLayer, DenseLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import CustomRecurrentLayer, ConcatLayer
from lasagne.nonlinearities import tanh, softmax, rectify

from neural_models.data.phys_weather.map_data import gen_map_data

from neural_models.lib import split_val

from neural_models.models import Model

from neural_models.hyper_optim import BayesHyperOptim, GridHyperOptim


class MapModel(Model):

    def get_default_param_filename(self):

        return 'params/phys_weather/map_model.p'

    def load_hyperparams(self, hyperparams):

        self.num_hidden = int(hyperparams['num_hidden'])
        self.num_epochs = int(hyperparams['num_epochs'])
        self.batch_size = int(hyperparams['batch_size'])
        self.dropout_val = float(hyperparams['dropout_val'])
        self.learning_rate = float(hyperparams['learning_rate'])
        self.grad_clip = float(hyperparams['grad_clip'])
        self.l2_reg_weight = float(hyperparams['l2_reg_weight'])
        self.num_filters = int(hyperparams['num_filters'])
        self.filter_size = int(hyperparams['filter_size'])
        self.pool_size = int(hyperparams['pool_size'])
        self.embedding = int(hyperparams['embedding'])

        self.width = int(hyperparams['width'])
        self.height = int(hyperparams['height'])
        self.timesteps = int(hyperparams['timesteps'])
        self.num_channels = 1

    def load_default_hyperparams(self):

        self.dropout_val = 0.0
        self.num_hidden = 512
        self.grad_clip = 100
        self.num_filters = 32
        self.filter_size = 3
        self.pool_size = 3
        self.embedding = 50
        self.num_epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.01
        self.l2_reg_weight = 0.0007

        self.width = 80
        self.height = 35
        self.timesteps = 10
        self.num_channels = 1

    def create_cnn(self):

        net = InputLayer(
                shape=(None, self.num_channels, self.width, self.height))

        net = Conv2DLayer(
                net,
                num_filters=self.num_filters,
                filter_size=(self.filter_size, self.filter_size),
                nonlinearity=rectify,
                W=init.GlorotUniform())

        net = MaxPool2DLayer(
                net,
                pool_size=(self.pool_size, self.pool_size))

        net = Conv2DLayer(
                net,
                num_filters=self.num_filters,
                filter_size=(self.filter_size, self.filter_size),
                nonlinearity=rectify)

        net = MaxPool2DLayer(
                net,
                pool_size=(self.pool_size, self.pool_size))

        net = DenseLayer(
                net,
                num_units=self.embedding,
                nonlinearity=rectify)

        return net

    def create_model(self, input_spread, output_spread):

        l_map_in = InputLayer(shape=(
            None, self.timesteps, self.num_channels,
            self.width, self.height))

        l_in_hid = self.create_cnn()

        l_hid_hid = DenseLayer(
                InputLayer(l_in_hid.output_shape),
                num_units=self.embedding)

        l_pre = CustomRecurrentLayer(
                l_map_in, l_in_hid, l_hid_hid)

        l_stat_in = InputLayer(shape=(None, self.timesteps, input_spread))

        net = ConcatLayer([l_pre, l_stat_in], axis=2)

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        net = SliceLayer(net, -1, 1)

        net = DenseLayer(
                net,
                num_units=output_spread,
                W=init.Normal(),
                nonlinearity=softmax)

        return net

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        supp_model_params = {}

        temp_spread = len(train_Xs[1][0, 0, :])
        supp_model_params['input_spread'] = temp_spread
        supp_model_params['output_spread'] = temp_spread

        return supp_model_params

    def get_data(self):

        [
                min_stat_train_X, min_train_y,
                _, _, _, _, _, _,
                min_map_train_X, _
        ] = gen_map_data(
                width=self.width, height=self.height,
                timesteps=self.timesteps, color='hsv')

        [
                map_train_X, map_val_X,
                stat_train_X, stat_val_X,
                train_y, val_y
        ] = split_val(min_map_train_X, min_stat_train_X, min_train_y)

        train_Xs = [map_train_X, stat_train_X]
        val_Xs = [map_val_X, stat_val_X]

        return train_Xs, val_Xs, train_y, val_y


def bayes_hyper_optim_station():

    model = MapModel()

    hp_ranges = {
            'num_hidden': (100, 1024),
            'num_epochs': (5, 100),
            'batch_size': (64, 512),
            'dropout_val': (0, 0.9),
            'learning_rate': (1e-5, 1e-1),
            'grad_clip': (50, 1000),
            'l2_reg_weight': (0, 1e-1)}

    optim = BayesHyperOptim(model, hp_ranges)

    optim.optimize()


def grid_hyper_optim_station():

    model = MapModel()

    hp_choices = {
            'num_hidden': (128, 256, 512),
            'num_epochs': (128,),
            'batch_size': (256,),
            'dropout_val': (0.4, 0.5, 0.6),
            'learning_rate': (1e-5, 1e-3, 1e-2, 1e-1),
            'grad_clip': (100,),
            'l2_reg_weight': (0, 1e-4, 1e-2)}

    optim = GridHyperOptim(model, hp_choices)

    optim.optimize()


def train_default():

    model = MapModel()
    model.train_with_data()


def main():

    train_default()

    # bayes_hyper_optim_station()

    # grid_hyper_optim_station()


if __name__ == '__main__':
    main()
