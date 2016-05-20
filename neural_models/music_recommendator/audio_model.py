from lasagne import init
from lasagne.layers import InputLayer, LSTMLayer
from lasagne.layers import DropoutLayer, SliceLayer, DenseLayer
from lasagne.layers import CustomRecurrentLayer, ConcatLayer
from lasagne.nonlinearities import tanh, softmax

from neural_models.data.music_recommendator.user_data import gen_audio_dataset

from neural_models.lib import split_val

from neural_models.models import Model

from neural_models.hyper_optim import BayesHyperOptim, GridHyperOptim


class AudioModel(Model):

    def get_default_param_filename(self):

        return 'params/phys_weather/station_model.p'

    def load_hyperparams(self, hyperparams):

        self.load_data_hyperparams(hyperparams)
        self.load_train_hyperparams(hyperparams)
        self.load_net_hyperparams(hyperparams)

    def load_net_hyperparams(self, hyperparams):

        self.num_hidden = int(hyperparams['num_hidden'])
        self.dropout_val = float(hyperparams['dropout_val'])
        self.grad_clip = float(hyperparams['grad_clip'])
        self.l2_reg_weight = float(hyperparams['l2_reg_weight'])

    def load_data_hyperparams(self, hyperparams):

        self.timesteps = int(hyperparams['timesteps'])

    def load_train_hyperparams(self, hyperparams):

        self.num_epochs = int(hyperparams['num_epochs'])
        self.learning_rate = float(hyperparams['learning_rate'])
        self.batch_size = int(hyperparams['batch_size'])

    def load_default_hyperparams(self):

        self.load_default_data_hyperparams()
        self.load_default_train_hyperparams()
        self.load_default_net_hyperparams()

    def load_default_data_hyperparams(self):

        self.timesteps = 30

    def load_default_net_hyperparams(self):

        self.num_hidden = 250
        self.dropout_val = 0.0
        self.grad_clip = 927
        self.l2_reg_weight = 0.0007

    def load_default_train_hyperparams(self):

        self.num_epochs = 54
        self.batch_size = 365
        self.learning_rate = 0.055

    def create_lstm_stack(self, net):

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        return net

    def create_song_embedding(self):

        # shape=(num_songs, song_length, bitwidth)
        net = InputLayer(shape=(None, None, self.bitwidth))

        for _ in range(2):
            net = self.create_lstm_stack(net)

        # output_shape=(num_songs, embedding)
        net = SliceLayer(net, -1, 1)
        net = DenseLayer(
                net,
                num_units=self.embedding,
                W=init.Normal(),
                nonlinearity=softmax)

        return net

    def create_input_song_encoder(self):

        # shape=(num_users, song_length, bitwidth)
        net = self.create_song_embedding()

        return net

    def create_song_encoder(self):

        # shape=(num_users, num_songs, song_length, bitwidth)
        l_songs_in = InputLayer(shape=(
            None, None, None, self.bitwidth))

        l_in_hid = self.create_song_embedding()

        l_hid_hid = DenseLayer(
                InputLayer(l_in_hid.output_shape),
                num_units=self.embedding)

        l_song_encoder = CustomRecurrentLayer(
                l_songs_in, l_in_hid, l_hid_hid)
        # output_shape=(num_users, num_songs, embedding)

        return l_song_encoder

    def create_pref_embedding(self):

        # shape=(num_users, num_songs, embedding + 1 (value is play_count))
        net = InputLayer(shape=(None, None, self.embedding + 1))

        for _ in range(2):
            net = self.create_lstm_stack(net)

        # output_shape=(num_users, embedding)
        net = SliceLayer(net, -1, 1)
        net = DenseLayer(
                net,
                num_units=self.embedding,
                W=init.Normal(),
                nonlinearity=softmax)

        return net

    def create_user_pref_encoder(self):

        # shape=(num_users, num_songs, embedding)
        l_song_encoder = self.create_song_encoder()

        # shape=(num_users, num_songs, 1 (value is play_count))
        l_song_counts_in = InputLayer(shape=(
            None, None, 1))

        # shape=(num_users, num_songs, embedding + 1 (value is play_count))
        l_song_vals = ConcatLayer([l_song_counts_in, l_song_encoder], axis=2)

        l_in_hid = self.create_pref_encoder()

        l_hid_hid = DenseLayer(
                InputLayer(l_in_hid.output_shape),
                num_units=self.embedding)

        l_user_prefs = CustomRecurrentLayer(
                l_song_vals, l_in_hid, l_hid_hid)
        # output_shape=(num_users, embedding)

        return l_user_prefs

    def create_model(self, **kwargs):

        # shape=(num_users, embedding)
        l_user_prefs = self.create_song_prefs()

        # shape=(num_users, embedding)
        l_input_song_encoder = self.create_input_song_encoder()

        # shape=(num_users, 2*embedding)
        net = ConcatLayer([l_user_prefs, l_input_song_encoder], axis=1)

        net = DenseLayer(
                net,
                num_units=self.num_hidden_units,
                W=init.Normal(),
                nonlinearity=softmax)

        net = DenseLayer(
                net,
                num_units=1,
                W=init.Normal(),
                nonlinearity=softmax)

        return net

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        return None

    def get_data(self):

        [
                train_user_songs_X, train_song_X, train_y,
                _, _, _
        ] = gen_audio_dataset(timesteps=self.timesteps)
        [
                train_user_songs_X, val_user_songs_X,
                train_song_X, val_song_X,
                train_y, val_y
        ] = split_val(train_user_songs_X, train_song_X, train_y)
        train_Xs = [train_user_songs_X, train_song_X]
        val_Xs = [val_user_songs_X, val_song_X]

        return train_Xs, val_Xs, train_y, val_y


def bayes_hyper_optim():

    model = AudioModel()

    hp_ranges = {
            'num_hidden': (100, 1024),
            'num_epochs': (5, 100),
            'timesteps': (5, 30),
            'batch_size': (64, 512),
            'dropout_val': (0, 0.9),
            'learning_rate': (1e-5, 1e-1),
            'grad_clip': (50, 1000),
            'l2_reg_weight': (0, 1e-1)}

    optim = BayesHyperOptim(model, hp_ranges)

    optim.optimize()


def grid_hyper_optim():

    model = AudioModel()

    hp_choices = {
            'num_hidden': (128, 256, 512),
            'num_epochs': (128,),
            'timesteps': (10, 30),
            'batch_size': (256,),
            'dropout_val': (0.4, 0.5, 0.6),
            'learning_rate': (1e-5, 1e-3, 1e-2, 1e-1),
            'grad_clip': (100,),
            'l2_reg_weight': (0, 1e-4, 1e-2)}

    optim = GridHyperOptim(model, hp_choices)

    optim.optimize()


def train_default():

    model = AudioModel()
    model.train_with_data()


def main():

    train_default()

    # bayes_hyper_optim_station()

    # grid_hyper_optim_station()


if __name__ == '__main__':
    main()
