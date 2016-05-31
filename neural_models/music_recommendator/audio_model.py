from lasagne import init
from lasagne.layers import InputLayer, LSTMLayer, \
    DropoutLayer, SliceLayer, DenseLayer, ConcatLayer, \
    Conv1DLayer, MaxPool1DLayer  # , ReshapeLayer
# from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.layers import get_output, get_all_layers
from lasagne.nonlinearities import tanh
from lasagne.updates import rmsprop

import theano

from neural_models.data.music_recommendator.user_data import \
    gen_audio_dataset, base_param_fnm
from neural_models.lib import net_on_seq
from neural_models.hyper_optim import GridHyperOptim

from neural_models.models import RegressionModel


class AudioModel(RegressionModel):

    def get_default_param_filename(self):

        return 'params/music_recommendator/audio_model.p'

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

        pass

    def load_train_hyperparams(self, hyperparams):

        self.num_epochs = int(hyperparams['num_epochs'])
        self.learning_rate = float(hyperparams['learning_rate'])
        self.batch_size = int(hyperparams['batch_size'])

    def load_default_hyperparams(self):

        self.load_default_data_hyperparams()
        self.load_default_train_hyperparams()
        self.load_default_net_hyperparams()

    def load_default_data_hyperparams(self):

        self.num_mels = 6
        self.song_length = 2000
        self.num_songs = 1500

    def load_default_net_hyperparams(self):

        self.num_hidden = 100
        self.dropout_val = 0.2
        self.grad_clip = 927
        self.l2_reg_weight = 0.0007
        self.embedding = 100

        self.num_filters = 32

    def load_default_train_hyperparams(self):

        self.num_epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.0001

    def create_lstm_stack(self, net):

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        return net

    def create_cnn_stack(self, net):

        net = Conv1DLayer(
            net,
            num_filters=self.num_filters,
            filter_size=3,
            stride=1,
            nonlinearity=tanh,
            W=init.GlorotUniform())

        net = MaxPool1DLayer(
            net,
            pool_size=2)

        net = DropoutLayer(net, self.dropout_val)

        return net

    def create_song_embedding(self):

        # shape=(num_songs, song_length, num_mels)
        i_song = InputLayer(
            shape=(None, self.num_mels, self.song_length),
            name='i_song')
        # i_song = ReshapeLayer(i_song, ([0], 1, [1], [2]))

        self.i_song = i_song
        net = i_song

        for _ in range(1):
            net = self.create_cnn_stack(net)

        # output_shape=(num_songs, embedding)
        # net = SliceLayer(net, -1, 1)
        net = DenseLayer(
                net,
                num_units=self.embedding,
                W=init.Normal(),
                nonlinearity=tanh)

        return net, i_song

    def create_song_encoder(self, l_song_embedding):

        # shape=(num_users, num_songs, song_length, num_mels)
        i_user_songs = InputLayer(
            shape=(None, None, self.num_mels, self.song_length),
            name='i_user_songs')

        l_song_encoder = net_on_seq(l_song_embedding, i_user_songs)
        # output_shape=(num_users, num_songs, embedding)

        return l_song_encoder, i_user_songs

    def create_pref_embedding(self, l_song_vals):

        # shape=(num_users, num_songs, embedding + 1 (value is play_count))
        net = l_song_vals

        for _ in range(1):
            net = self.create_lstm_stack(net)

        # output_shape=(num_users, embedding)
        net = SliceLayer(net, -1, 1)
        net = DenseLayer(
                net,
                num_units=self.embedding,
                W=init.Normal(),
                nonlinearity=tanh)

        self.pref_embedding = net

        return net

    def create_user_pref_encoder(self, l_song_embedding):

        # shape=(num_users, num_songs, embedding)
        l_song_encoder, i_user_songs = self.create_song_encoder(l_song_embedding)
        self.layers += get_all_layers(l_song_encoder)

        l_song_encoder = InputLayer(
            shape=(None, None, self.embedding),
            input_var=get_output(l_song_encoder),
            name='l_song_encoder')
        self.i_user_song_embeddings = l_song_encoder

        # shape=(num_users, num_songs, 1 (value is play_count))
        i_user_counts = InputLayer(
            shape=(None, None, 1),
            name='i_user_counts')

        # shape=(num_users, num_songs, embedding + 1 (value is play_count))
        l_song_vals = ConcatLayer(
            [i_user_counts, l_song_encoder],
            axis=2,
            name='l_song_vals')

        # output_shape=(num_users, embedding)
        l_user_prefs = self.create_pref_embedding(l_song_vals)

        return l_user_prefs, i_user_songs, i_user_counts

    def dense_stack(self, net, num_units=None, nonlinearity=tanh):

        if num_units is None:
            num_units = self.num_hidden

        net = DenseLayer(
                net,
                num_units=num_units,
                W=init.Uniform(),
                nonlinearity=nonlinearity)

        return net

    def create_model(self):

        # shape=(num_users, embedding)
        l_song_embedding, i_input_song = self.create_song_embedding()
        self.layers += get_all_layers(l_song_embedding)
        self.i_input_song = i_input_song
        self.song_embedding = l_song_embedding

        i_input_song_embedding = InputLayer(
            (None, self.embedding),
            input_var=get_output(l_song_embedding),
            name='i_input_song_embedding')
        self.i_input_song_embedding = i_input_song_embedding

        # shape=(num_users, embedding)
        l_user_prefs, i_user_songs, i_user_counts = \
            self.create_user_pref_encoder(l_song_embedding)
        self.i_user_songs = i_user_songs
        self.i_user_counts = i_user_counts
        self.layers += get_all_layers(l_user_prefs)

        l_user_prefs = InputLayer(
            (None, self.embedding),
            input_var=get_output(l_user_prefs),
            name='l_user_prefs')
        self.i_prefs = l_user_prefs

        # shape=(num_users, 2*embedding)
        net = ConcatLayer(
            [i_input_song_embedding, l_user_prefs],
            axis=1,
            name='concat')

        for _ in range(3):
            net = self.dense_stack(net)

        net = self.dense_stack(net, nonlinearity=None)

        net = self.dense_stack(net, nonlinearity=None, num_units=1)

        net = SliceLayer(net, 0, 1)

        self.net = net
        self.layers += get_all_layers(net)

        return [i_user_songs, i_user_counts, i_input_song]

    def build_song_embedding_fn(self):

        embedding_out = get_output(
            self.song_embedding,
            deterministic=True)
        embedding_fn = theano.function(
            [self.i_input_song.input_var], embedding_out,
            allow_input_downcast=True)

        self.get_song_embedding = embedding_fn

    def build_pref_embedding_fn(self):

        pref_out = get_output(
            self.pref_embedding,
            deterministic=True)
        pref_fn = theano.function(
            [self.i_user_song_embeddings.input_var, self.i_user_counts.input_var],
            pref_out,
            allow_input_downcast=True)

        self.get_user_prefs = pref_fn

    def build_pred_fn(self):

        pred_out = get_output(
            self.net,
            deterministic=True)
        pred_fn = theano.function(
            [self.i_input_song_embedding.input_var, self.i_prefs.input_var],
            pred_out,
            allow_input_downcast=True)

        self.get_preds = pred_fn

    def build_std_pred_fn(self):

        pred_out = get_output(
            self.net,
            deterministic=True)
        std_pred_fn = theano.function(
            [self.i_input_song.input_var, self.i_user_songs.input_var, self.i_user_counts.input_var],
            pred_out,
            allow_input_downcast=True)

        self.get_std_preds = std_pred_fn

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        return None

    def update_params(self, loss, all_params):

        return rmsprop(loss, all_params, self.learning_rate)

    def load_train_with_data_config(self):

        self.verbose = True
        self.save = True
        self.epoch_save = False

    def load_train_with_megabatches_config(self):

        self.verbose = True
        self.save = True
        self.epoch_save = False

    def get_megabatches(self):

        data = None

        for i in range(100):
            print('-' * 10 + 'Megabatch %d' % i + '-' * 10)
            del data
            data = self.get_data(verbose=False)
            yield data

    def test_hyperparams(self, **hyperparams):

        self.load_test_hyperparams_config()

        self.set_hyperparams(hyperparams)

        try:
            data = self.get_data()

            self.pretrain_compile(*data)

            val_loss, val_acc = self.train_model(
                    *data,
                    save=False, verbose=True, val=True)
        except:
            return -10.0

        return -val_acc

    def get_data(self, verbose=True):

        [
                train_user_songs_X, train_user_count_X,
                train_song_X, train_y
        ] = gen_audio_dataset(
            mode='train',
            num_truncated_songs=self.num_songs,
            verbose=verbose)

        [
                val_user_songs_X, val_user_count_X,
                val_song_X, val_y
        ] = gen_audio_dataset(
            mode='val',
            num_truncated_songs=int(self.num_songs * 0.70),
            shuffle=False,
            verbose=verbose)

        train_Xs = [
                train_user_songs_X,
                train_user_count_X,
                train_song_X]

        val_Xs = [
                val_user_songs_X,
                val_user_count_X,
                val_song_X]

        return train_Xs, val_Xs, train_y, val_y


def train_default():

    param_fnm = base_param_fnm + 'audio_model_strict_n3500,l0.015,t5.p'
    model = AudioModel(param_filename=param_fnm)
    model.train_with_data()


def train_megabatches():

    param_fnm = base_param_fnm + 'audio_model_strict_n3500,l0.015,t5.p'
    model = AudioModel(param_filename=param_fnm)
    model.train_with_megabatches()


def grid_hp_search():

    hp_choices = {
        'num_hidden': (100, 200),
        'dropout_val': (0.2, 0.4),
        'grad_clip': (1000,),
        'l2_reg_weight': (0.0,),
        'embedding': (100, 200),
        'num_epochs': (2, 5, 10),
        'batch_size': (12, ),
        'learning_rate': (0.00001, 0.0001, 0.001)}

    param_fnm = base_param_fnm + 'audio_model_strict_n3500,l0.015,hp4.p'
    model = AudioModel(param_filename=param_fnm)

    optim = GridHyperOptim(model, hp_choices)
    optim.optimize()


def main():

    # grid_hp_search()
    # train_default()
    train_megabatches()


if __name__ == '__main__':
    main()
