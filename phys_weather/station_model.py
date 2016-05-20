from neural_models.phys_weather.station_model import StationModel
from neural_models.hyper_optim import GridHyperOptim, BayesHyperOptim


def bayes_hyper_optim_station():

    model = StationModel()

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

    model = StationModel()

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

    model = StationModel()
    model.train_with_data()


def main():

    train_default()

    # bayes_hyper_optim_station()

    # grid_hyper_optim_station()


if __name__ == '__main__':
    main()
