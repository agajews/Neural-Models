import itertools

from bayes_opt import BayesianOptimization


class HyperOptim():

    def __init__(self):

        pass

    def compute_results(self):

        raise NotImplementedError()

    def display_results(self, results):

        print('Final Results: %f%%' % (results * 100))

    def optimize(self):

        res = self.compute_results()
        self.display_results(res)


class GridHyperOptim(HyperOptim):

    def __init__(self, model, hp_choices, key_order=None):

        self.key_order = self.get_key_order(key_order, hp_choices)
        self.hp_choices = hp_choices
        self.model = model

    def set_key_order(self, key_order):

        self.key_order = key_order

    def set_hp_choices(self, hp_choices):

        self.hp_choices = hp_choices

    def set_model(self, model):

        self.model = model

    def build_choices(self):

        choices = []

        for key in self.key_order:
            array = list(self.hp_choices[key])
            choices.append(array)

        return itertools.product(*choices)

    def get_hyperparams(self, choice):

        hyperparams = {}

        for i, key in enumerate(self.key_order):
            hyperparams[key] = choice[i]

        return hyperparams

    def display_choice(self, hyperparams, res):

        print(hyperparams)
        print(res)

    def evaluate_choice(self, choice):

        hyperparams = self.get_hyperparams(choice)

        res = self.model.test_hyperparams(**hyperparams)

        self.display_choice(hyperparams, res)

        return res

    def get_key_order(self, key_order, hp_choices):

        if key_order is None:
            key_order = hp_choices.keys()

        return key_order

    def compute_results(self):

        choices = self.build_choices()

        all_results = []

        for choice in choices:

            res = self.evaluate_choice(choice)
            all_results.append(res)

        return min(all_results)


class BayesHyperOptim(HyperOptim):

    def __init__(self,
            model, hp_ranges,
            kappa=5, acq='ucb',
            corr='cubic', nugget=1,
            num_iter=50,
            num_init_points=50):

        self.model = model
        self.hp_ranges = hp_ranges

        self.kappa = kappa
        self.corr = corr
        self.nugget = nugget
        self.acq = acq
        self.num_iter = num_iter
        self.num_init_points = num_init_points

    def get_gp_params(self):

        gp_params = {
                'corr': self.corr,
                'nugget': self.nugget,
                'init_points': self.num_init_points,
                'n_iter': self.num_iter,
                'acq': self.acq,
                'kappa': self.kappa}

        return gp_params

    def get_results(self, optim):

        maximum = optim.res['max']['max_val']

        return maximum

    def compute_results(self):

        optim = BayesianOptimization(self.model.test_hyperparams, self.hp_ranges)

        gp_params = self.get_gp_params()

        optim.maximize(**gp_params)

        return self.get_results(optim)
