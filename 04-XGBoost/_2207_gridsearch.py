import warnings
from sklearn.model_selection     import GridSearchCV
import numpy as np

class GridSearch():

    def __init__(
        self, 
        x_train,
        y_train,
        model_gridsearch,
        param_grid,
        n_jobs=-1,
        cv=4,
        verbose=4
        ):
        
        self.x_train  = x_train
        self.y_train = y_train
        self.model_gridsearch    = model_gridsearch
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose

    # perform grid search and return results
    def get_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(estimator=self.model_gridsearch, param_grid=self.param_grid, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
            self.grid_result = grid.fit(self.x_train, self.y_train, verbose=self.verbose)
        return self.grid_result

    # print sorted mean scores
    def print_result(self, automatic_best=True, verbose=1):
        means = self.grid_result.cv_results_['mean_test_score']
        stds = self.grid_result.cv_results_['std_test_score']
        params = np.array(self.grid_result.cv_results_['params'])

        idx   = np.argsort(means)[::-1]
        means = means[idx]
        stds  = stds[idx]
        params = params[idx]

        if automatic_best:
            print("Best: %f using %s" % (self.grid_result.best_score_, self.grid_result.best_params_))
        else:
            print("Best: %f using %s" % (means[0], params[0]))

        if verbose > 0:
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))