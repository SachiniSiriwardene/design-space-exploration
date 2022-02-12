import GPyOpt
import GPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yaml
import os
import examples as ex

def obj_func_2d(x,y):
    return((x**2 + y**2)*(np.sin(x)**2 - np.cos(y)))

def drawTruePlot(n):

    fig = plt.figure(figsize=plt.figaspect(0.3))
    #X = np.arange(10, 10, 0.25)
    #Y = np.arange(0, 10, 0.25)
    #X, Y = np.meshgrid(X[:,0], X[:,1])

    x1 = np.outer(np.linspace(10, 100, n, dtype=int), np.ones(n))
    x2 = np.outer(np.linspace(10, 100, n, dtype=int), np.ones(n)).T
    #z = eval_func({"x1": x1.flatten(), "x2": x2.flatten()}).reshape(n, n)

    #x1 = np.linspace(min_X, max_X, 30).flatten()
    test_x1 = np.linspace(10, 100, n).flatten()

    #x2 = np.linspace(min_X, max_X, 30).flatten()
    test_x2 = np.linspace(10, 100, n).flatten()

    test_y = ex.run_experiment_for_eval(test_x1, test_x2)
    z = np.outer(test_y, np.ones(n))
    ax = fig.add_subplot(1, 2, 2, projection='3d')


    # Plot the surface.
    surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()
    plt.savefig("true_plot_50_new_int.png")
    return test_y

def plotPredicted(model, n):

    fig = plt.figure(figsize=plt.figaspect(0.3))


    pred_x1 = np.linspace(10, 100, n, dtype=int)
    pred_x2 = np.linspace(10, 100, n, dtype=int)


    x1 = np.outer(pred_x1, np.ones(n))
    x2 = np.outer(pred_x2, np.ones(n)).T
    z = []



    for a,b in np.c_[pred_x1.flatten(), pred_x2.flatten()]:
     mean, var = model.predict(np.array([a, b]))
     z.append(mean[0][0])

    predicted_points = z
    z = np.outer(z, np.ones(n))

    ax = fig.add_subplot(1, 2, 2, projection='3d')


    # Plot the surface.
    surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()

    plt.savefig("EI_Predicted_optimized_50.png")
    return predicted_points

def getNextValue(x_next):

    with open('config.yaml') as f:
        confDict = yaml.load(f, Loader=yaml.FullLoader)
    confDict['execution'][0]['concurrency'] = int(x_next[0,0])
    message = "x" * x_next[0,1]
    #'{"Message": ' + message + '}'
    confDict['scenarios']['sample']['requests'][0]['body'] = '{"Message": ' + message + '}'
    with open('config.yaml', "w") as f:
        yaml.dump(confDict, f)

    os.system('bzt config.yaml')
    data = pd.read_csv('results.csv')
    return data._get_value(1, 'avg_lt')

noise = 0.2
bounds = np.array([[-1.0, 2.0]])

def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

def sampleOpt():

    kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
    bds = [{'name': 'X', 'type': 'continuous', 'domain': bounds.ravel()}]

    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init)
    optimizer = GPyOpt.methods.BayesianOptimization(f=f,
                                 domain=bds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.01,
                                 X=X_init,
                                 Y=-Y_init,
                                 noise_var = noise**2,
                                 exact_feval=False,
                                 normalize_Y=False,
                                 maximize=True)

    optimizer.run_optimization(max_iter=10)
    optimizer.plot_acquisition("acq_sample.png")


def getNextPoint(X_init, Y_init, n, f):

    domain =[{'name': 'concurrency', 'type': 'discrete', 'domain': (10,100)},
             {'name': 'messagelength', 'type': 'discrete', 'domain': (10,100)}]
    X_step = X_init
    Y_step = Y_init
    current_iter = 0
    bo_step = None
    while current_iter < n:

        bo_step = GPyOpt.methods.BayesianOptimization(f = getNextValue, domain = domain, model_type='GP',
                                                      acquisition_type ='EI',
                                                      acquisition_jitter = 0.01,
                                                      X = X_step, Y = Y_step, noise_var = noise**2,
                                                      normalize_Y=False, acquisition_optimizer_type='lbfgs')

        x_next =  bo_step.suggest_next_locations()
        y_next = getNextValue(x_next)

        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_next))

        bo_step.run_optimization()
        bo_step.model.updateModel(X_init, Y_init, X_step, Y_step)

        current_iter += 1

    bo_step.plot_acquisition("acq_new_optimized-100.png")
    bo_step.plot_convergence("conv_new_optimized-100.png")
    y_pred = plotPredicted(bo_step.model, 50)
    y_real = drawTruePlot(50)
    bo_step.save_models("BayezOptModel")
    bo_step.save_evaluations("evaluationPoints")
    calculateError(y_real, y_pred)

def trainModel():

    domain =[{'name': 'concurrency', 'type': 'discrete', 'domain': (10,100)},
             {'name': 'messagelength', 'type': 'discrete', 'domain': (10,100)}]
    kernel = GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)

    iter_count = 10
    current_iter = 0
    noise = 0.2
    X_step = X_init
    Y_step = Y_init

    #kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)

    #while current_iter < iter_count:
    #bo_step = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step)

    bo_step = GPyOpt.methods.BayesianOptimization(f = getNextValue, domain = domain, model_type='GP',
                                                  acquisition_type ='EI',
                                                  acquisition_jitter = 0.01,
                                                  X = X_step, Y = Y_step, noise_var = noise**2,
                                                  normalize_Y=False)

    bo_step.run_optimization(max_iter=100)

    ins = bo_step.get_evaluations()
    print("Inputs:")
    print(ins)
    print("Outputs")
    outs = bo_step.get_evaluations()
    print(outs)

    print("Parameters :")
    print("Lengthscale :")
    print(kernel.lengthscale)
    print("Variance")
    print(kernel.variance)

    bo_step.plot_acquisition("acq.png")
    bo_step.plot_convergence("convergence.png")
    plotPredicted(bo_step.model, 30)


def calculateError(Y_real, Y_pred):

    gprrow = np.array([ex.mean_squared_error(Y_real, Y_pred),
                       ex.mean_absolute_percentage_error(Y_real, Y_pred),"GPR"])
    ex.write_error_toFile("error_new.csv", gprrow)


if __name__ == '__main__':


    #sampleOpt()

    #drawTruePlot()

    #func = GPyOpt.objective_examples.experiments1d.forrester()
    #X_init = np.array([[0.0],[0.5],[1.0]])
    #Y_init = func.f(X_init)

    init_df = pd.read_csv("data/out_1644133061.csv")
    target_col = "target"
    params = [param for param in init_df.columns if param != target_col]
    X_init = init_df[params].values
    Y_eval = init_df[target_col].values


    Y_init = init_df[target_col].values[:, np.newaxis]
    #X_init = np.array([[0.0],[0.5],[1.0]])
    #Y_init = func.f(X_init)

    #getNextPoint(X_init, Y_init, 100, getNextValue)
    drawTruePlot(50)


