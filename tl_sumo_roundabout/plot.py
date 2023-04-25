import pickle

import numpy as np
from numpy import ndarray
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from matplotlib import pyplot as plt


def moving_average(values: ndarray, window: int) -> ndarray:
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(
    log_folder: str, learning_curve_path: str, window: int = 100
) -> tuple[ndarray, ndarray]:
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x_orig, y_orig = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y_orig, window=window)
    # Truncate x
    x = x_orig[len(x_orig) - len(y) :]

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.savefig(learning_curve_path)

    with open(
        learning_curve_path + "_reward.pickle",
        mode="wb",
    ) as f:
        pickle.dump((x_orig, y_orig), f)

    return x_orig, y_orig


def plot_ablation(
    lcs: list[tuple[ndarray, ndarray]], plot_save_path: str, max_x: int, window: int
) -> None:
    xvals: ndarray = np.arange(0, max_x, 0.5)
    Y_interp: ndarray = np.array([np.interp(xvals, x, y) for x, y in lcs])
    y_min: ndarray = np.amin(Y_interp, axis=0)
    y_max: ndarray = np.amax(Y_interp, axis=0)
    y_mean: ndarray = np.mean(Y_interp, axis=0)

    fig, ax = plt.subplots()

    ax.fill_between(
        xvals,
        moving_average(y_min, window),
        moving_average(y_max, window),
        alpha=0.2,
        color="b",
    )
    ax.plot(
        xvals,
        moving_average(y_mean, window),
        color="b",
    )

    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("Rewards")
    plt.savefig(plot_save_path, dpi=600)
    plt.clf()
    plt.close()
