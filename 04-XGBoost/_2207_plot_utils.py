import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
from sklearn import metrics
from xgboost import plot_tree

def plot_data(
    x, t, fig, 
    n_samples  = 3,
    subplot_id = 111, 
    title      = "data", 
    fontsize   = 14,
    lw         = 2,
    colors     = ("tab:blue", "tab:orange", "tab:green"),
    labels     = (0, 1, 2),
    legend     = True,
):

    ax = fig.add_subplot(subplot_id)

    for i in range(n_samples):
        ax.plot(t[i], x[i], lw=2, color=colors[i % 3], label=labels[i % 3])

    ax.set_title(title,   fontsize=fontsize+4)
    ax.set_xlabel("time", fontsize=fontsize)
    ax.tick_params(axis="x", which="major", labelsize=fontsize, length=5)



    if legend:
        custom_lines = [
            Line2D([0], [0], color="tab:blue",   lw=lw),
            Line2D([0], [0], color="tab:orange", lw=lw),
            Line2D([0], [0], color="tab:green",  lw=lw)
        ]
        custom_labels = [
            "0", "1", "2"
        ]
                
        ax.legend(custom_lines, custom_labels, fontsize=fontsize-2, title="label", title_fontsize=fontsize)

    return ax

def scatter_results(
    parameter,
    result,
    error, 
    fig, 
    subplot_id   = 111,
    ax           = None,
    label        = None,
    par_label    = None,
    metric_label = None,
    color        = "tab:blue",
    lw           = 1,
    ls           = "-",
    ms           = 12,
    mnorm        = None,
    mstyle       = "o",
    fontsize     = 18,
    legend       = True,
    title        = "plot"
):
    if ax is None:
        ax = fig.add_subplot(subplot_id)
    
    ax.set_title(title,         fontsize=fontsize+4)
    ax.set_xlabel(par_label,     fontsize=fontsize)
    ax.set_ylabel(metric_label, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)

    sns.scatterplot(
        x         = parameter, 
        y         = result,
        size      = np.array(ms),
        sizes     = (100, 500),
        size_norm = mnorm,
        marker    = mstyle,
        palette   = color,
        color     = color,
        label     = label,
        ax        = ax,
        legend    = False
    )

    ax.errorbar(
        parameter, 
        result,
        error,
        color = color,
        ls    = ls,
        lw    = lw,
        zorder = 0
    )

    if legend:
        ax.legend(fontsize=fontsize-4)

    return ax

def plot_confusion_matrix(
    cm,
    ax,
    cmap     = "GnBu_r",
    labels   = [0, 1],
    fontsize = 18,
    title    = None,
):

    mat = ax.matshow(cm, cmap=cmap)

    threshold = mat.norm(cm.max())/2.
    textcolors = ["white", "black"]
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(
                j, 
                i, 
                f"{cm[i, j]}",#*100:.1f}%", 
                ha       = "center", 
                va       = "center", 
                color    = textcolors[int(mat.norm(cm[i, j]) > threshold)],
                fontsize = fontsize
            )

    ax.set_title(title,      fontsize=fontsize+4)
    ax.set_xlabel("predicted labels", fontsize=fontsize)
    ax.set_ylabel("true labels", fontsize=fontsize)

    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)
    
    return ax


def show_perf(
    clf, 
    ds,
    S, 
    fname="tree-classif.png"
):

    cmap1 = 'winter'
    cmap2 = 'plasma'
    
    print("errors: {:.2f}%".format(100*(1-clf.score(ds.xtest, ds.ytest))))
    ypred  = clf.predict(ds.xtest)

    dx     = 0.02
    x_plot = np.mgrid[-S:S+dx:dx, -S:S+dx:dx].reshape(2, -1).T
    y_plot = clf.predict(x_plot)

    # plot data and predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.set_title(str(clf))
    ax1.scatter(x_plot[:,0], x_plot[:,1], c=y_plot, cmap=cmap1, vmax=1.2, s=1, alpha=0.8)
    ax1.scatter(ds.xtrain[:,0], ds.xtrain[:,1], c=ds.ytrain, cmap=cmap2, vmax=1.1, s=7)
    # plot cm
    plot_confusion_matrix(cm=metrics.confusion_matrix(ds.ytest, ypred), ax=ax2)
    plt.show()

    # plot trees
    num_trees = len(clf.get_booster().get_dump())

    fig, axs = plt.subplots(num_trees,1,figsize=(30, 30))
    for i, ax in enumerate(axs):
        plot_tree(clf, num_trees=i, ax=ax)
    fig.savefig("DATA/"+fname, dpi=300, pad_inches=0.02)   
    plt.show()
