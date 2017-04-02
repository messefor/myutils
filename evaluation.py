"""
==========================
Utils for evaluation

ODA, Daisuke
==========================

Todo:
- make useful pairplot function
-

""""

import decimal
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _target_table_ratio(data, x, y , trg_value='1', ratio_column='ratio'):
    """Calc target class ratio table """
    df_table = \
        data.groupby([x, y])[[x]].count().reset_index(y).pivot(columns=y)
    df_table.columns = df_table.columns.droplevel()
    df_table[ratio_column] = df_table[trg_value] / df_table.sum(axis=1)
    return df_table


def plot_trg_ratio(data, trg_col, trg_value=1):
    """Plot target flg ratio for each column value

    Todo
    ------------------
    - Add keywordarg(**keywd) to pass pyplot
    - Make userful pairplot
    - Plot histogram not line
    """
    data_cp = data.copy()

    types_to_cut = (float, decimal.Decimal, int)
    bins = 50
    max_rows = bins

    plot_col = [ col for col in data_cp.columns if col != trg_col ]
    ncols = 4
    nrows = int(math.ceil(len(plot_col) * 1.0 / ncols))

    figsize=(17, 4 * nrows)
    fig = plt.figure(figsize=figsize)

    for idx, col in enumerate(plot_col):

        # cut if continuos values
        first_value = data_cp[[col]].dropna().ix[0,0]
        if isinstance(first_value, types_to_cut) and\
            len(data_cp[[col]].groupby(col).groups) > bins * 2:
            data_cp[col] = pd.cut(data_cp[col], bins=bins, include_lowest=True)

        ax = fig.add_subplot(nrows, ncols, idx + 1,)

        df_plot = _target_table_ratio(data_cp, x=col, y=trg_col,
            trg_value=trg_value)

        plot_data=df_plot.head(max_rows)

        ax.plot(plot_data[0], 'g-')
        ax.plot(plot_data[1], 'b-')
        ax2 = ax.twinx()
        ax2.plot(plot_data['ratio'],'r--')
        ax.legend(loc='upper right')
        ax2.legend(loc='center right')

        ax.set_xlabel(col)
        ax.set_title(col)

    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=2.0)
    fig.subplots_adjust(top=0.9, bottom=0.25)
    plt.suptitle('Ratio of target flag')
    plt.show()


def get_learning_curve_score(fit_function, X_train, y_train, X_test, y_test,
    n_sample_seq=None):
    """Calc metrics to build learning curve"""

    hist_train = []
    hist_test = []

    if n_sample_seq is None:
        n_sample_seq = range(100, X_train.shape[0], 100)

    for n_sample in n_sample_seq:

        rowidx = np.random.choice(range(X_train.shape[0]), n_sample)

        X_sampled = X_train[rowidx,:]
        y_sampled = y_train[rowidx]

        model = fit_function(X_sampled, y_sampled)

        auc_train, rmse_train = get_pred_metrics(model, X_sampled, y_sampled)
        auc_test, rmse_test = get_pred_metrics(model, X_test, y_test)

        hist_train.append(auc_train) # RNSE
        hist_test.append(auc_test)

    return {'n_sample': n_sample_seq, 'train': hist_train, 'test': hist_test}




def get_pred_metrics(model, X, y_true):

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    auc = get_metrics(y_true, y_prob, y_pred)['auc']
    rmse = ((y_prob - y_true) ** 2).mean()
    return auc, rmse




def plot_roc(plot_confs):

    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    plt.figure(figsize=(20, 20))
    lw = 2

    for plot_conf in plot_confs:
        label='{caption} ROC(auc={auc:0.5f})'.format(**plot_conf)
        plot_opts_keys = ['color']
        plot_opts = { k: v for k, v in plot_conf.items() if k in plot_opts_keys }
        plt.plot(plot_conf['fpr'], plot_conf['tpr'], label=label, **plot_opts)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver oprating charastistic')
    plt.legend(loc='lower right',prop={'size':14})
    plt.show()


def get_metrics(y_true, y_score, y_pred):
    """Returns evaluation matrix

    Returns: {dict}
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    area = auc(fpr, tpr)
    return \
    {'f1': f1_score(y_true, y_pred),
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'ap': average_precision_score(y_true, y_score) ,
    'fpr': fpr,
    'tpr': tpr,
    'auc': area,
    'cm': confusion_matrix(y_true, y_pred)
    }
