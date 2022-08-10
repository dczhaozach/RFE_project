import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import yaml
import copy


def parse_config(config_file):
    file_path = Path.cwd() / config_file
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_path):
    """
    Read more about logging: https://www.machinelearningplus.com/python/python-logging-guide/
    Args:
        log_path [str]: eg: "../log/train.log"
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger

def lag_variable(df, time_var, id_var, var, lags):
    """
    lag_variable create lag variables by groups   
    Args:
        df [DataFrame]: dataframe
        time_var [lst]: time variable
        id_var [list of string]: id variable
        var [lst]: lagged varible
        lags: number of lags (1 means one lag)
    """

    index_var = time_var + id_var

    df = df.set_index(index_var)
    df_shift = df[var]
    
    shifted = df_shift.groupby(level=id_var).shift(-lags)
    shifted = shifted.rename(columns={var[0]: f"L_{lags}_{var[0]}"})
    df = df.join(shifted)
    df = df.reset_index()
    
    return df

def coef_dict(v_names, res, ceof_dict, age):
    """
    coef_dict create dictionary of estimates and C.I. for selected parameters 
    Args:
        v_name [list]: list of parameters
        res []: regression results
        ceof_dict [lst]: list to append 
        
    Return:
        new dictionary
    """
    temp_dict = copy.deepcopy(ceof_dict)
    for v_name in v_names:
        dict1 = {}
        coefs_value = res.params[v_name]
        lower_ci = res.conf_int().loc[v_name, "lower"]
        upper_ci = res.conf_int().loc[v_name, "upper"]
        sign = (lower_ci * upper_ci > 0)
        dict1.update({"name": v_name, "age": age, "Coef": coefs_value,
                      "lower_ci": lower_ci, "upper_ci": upper_ci,
                      "significance": sign}) 
        temp_dict.append(dict1)
    
    return temp_dict


def plot_lp(v_names, df, depend_var, plot_name, fig_path, var_names, std):
    """
    plot_lp plot local projection graph
    Args:
        df [DataFrame]: Data contains coefs
        depend_var [str]: dependent variable
        plot_name [str]: name for the plot 
        var_names [lst]: lst of string of variable names
        fig_path [Path]: path of the figure
    Return:
        None
    """
    n_plot = len(v_names)
    cols = 2
    rows = n_plot//2 + (n_plot%2 > 0)
    fig = plt.figure()
    gs = gridspec.GridSpec(rows, cols)
    depend_name = depend_var.replace("_", " ").title()

    for i in range(0, n_plot):
        v_name = v_names[i]
        var_name = var_names[i]
        df_sub = df[df.name == v_name]
        Age = df_sub.age.to_numpy().T
        Age = Age.astype(int)
        Coef = df_sub.Coef.to_numpy().T 
        Coef = Coef * std
        lower_ci = - df_sub[["lower_ci"]].to_numpy().T + df_sub.Coef.to_numpy().T 
        lower_ci  = lower_ci * std
        upper_ci = df_sub[["upper_ci"]].to_numpy().T - df_sub.Coef.to_numpy().T 
        upper_ci = upper_ci * std
        yerr = np.vstack((lower_ci, upper_ci))
        
        ax = fig.add_subplot(gs[i])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[i].set_ylim([-0.1, 0.1])
        ax.scatter(Age, Coef)
        ax.errorbar(Age, Coef, yerr = yerr, fmt = 'o',color = 'orange', 
            ecolor = 'lightgreen', elinewidth = 3, capsize=5)   
        ax.axhline(y=0, color='r', linestyle=':')
        
        ax.set_title(f"{var_name}".title())
        ax.set_xlabel("Age")
        ax.set_ylabel(f"{depend_name}".title())
    
    fig.suptitle(f'Effects of Regulation on {depend_name}')        
    fig.tight_layout()
    fig_final_path = fig_path/f"{plot_name}.png"
    fig.savefig(fig_final_path, facecolor='white', transparent=False)
    
    return None

