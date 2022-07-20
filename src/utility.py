import logging
from pathlib import Path
import pandas as pd
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
        time_var [str]: time variable
        id_var [list of string]: id variable
        var [str]: lagged varible
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
        ceof_dict [dict]: dictionary to append 
        
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