"""
This script build the regression models
"""

import warnings

import logging
import click
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from linearmodels.panel import PanelOLS

from src.utility import parse_config

# options of pandas
pd.options.mode.use_inf_as_na = True


def model_cohort(config):
    """
    model_cohort function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """

    ####################
    # Load data
    ####################
    
    # load data paths
    cleaned_data_path = Path(config["model"]["cohort_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)

    ####################
    # Regression
    ####################

    for naics_curr in range(2, 5):
        # load data
        data = df
        # sample restriction
        data = data[data.year > 1985]
        data = data[data.age_grp_dummy <= 5]
        
        # regression
        data = data.set_index(['sector', 'cohort'])
        data = data.loc[:, ['entry_rate_whole', 'entry_rate_whole_cohort', 'death_rate', 'age_grp_dummy', 'year', 'log_gdp', 'log_gdp_cohort',
                            f'log_restriction_2_{naics_curr}', f'log_restriction_2_{naics_curr}_cohort']].dropna()
        mod = PanelOLS.from_formula(formula = f'death_rate ~ log_restriction_2_{naics_curr} + log_restriction_2_{naics_curr}_cohort \
                                                + entry_rate_whole + entry_rate_whole_cohort  \
                                                + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data, drop_absorbed=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary

        file_path = Path.cwd()/results_tables_path/f"results_reg_naics_{naics_curr}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
            
            
def model_life_path(config):
    """
    model_life_path function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """
    ####################
    # Load data
    ####################

    # load data paths
    cleaned_data_path = Path(config["model"]["life_path_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)  
    
    for age in range(1, 6):
        # load data
        data = df
        
        # sample restriction
        data = data[data.year > 1985]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        
        # create exogenous variable (list)
        exo_var_list = []
        for lags in range(0, age + 1):
            exo_var_list.append(f"L_{lags}_log_restriction_2_0")
            exo_var_list.append(f"L_{lags}_entry_rate_whole")
            exo_var_list.append(f"L_{lags}_log_gdp")
        exo_var_list.sort()
        sample_data = data.loc[:, ["death_rate"] + exo_var_list].dropna()
        exo_vars = sm.add_constant(sample_data[exo_var_list])
        
        # regression
        mod = PanelOLS(sample_data.death_rate, exo_vars, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary
        coef = res.params
        ci = res.conf_int
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"results_path_age_{age}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
            
        
    # fig
    fig, ax = plt.subplots()
        
        
def model_average(config):
    """
    model_average function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """    
    # load data paths
    cleaned_data_path = Path(config["model"]["average_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    
    # load data
    data = df
    
    # sample restriction
    data = data[data.year > 1985]
    
    # regression
    data = data.set_index(['sector', 'year'])
    
    # regression
    data = data.loc[:, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "avg_log_restriction_2_0", "avg_entry_rate_whole", "avg_log_gdp",
                    'death_rate', 'age_grp_dummy']].dropna()
    mod = PanelOLS.from_formula(formula = f'death_rate ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                            + avg_log_restriction_2_0 + avg_entry_rate_whole + avg_log_gdp \
                                            + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data, drop_absorbed=True)
    res = mod.fit(cov_type = 'heteroskedastic')

    # results
    results = res.summary

    file_path = Path.cwd()/results_tables_path/f"results_average_all.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())
        
@click.command()
@click.argument("config_file", type=str, default="src/config.yaml") 
def model_output(config_file):
    """
    model_output function output results
    Args:
        config_file [str]: path to config file
    Returns:
        Final data
    """

    ####################
    # Load data and config file
    ####################
    config = parse_config(config_file)
    
    
    ####################
    # Output
    ####################
    model_cohort(config)
    model_life_path(config)
    model_average(config)
    
if __name__ == "__main__":
    model_output()