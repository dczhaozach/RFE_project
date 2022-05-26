"""
This script build the regression models
"""

import warnings

import logging
import click
import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels.panel import PanelOLS

from src.utility import parse_config

# options of pandas
pd.options.mode.use_inf_as_na = True


@click.command()
@click.argument("config_file", type=str, default="src/config.yaml") 
def model_cohort(config_file):
    """
    model_cohort function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config_file [str]: path to config file
    Returns:
        Final data
    """

    ####################
    # Load data and config_file
    ####################

    config = parse_config(config_file)

    # load data paths
    cleaned_data_path = Path(config["model"]["cleaned_data_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)

    ####################
    # Regression
    ####################
    
    results_dict_1 = dict()

    for naics_curr in range(2, 5):
        # load data
        data = df
        # sample restriction
        data = data[data.year > 1985]
        data = data[data.age_grp_dummy <= 5]
        
        # regression
        data = data.set_index(['sector', 'cohort'])
        data = data.loc[:, ['entry_rate_whole', 'death_rate_age', 'age_grp_dummy', 'year', 'log_incumbents_whole', 'log_gdp', 'log_gdp_cohort',
                            f'log_restriction_2_{naics_curr}', f'log_restriction_2_{naics_curr}_cohort']].dropna()
        mod = PanelOLS.from_formula(formula = f'death_rate_age ~ log_restriction_2_{naics_curr} + log_restriction_2_{naics_curr}_cohort \
                                                + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data, drop_absorbed=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary

        file_path = Path.cwd()/results_tables_path/f"results_reg_naics_{naics_curr}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())

if __name__ == "__main__":
    model_cohort()