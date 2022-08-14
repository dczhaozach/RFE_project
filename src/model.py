"""
This script build the regression models
"""

import click
import pandas as pd
from pathlib import Path
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

from Src.utility import parse_config
from Src.utility import coef_dict
from Src.utility import plot_lp

# options of pandas
pd.options.mode.use_inf_as_na = True
            
            
def model_sector(config, depend_vars):
    """
    model_sector function load the clean data and run the regression
    to study the effects of regulation on firm dynamism
    Args:
        config [str]: config file
        depend_vars [str]: dependent variables
    Returns:
        Final data
    """
    ####################
    # Load data and config
    ####################
    
    # load data paths
    cleaned_data_path = Path(config["model"]["sector_panel"])
    results_tables_path = Path(config["model"]["results_tables_path"])
    
    df = pd.read_hdf(Path.cwd()/cleaned_data_path, key="data")
    
    ####################
    # OLS
    ####################
    dict1 = {}
    dict1["index"] = [
        "OLS Coef", "", "# obs", "",
        "OLS IV", "",  "# obs", ""
        ]
    for depend_var in depend_vars:

        # load data
        data = df

        # sample restriction
        data = data[(data.year > 1985)&(data.year < 2020)]

        # regression
        data_ols = data.set_index(['sector', 'year'])
        
        var_lst = [
        "L_0_log_restriction_2_0",
        "L_0_log_gdp", "L_1_log_gdp",
        depend_var, "sector_2", "firms"
        ]
        
        formula_txt = f'{depend_var} ~ L_0_log_gdp + L_1_log_gdp + L_0_log_restriction_2_0 + EntityEffects + TimeEffects'
        
        # regression
        data_ols = data_ols.loc[:, var_lst].dropna()
        mod_ols = PanelOLS.from_formula(formula = formula_txt, weights=data_ols['firms'], data = data_ols, drop_absorbed=True)

        res_ols = mod_ols.fit(cov_type='heteroskedastic')
        results_ols = res_ols.summary
            
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_sector_panel_ols.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results_ols.as_csv())  

        ####################
        # PANEL
        ####################
        # load data
        data = df

        # sample restriction
        data = data[(data.year > 1985)&(data.year < 2020)]
        
        data_iv = data
        
        var_list = [
        "L_0_log_restriction_2_0", "L_0_bartik_iv",
        "L_0_log_gdp", "L_1_log_gdp",
        depend_var, "sector_2", 'firms', "sector", "year"]
        
        formula_txt = f'{depend_var} ~ C(sector) + C(year) + L_0_log_gdp + L_1_log_gdp + [L_0_log_restriction_2_0 ~ L_0_bartik_iv]'
        
        # regression
        data_iv = data_iv.loc[:, var_list].dropna()
        mod_iv = IV2SLS.from_formula(formula = formula_txt, weights=data_iv['firms'], data = data_iv)

        res_iv = mod_iv.fit(cov_type='heteroskedastic')
        results_iv = res_iv.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_sector_panel_iv.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results_iv.as_csv())  
        
        
        # create result table
        v_name = "L_0_log_restriction_2_0"
        
        # ols
        coefs_value_ols = res_ols.params[v_name]
        coefs_value_ols = str(round(coefs_value_ols, 5))
        std_ols = res_ols.std_errors[v_name]
        std_ols = "[" + str(round(std_ols, 5)) + "]"
        p_value_ols = res_ols.pvalues[v_name]
        stars_ols = int(p_value_ols < 0.01)*"*" + int(p_value_ols < 0.05)*"*" + int(p_value_ols < 0.1)*"*"
        coefs_value_ols = coefs_value_ols + stars_ols
        nobs_ols = res_ols.nobs
        
        # iv 
        coefs_value_iv = res_iv.params[v_name]
        coefs_value_iv = str(round(coefs_value_iv, 5))
        std_iv = res_iv.std_errors[v_name]
        std_iv = "[" + str(round(std_iv, 5)) + "]"
        
        p_value_iv = res_iv.pvalues[v_name]
        stars_iv = int(p_value_iv < 0.01)*"*" + int(p_value_iv < 0.05)*"*" + int(p_value_iv < 0.1)*"*"
        coefs_value_iv = coefs_value_iv + stars_iv
        nobs_iv = res_iv.nobs
           
        dict1[f"{depend_var}"] =  [
            coefs_value_ols, std_ols, nobs_ols, "",
            coefs_value_iv, std_iv, nobs_iv, "",
            ]
        
    df_coefs = pd.DataFrame(dict1)
    df_coefs.to_csv(Path.cwd()/results_tables_path/"key_results"/"sector_panel_summary.csv") 

    return None

    
def model_sector_age(config, depend_vars):
    """
    model_sector_age function load the clean data and run the regression
    to study the effects of regulation on firm dynamism
    Args:
        config [str]: config file
        depend_vars [str]: dependent variables
    Returns:
        Final data
    """
    ####################
    # Load data and config
    ####################
    
    # load data paths
    cleaned_data_path = Path(config["model"]["sector_age_panel"])
    results_figs_path = Path(config["model"]["results_figs_path"])
    fig_path = Path.cwd()/results_figs_path
    
    df_ag = pd.read_hdf(Path.cwd()/cleaned_data_path, key="data")
    std_reg = df_ag["L_0_log_restriction_2_0"].std()
    
    # iv estimation
    ceof_dict = []
    for depend_var in depend_vars:
        for age in df_ag.age_coarse.unique()[1:]:
            
            # load data
            data = df_ag[df_ag.age_coarse == age]

            # sample restriction
            data = data[(data.year > 1985) & (data.year < 2020)]

            var_lst = [
            "L_0_log_restriction_2_0", "L_0_bartik_iv", 
            "L_0_entry_rate",
            "L_0_log_gdp",
            "sector", "year",
            depend_var, "sector_2", 'firms'
            ]
            
            formula_txt = f'{depend_var} ~ C(year) + C(sector) \
                                        + L_0_entry_rate  \
                                        + L_0_log_gdp     \
                                        + [L_0_log_restriction_2_0 ~ L_0_bartik_iv]'
                
            # regression
            data1 = data.loc[:, var_lst].dropna()
            mod1 = IV2SLS.from_formula(
            formula = formula_txt, weights=data1['firms'], data = data1
            )

            res = mod1.fit(cov_type='heteroskedastic')
            
            ceof_dict = coef_dict(depend_var, "L_0_log_restriction_2_0", res, ceof_dict, age)

    df_coefs_age = pd.DataFrame(ceof_dict)
    df_coefs_age = df_coefs_age.sort_values(by=['depend_var', 'age'])

    for depend_var in depend_vars:
        plot_lp(df_coefs_age, depend_var, "Sector_Age_Panel", fig_path, std_reg)
        
        
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
    print("loading config file")
    config = parse_config(config_file)
    variable_list = config["model"]["dep_var"]
    variable_list.sort()
    
    ####################
    # Output
    ####################
    print("running models for sector panel")

    model_sector(config, variable_list)

    print("running models sector age panel")
    variable_list.remove("L_0_entry_rate")
    model_sector_age(config, variable_list)    
    
    
if __name__ == "__main__":
    model_output()