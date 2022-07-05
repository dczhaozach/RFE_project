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

from Src.utility import parse_config

# options of pandas
pd.options.mode.use_inf_as_na = True
            
            
def model_life_path(config, depend_var):
    """
    model_life_path function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
        depend_var [str]: dependent variable
    Returns:
        Final data
    """
    ####################
    # Load data
    ####################

    # load data paths
    cleaned_data_path = Path(config["model"]["life_path_sec_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path) 
    
    ####################
    # Regression including controls at entry by each age
    ####################
    coefs_cohort = []
    
    for age in range(1, 6):
        # load data
        data = df
        
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        
        # create exogenous variable (list)
        exo_var_list = []
        exo_var_list.append(f"L_0_log_restriction_2_0")
        exo_var_list.append(f"L_0_entry_rate_whole")
        exo_var_list.append(f"L_0_log_gdp")
        exo_var_list.append(f"L_{age}_log_restriction_2_0")
        exo_var_list.append(f"L_{age}_entry_rate_whole")
        exo_var_list.append(f"L_{age}_log_gdp")
        exo_var_list.append(f"log_emp_cohort")
        
        exo_var_list.sort()
        sample_data = data.loc[:, [depend_var, 'sector_2'] + exo_var_list].dropna()
        exo_vars = sm.add_constant(sample_data[exo_var_list])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type='clustered', clusters=sample_data.sector_2)

        # results
        results = res.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_cohort_age_{age}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())  
            
        # coefs
        
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_name = f"L_{age}_log_restriction_2_0"
        coefs_value = res.params[v_name]
        lower_ci = res.conf_int().loc[v_name, "lower"]
        upper_ci = res.conf_int().loc[v_name, "upper"]
        dict1.update({"age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
        coefs_cohort.append(dict1)

    df_coefs_cohort = pd.DataFrame(coefs_cohort)
    df_coefs_cohort.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_cohort_age_LP.csv") 

    ####################
    # Regression including all controls in life path by each age
    ####################
    
    coefs_age = []
    for age in range(1, 6):
        # load data
        data = df
        
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        
        # create exogenous variable (list)
        exo_var_list = []
        for lags in range(0, age + 1):
            exo_var_list.append(f"L_{lags}_log_restriction_2_0")
            exo_var_list.append(f"L_{lags}_entry_rate_whole")
            exo_var_list.append(f"L_{lags}_log_gdp")
        
        exo_var_list.append(f"log_emp_cohort")
        
        exo_var_list.sort()
        sample_data = data.loc[:, [depend_var, 'sector_2'] + exo_var_list].dropna()
        exo_vars = sm.add_constant(sample_data[exo_var_list])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type='clustered', clusters=sample_data['sector_2'])

        # results
        results = res.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_path_age_{age}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())

        # coefs
        
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_name = f"L_{age}_log_restriction_2_0"
        coefs_value = res.params[v_name]
        lower_ci = res.conf_int().loc[v_name, "lower"]
        upper_ci = res.conf_int().loc[v_name, "upper"]
        dict1.update({"age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
        coefs_age.append(dict1)

    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_path_age_LP.csv")
    
    
def model_life_path_hetero(config, depend_var):
    """
    model_life_path function load the clean data and run the regression
    to study the heterogeneous effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """
    ####################
    # Load data
    ####################

    # load data paths
    cleaned_data_path = Path(config["model"]["life_path_sec_sz_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path) 
    
    ####################
    # Regression including controls at entry by each age
    ####################
    
    for age in range(1, 6):
        # load data
        data = df
        
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        data["cross"] = data[f"L_{age}_log_restriction_2_0"] * data["large_firm"]
        
        # create exogenous variable (list)
        exo_var_list = []
        exo_var_list.append(f"L_0_log_restriction_2_0")
        exo_var_list.append(f"L_0_entry_rate_whole")
        exo_var_list.append(f"L_0_log_gdp")
        exo_var_list.append(f"L_{age}_log_restriction_2_0")
        exo_var_list.append(f"L_{age}_entry_rate_whole")
        exo_var_list.append(f"L_{age}_log_gdp")
        
        exo_var_list.append(f"log_emp_cohort")
        exo_var_list.append("large_firm")    
        exo_var_list.append("cross")
        exo_var_list.sort()
        sample_data = data.loc[:, [depend_var] + exo_var_list].dropna()
        exo_vars = sm.add_constant(sample_data[exo_var_list])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_cohort_age_{age}_hetero.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())  

    ####################
    # Regression including all controls in life path by each age
    ####################
        
    for age in range(1, 6):
        # load data
        data = df
        
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        data["cross"] = data[f"L_{age}_log_restriction_2_0"] * data["large_firm"]
        # create exogenous variable (list)
        exo_var_list = []
        for lags in range(0, age + 1):
            exo_var_list.append(f"L_{lags}_log_restriction_2_0")
            exo_var_list.append(f"L_{lags}_entry_rate_whole")
            exo_var_list.append(f"L_{lags}_log_gdp")
        
        exo_var_list.append(f"log_emp_cohort")
        exo_var_list.append("large_firm")    
        exo_var_list.append("cross")
        exo_var_list.sort()
        sample_data = data.loc[:, [depend_var] + exo_var_list].dropna()
        exo_vars = sm.add_constant(sample_data[exo_var_list])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_path_age_{age}_hetero.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
        
            
        
def model_average(config, depend_var):
    """
    model_average function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """    
    # load data paths
    cleaned_data_path = Path(config["model"]["average_sec_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    
    # load data
    data = df
    
    # sample restriction
    data = data[data.year > 1981]
    
    # regression
    data = data.set_index(['sector', 'year'])

    # regression
    data1 = data.loc[:, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "avg_log_restriction_2_0", "avg_entry_rate_whole", "avg_log_gdp", 'sector_2', "log_emp_cohort", 
                    depend_var, 'age_grp_dummy']].dropna()
    mod1 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                            + avg_log_restriction_2_0 + avg_entry_rate_whole + avg_log_gdp + log_emp_cohort \
                                            + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data1, drop_absorbed=True)
    res1 = mod1.fit(cov_type='clustered', clusters=data1["sector_2"])

    # results
    results = res1.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_all.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())
        
    # regression
    data2 = data.loc[:, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "inc_avg_log_restriction_2_0", "cohort_log_restriction_2_0",
                    "inc_avg_entry_rate_whole", "cohort_entry_rate_whole", 'sector_2', "log_emp_cohort",
                    "inc_avg_log_gdp", "cohort_log_gdp",
                    depend_var, 'age_grp_dummy']].dropna()
    mod2 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                            + cohort_log_restriction_2_0 + inc_avg_log_restriction_2_0 \
                                            + inc_avg_entry_rate_whole + cohort_entry_rate_whole \
                                            + inc_avg_log_gdp + cohort_log_gdp + log_emp_cohort \
                                            + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data2, drop_absorbed=True)
    res2 = mod2.fit(cov_type='clustered', clusters=data2["sector_2"])

    # results
    results = res2.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_cohort_inc.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())
        
        
        
def model_average_hetero(config, depend_var):
    """
    model_average function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """    
    # load data paths
    cleaned_data_path = Path(config["model"]["average_sec_sz_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    
    # load data
    data = df
    
    # sample restriction
    data = data[data.year > 1981]
    
    # regression
    data = data.set_index(['sector', 'year'])
    
    # regression
    data = data.loc[:, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "avg_log_restriction_2_0", "avg_entry_rate_whole", "avg_log_gdp", "log_emp_cohort",
                    depend_var, 'age_grp_dummy', "large_firm"]].dropna()
    mod = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0  \
        + L_0_entry_rate_whole + L_0_log_gdp + avg_entry_rate_whole + avg_log_gdp \
        + avg_log_restriction_2_0 + large_firm + large_firm * avg_log_restriction_2_0 + log_emp_cohort \
        + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data, drop_absorbed=True)
    res = mod.fit(cov_type = 'heteroskedastic')

    # results
    results = res.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_all_hetero.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())  


def model_average_age(config, depend_var):
    """
    model_average function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """    
    # load data paths
    cleaned_data_path = Path(config["model"]["average_sec_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    
    # load data
    data = df
    
    # sample restriction
    data = data[data.year > 1981]
    
    # regression
    data = data.set_index(['sector', 'year'])
    
    # age = 1
    # regression
    coefs_age = []
    
    age = 1
    data1 = data.loc[data.age_grp_dummy == age, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "avg_log_restriction_2_0", "avg_entry_rate_whole", "avg_log_gdp", "sector_2", "log_emp_cohort",
                    depend_var]].dropna()
    
    mod1 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                            + avg_log_restriction_2_0 + avg_entry_rate_whole + avg_log_gdp + log_emp_cohort \
                                            + EntityEffects + TimeEffects', data = data1, drop_absorbed=True)
    res1 = mod1.fit(cov_type='clustered', clusters=data1["sector_2"])

    # results
    results = res1.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_all_age_{age}.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())
        
    # regression
    data2 = data.loc[data.age_grp_dummy == age, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "cohort_log_restriction_2_0",
                    "cohort_entry_rate_whole",
                    "cohort_log_gdp", "sector_2", "log_emp_cohort",
                    depend_var]].dropna()
    mod2 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                            + cohort_log_restriction_2_0 \
                                            + cohort_entry_rate_whole \
                                            + cohort_log_gdp \
                                            + log_emp_cohort \
                                            + EntityEffects + TimeEffects', data = data2, drop_absorbed=True)
    res2 = mod2.fit(cov_type='clustered', clusters=data2["sector_2"])

    # results
    results = res2.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_cohort_inc_age_{age}.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())
        
        
    # coefs
        
    # get input row in dictionary format
    # key = col_name
    dict1 = {}
    
    v_name = f"cohort_log_restriction_2_0"
    coefs_value = res2.params[v_name]
    lower_ci = res2.conf_int().loc[v_name, "lower"]
    upper_ci = res2.conf_int().loc[v_name, "upper"]
    dict1.update({"age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
    coefs_age.append(dict1)
    

    coefs_age_2 = []
    for age in range(2, 6):
        # regression
        data1 = data.loc[data.age_grp_dummy == age, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                        "avg_log_restriction_2_0", "avg_entry_rate_whole", "avg_log_gdp", "sector_2", "log_emp_cohort",
                        depend_var]].dropna()
        
        mod1 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                                + avg_log_restriction_2_0 + avg_entry_rate_whole + avg_log_gdp + log_emp_cohort\
                                                + EntityEffects + TimeEffects', data = data1, drop_absorbed=True)
        res1 = mod1.fit(cov_type='clustered', clusters=data1["sector_2"])

        # results
        results = res1.summary

        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_all_age_{age}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
            
        # regression
        data2 = data.loc[data.age_grp_dummy == age, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                        "inc_avg_log_restriction_2_0", "cohort_log_restriction_2_0",
                        "inc_avg_entry_rate_whole", "cohort_entry_rate_whole",
                        "inc_avg_log_gdp", "cohort_log_gdp", "sector_2", "log_emp_cohort",
                        depend_var]].dropna()
        mod2 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                                + cohort_log_restriction_2_0 + inc_avg_log_restriction_2_0 \
                                                + inc_avg_entry_rate_whole + cohort_entry_rate_whole \
                                                + inc_avg_log_gdp + cohort_log_gdp + log_emp_cohort \
                                                + EntityEffects + TimeEffects', data = data2, drop_absorbed=True)
        res2 = mod2.fit(cov_type='clustered', clusters=data2["sector_2"])

        # results
        results = res2.summary

        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_cohort_inc_age_{age}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
        
            # coefs
        
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_name = f"cohort_log_restriction_2_0"
        coefs_value = res2.params[v_name]
        lower_ci = res2.conf_int().loc[v_name, "lower"]
        upper_ci = res2.conf_int().loc[v_name, "upper"]
        dict1.update({"age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
        coefs_age.append(dict1)
        
        # get input row in dictionary format
        # key = col_name
        dict2 = {}
        v_name = f"inc_avg_log_restriction_2_0"
        coefs_value = res2.params[v_name]
        lower_ci = res2.conf_int().loc[v_name, "lower"]
        upper_ci = res2.conf_int().loc[v_name, "upper"]
        dict2.update({"age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
        coefs_age_2.append(dict2)
        
    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_average_cohort_inc_age_LP.csv")
        
    df_coefs_age_2 = pd.DataFrame(coefs_age_2)
    df_coefs_age_2.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_average_cohort_inc_age_LP_2.csv")
        

def model_cohort_robust(config, depend_var):
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
    cleaned_data_path = Path(config["model"]["cohort_robust_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)

    ####################
    # Regression
    ####################

    for naics_curr in range(2, 5):
        # load data
        data = df
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy <= 5]
        
        # regression
        data = data.set_index(['sector', 'cohort'])
        X_array = ['entry_rate_whole', 'entry_rate_whole_cohort', 'age_grp_dummy', 'year', 'log_gdp', 'log_gdp_cohort',
                            f'log_restriction_2_{naics_curr}', f'log_restriction_2_{naics_curr}_cohort', 'sector_2', "log_emp_cohort"]
        y_array = [depend_var]
        all_array = y_array + X_array
        
        data = data.loc[:, all_array].dropna()
        mod = PanelOLS.from_formula(formula = f'{depend_var} ~ log_restriction_2_{naics_curr} + log_restriction_2_{naics_curr}_cohort \
                                                + entry_rate_whole + entry_rate_whole_cohort + log_emp_cohort \
                                                + C(age_grp_dummy) + EntityEffects + TimeEffects', data = data, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', clusters= data["sector_2"])

        # results
        results = res.summary

        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_reg_naics_{naics_curr}.csv"
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
    print("loading config file")
    config = parse_config(config_file)
    variable_list = config["make_data"]["dep_var"]
    
    ####################
    # Output
    ####################
    print("running models for each variables")
    for variable in variable_list:
        print(f"- for {variable}:")
        print(f"  - life path")
        model_life_path(config, variable)
        
        print(f"  - life path hetero")
        model_life_path_hetero(config, variable)
        
        print(f"  - average")
        model_average(config, variable)
        
        print(f"  - average hetero")
        model_average_hetero(config, variable)
        
        print(f"  - average by age")
        model_average_age(config, variable)
        
        print(f"  - robust")
        model_cohort_robust(config, variable)
    
if __name__ == "__main__":
    model_output()