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
    # Load data and config
    ####################

    error_type = config["model"]["error_type"]
    
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
        exo_vars = []
        for lags in range(0, age+1):
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"full_chg_restriction_2_0")
        exo_vars.append(f"L_{age + 1}_log_restriction_2_0")
        exo_vars.append(f"L_{age + 1}_entry_rate_whole")
        exo_vars.append(f"L_{age + 1}_log_gdp")
        
        #exo_vars.append(f"log_emp_cohort")
        exo_vars.sort()
        
        wgt_var = ["firms_cohort"]
        cluster_vars = ["sector_2"]
        
        var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
        sample_data = data.loc[:, var_lst].dropna()
        exo_vars_lst = sm.add_constant(sample_data[exo_vars])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var], 
                       entity_effects=True, time_effects=True)
        
        match error_type:
            case "clustered":
                res = mod.fit(cov_type='clustered', clusters=sample_data[cluster_vars])
            case "heteroskedastic":
                res = mod.fit(cov_type='heteroskedastic')
                
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
        v_names = [f"L_{age + 1}_log_restriction_2_0", f"full_chg_restriction_2_0"]
        for v_name in v_names:
            coefs_value = res.params[v_name]
            lower_ci = res.conf_int().loc[v_name, "lower"]
            upper_ci = res.conf_int().loc[v_name, "upper"]
            dict1.update({"name": v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
            coefs_cohort.append(dict1)

    df_coefs_cohort = pd.DataFrame(coefs_cohort)
    df_coefs_age = df_coefs_age.sort_values(by=['name'])
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
        exo_vars = []
        for lags in range(0, age+1):
            exo_vars.append(f"L_{lags}_chg_restriction_2_0")
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        pre_entry_age = age + 1
        exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
        exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
        exo_vars.append(f"L_{pre_entry_age}_log_gdp")
        #exo_var_list.append(f"log_emp_cohort")
        
        exo_vars.sort()
        wgt_var = ["firms_cohort"]
        cluster_vars = ["sector_2"]
        
        var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
        sample_data = data.loc[:, var_lst].dropna()
        exo_vars_lst = sm.add_constant(sample_data[exo_vars])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                       entity_effects=True, time_effects=True)
        
        match error_type:
            case "clustered":
                res = mod.fit(cov_type='clustered', clusters=sample_data[cluster_vars])
            case "heteroskedastic":
                res = mod.fit(cov_type='heteroskedastic')

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
        v_name = f"L_{pre_entry_age}_log_restriction_2_0"
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
    coefs_age = []
    for age in range(1, 6):
        # load data
        data = df
        
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        pre_entry_age = age + 1
        data["cross"] = data[f"L_{pre_entry_age}_log_restriction_2_0"] * data["large_firm"]
        
        # create exogenous variable (list)
        exo_vars = []
        for lags in range(0, age+1):
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"full_chg_restriction_2_0")
        exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
        exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
        exo_vars.append(f"L_{pre_entry_age}_log_gdp")
        exo_vars.append("large_firm")    
        exo_vars.append("cross")
                
        #exo_var_list.append(f"log_emp_cohort")
        exo_vars.sort()
        wgt_var = ["firms_cohort"]
        cluster_vars = ["sector_2"]
        
        var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
        sample_data = data.loc[:, var_lst].dropna()
        exo_vars_lst = sm.add_constant(sample_data[exo_vars])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                       entity_effects=True, time_effects=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_cohort_age_{age}_hetero.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
            
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_names = ["cross", f"L_{pre_entry_age}_log_restriction_2_0", "large_firm"]
        for v_name in v_names:
            coefs_value = res.params[v_name]
            lower_ci = res.conf_int().loc[v_name, "lower"]
            upper_ci = res.conf_int().loc[v_name, "upper"]
            dict1.update({"name":v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
            coefs_age.append(dict1)
    
    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_cohort_age_h_LP.csv")
      

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
        pre_entry_age = age + 1
        data = data.set_index(['sector', 'year'])
        data["cross"] = data[f"L_{pre_entry_age}_log_restriction_2_0"] * data["large_firm"]
        
        # create exogenous variable (list)
        exo_vars = []
        for lags in range(0, age + 1):
            exo_vars.append(f"L_{lags}_chg_restriction_2_0")
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
        exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
        exo_vars.append(f"L_{pre_entry_age}_log_gdp")
        exo_vars.append("large_firm")    
        exo_vars.append("cross")
                
        #exo_var.append(f"log_emp_cohort")

        exo_vars.sort()
        wgt_var = ["firms_cohort"]
        cluster_vars = ["sector_2"]
        
        var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
        sample_data = data.loc[:, var_lst].dropna()
        exo_vars_lst = sm.add_constant(sample_data[exo_vars])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                       entity_effects=True, time_effects=True)
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary
        
        # saving results
        # table
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_path_age_{age}_hetero.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
            
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_names = ["cross", f"L_{pre_entry_age}_log_restriction_2_0", "large_firm"]
        for v_name in v_names:
            coefs_value = res.params[v_name]
            lower_ci = res.conf_int().loc[v_name, "lower"]
            upper_ci = res.conf_int().loc[v_name, "upper"]
            dict1.update({"name":v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
            coefs_age.append(dict1) 
            
    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_life_age_h_LP.csv")
       
            
        
def model_average(config, depend_var):
    """
    model_average function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
    Returns:
        Final data
    """    
    
    error_type = config["model"]["error_type"]
    
    # load data paths
    cleaned_data_path = Path(config["model"]["average_sec_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    
    # regression by age
    coefs_age = [] 
    
    # age = 1
    age = 1
    
    # load data
    data = df

    # sample restriction
    data = data[data.year > 1981]
    data = data[data.age_grp_dummy == age]
    
    # regression
    data = data.set_index(['sector', 'year'])
    
    # regression
    pre_entry_age = age + 1
    exo_vars = ["curr_chg_restriction_2_0", "enter_chg_restriction_2_0"]
    
    for lags in range(0, age + 1):
        exo_vars.append(f"L_{lags}_entry_rate_whole")
        exo_vars.append(f"L_{lags}_log_gdp")
    
    exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
    exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
    exo_vars.append(f"L_{pre_entry_age}_log_gdp")
    
    exo_vars.sort()
    wgt_var = ["firms_cohort"]

    cluster_vars = ["sector_2"]
    
    var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
    sample_data = data.loc[:, var_lst].dropna()
    
    exo_vars_lst = sm.add_constant(sample_data[exo_vars])
    
    # regression
    mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                entity_effects=True, time_effects=True)

    match error_type:
        case "clustered":
            res = mod.fit(cov_type='clustered', clusters=sample_data[cluster_vars])
        case "heteroskedastic":
            res = mod.fit(cov_type='heteroskedastic')

    # results
    results = res.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_age_{age}.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())

    # get input row in dictionary format
    # key = col_name
    dict1 = {}
    v_names = ["curr_chg_restriction_2_0",
            "enter_chg_restriction_2_0", f"L_{pre_entry_age}_log_restriction_2_0"]
    for v_name in v_names:
        coefs_value = res.params[v_name]
        lower_ci = res.conf_int().loc[v_name, "lower"]
        upper_ci = res.conf_int().loc[v_name, "upper"]
        dict1.update({"name":v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
        coefs_age.append(dict1)
    
    # age = 2:5
    for age in range(2, 6):
        # load data
        data = df
    
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        
        # regression
        pre_entry_age = age + 1
        exo_vars = ["curr_chg_restriction_2_0", "life_chg_restriction_2_0", "enter_chg_restriction_2_0"]
        
        for lags in range(0, age + 1):
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
        exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
        exo_vars.append(f"L_{pre_entry_age}_log_gdp")
        
        exo_vars.sort()
        wgt_var = ["firms_cohort"]

        cluster_vars = ["sector_2"]
        
        var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
        sample_data = data.loc[:, var_lst].dropna()
        
        exo_vars_lst = sm.add_constant(sample_data[exo_vars])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                       entity_effects=True, time_effects=True)
    
        match error_type:
            case "clustered":
                res = mod.fit(cov_type='clustered', clusters=sample_data[cluster_vars])
            case "heteroskedastic":
                res = mod.fit(cov_type='heteroskedastic')

        # results
        results = res.summary

        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_age_{age}.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv())
    
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_names = ["curr_chg_restriction_2_0", "life_chg_restriction_2_0",
                   "enter_chg_restriction_2_0", f"L_{pre_entry_age}_log_restriction_2_0"]
        for v_name in v_names:
            coefs_value = res.params[v_name]
            lower_ci = res.conf_int().loc[v_name, "lower"]
            upper_ci = res.conf_int().loc[v_name, "upper"]
            dict1.update({"name":v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
            coefs_age.append(dict1) 
            
    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_average_age_LP.csv")
       
       
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
    
    # regression by age
    coefs_age = [] 
    
    # age = 1
    age = 1
    # load data
    data = df

    # sample restriction
    data = data[data.year > 1981]
    data = data[data.age_grp_dummy == age]
    
    # regression
    data = data.set_index(['sector', 'year'])
    
    pre_entry_age = age + 1
    data["cross"] = data[f"L_{pre_entry_age}_log_restriction_2_0"] * data["large_firm"]
    
    exo_vars = ["curr_chg_restriction_2_0", "enter_chg_restriction_2_0"]
    
    for lags in range(0, age + 1):
        exo_vars.append(f"L_{lags}_entry_rate_whole")
        exo_vars.append(f"L_{lags}_log_gdp")
    
    exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
    exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
    exo_vars.append(f"L_{pre_entry_age}_log_gdp")
    exo_vars.append("large_firm")    
    exo_vars.append("cross")
    
    exo_vars.sort()
    wgt_var = ["firms_cohort"]
    cluster_vars = ["sector_2"]
    
    var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
    sample_data = data.loc[:, var_lst].dropna()
    exo_vars_lst = sm.add_constant(sample_data[exo_vars])
    
    # regression
    mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                    entity_effects=True, time_effects=True)

    res = mod.fit(cov_type = 'heteroskedastic')

    # results
    results = res.summary
    
    file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_age_{age}_hetero.csv"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv()) 
    
    # get input row in dictionary format
    # key = col_name
    dict1 = {}
    v_names = ["cross", f"L_{pre_entry_age}_log_restriction_2_0", "large_firm"]
    for v_name in v_names:
        coefs_value = res.params[v_name]
        lower_ci = res.conf_int().loc[v_name, "lower"]
        upper_ci = res.conf_int().loc[v_name, "upper"]
        dict1.update({"name":v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
        coefs_age.append(dict1) 
    
    # age = 2:5
    for age in range(2, 6):
        # load data
        data = df
    
        # sample restriction
        data = data[data.year > 1981]
        data = data[data.age_grp_dummy == age]
        
        # regression
        data = data.set_index(['sector', 'year'])
        
        pre_entry_age = age + 1
        data["cross"] = data[f"L_{pre_entry_age}_log_restriction_2_0"] * data["large_firm"]
        
        exo_vars = ["curr_chg_restriction_2_0", "life_chg_restriction_2_0", "enter_chg_restriction_2_0"]
        
        for lags in range(0, age + 1):
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"L_{pre_entry_age}_log_restriction_2_0")
        exo_vars.append(f"L_{pre_entry_age}_entry_rate_whole")
        exo_vars.append(f"L_{pre_entry_age}_log_gdp")
        exo_vars.append("large_firm")    
        exo_vars.append("cross")
        
        exo_vars.sort()
        wgt_var = ["firms_cohort"]
        cluster_vars = ["sector_2"]
        
        var_lst = [depend_var] + exo_vars + wgt_var + cluster_vars
        sample_data = data.loc[:, var_lst].dropna()
        exo_vars_lst = sm.add_constant(sample_data[exo_vars])
        
        # regression
        mod = PanelOLS(sample_data[depend_var], exo_vars_lst, weights=sample_data[wgt_var],
                       entity_effects=True, time_effects=True)
    
        res = mod.fit(cov_type = 'heteroskedastic')

        # results
        results = res.summary
        
        file_path = Path.cwd()/results_tables_path/f"{depend_var}_results_average_age_{age}_hetero.csv"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results.as_csv()) 
        
        # get input row in dictionary format
        # key = col_name
        dict1 = {}
        v_names = ["cross", f"L_{pre_entry_age}_log_restriction_2_0", "large_firm"]
        for v_name in v_names:
            coefs_value = res.params[v_name]
            lower_ci = res.conf_int().loc[v_name, "lower"]
            upper_ci = res.conf_int().loc[v_name, "upper"]
            dict1.update({"name":v_name, "age": age, "Coef": coefs_value, "lower_ci": lower_ci, "upper_ci": upper_ci}) 
            coefs_age.append(dict1) 
            
    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{depend_var}_results_life_age_h_LP.csv")


def panel_reg(config, depend_var):
    """
    cross_section function load the clean data and run the regression
    to study the effects of regulation on firm exit rates
    Args:
        config [str]: config file
        depend_var [str]: dependent variable
    Returns:
        Final results
    """
    ####################
    # Load data
    ####################
    error_type = config["model"]["error_type"]
    
    # load data paths
    cleaned_data_path = Path(config["model"]["life_path_sec_ag_path"])
    results_tables_path = Path(config["model"]["results_tables_path"])

    df = pd.read_csv(Path.cwd()/cleaned_data_path) 
    
    ####################
    # Regression including controls at entry by each age
    ####################
    # load data
    data = df
    
    # sample restriction
    data = data[data.year > 1981]
    
    # regression
    data = data.set_index(['sector', 'year'])

    # regression
    data1 = data.loc[:, ["L_0_log_restriction_2_0", "L_0_entry_rate_whole", "L_0_log_gdp",
                    "avg_log_restriction_2_0", "avg_entry_rate_whole", "avg_log_gdp", 'sector_2', "log_emp_cohort", 
                    depend_var, 'age_grp_dummy', 'firms']].dropna()
    mod1 = PanelOLS.from_formula(formula = f'{depend_var} ~ L_0_log_restriction_2_0 + L_0_entry_rate_whole + L_0_log_gdp \
                                            + C(age_grp_dummy) + EntityEffects + TimeEffects',
                                            weights=data1['firms'], data = data1, drop_absorbed=True)
    match error_type:
        case "clustered":
            res1 = mod1.fit(cov_type='clustered', clusters=data1.sector_2)
        case "heteroskedastic":
            res1 = mod1.fit(cov_type='heteroskedastic')

        # results
    results = res1.summary

    file_path = Path.cwd()/results_tables_path/f"{depend_var}_panel_reg.csv"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results.as_csv())
        
        
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
    error_type = config["model"]["error_type"]
    
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
        data = data.set_index(['sector', 'pre_cohort'])
        X_array = ['entry_rate_whole', 'entry_rate_whole_pre_cohort', 'age_grp_dummy', 'year', 'log_gdp', 'log_gdp_pre_cohort',
                            'chg_restriction_2_0_{naics_curr}', f'log_restriction_2_{naics_curr}_pre_cohort',
                            'firms', 'sector_2', "log_emp_pre_cohort"]
        y_array = [depend_var]
        all_array = y_array + X_array
        
        data = data.loc[:, all_array].dropna()
        mod = PanelOLS.from_formula(formula = f'{depend_var} ~ "chg_restriction_2_0_{naics_curr}" + log_restriction_2_{naics_curr}_pre_cohort \
                                                + entry_rate_whole + entry_rate_whole_pre_cohort + log_emp_pre_cohort \
                                                + C(age_grp_dummy) + EntityEffects + TimeEffects',
                                                weights=data['firms'], data = data, drop_absorbed=True)
        match error_type:
            case "clustered":
                res = mod.fit(cov_type='clustered', clusters=data.sector_2)
            case "heteroskedastic":
                res = mod.fit(cov_type='heteroskedastic')

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
    variable_list = config["model"]["dep_var"]
    
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
        
        print(f"  - panel regression")
        panel_reg(config, variable)
        
        print(f"  - robust")
        model_cohort_robust(config, variable)
    
if __name__ == "__main__":
    model_output()