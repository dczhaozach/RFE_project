"""
This script load and clean the raw data to prepare for regression that study
the regulation at the time of entering on firm exit rates
"""

import warnings

from pathlib import Path
import logging
import click
import pandas as pd
import numpy as np

from src.utility import parse_config

# options of pandas
pd.options.mode.use_inf_as_na = True


def data_load(config):
    """
    data_load function that
        - load raw Regdata and BDS data
        - create Entry at sector level
        - create simple measurement (firm exit)

    Args:
        config [str]: config file
    Returns:
        BDS data, Reg data, GDP data, Entry Measures

    """

    ####################
    # Load and clean data
    ####################

    # load data paths
    data_file_path = Path(config["make_data"]["data_file_path"])
    regdata_path = Path(config["make_data"]["regdata_path"])
    bds_naics_4_path = Path(config["make_data"]["bds_naics_4_path"])
    gdp_path = Path(config["make_data"]["gdp_path"])
    bds_sector_size_path = Path(config["make_data"]["bds_sector_size"])

    # reg data
    regdata = pd.read_csv(data_file_path/regdata_path)
    regdata["sector_reg"] = regdata["NAICS"]
    regdata = regdata.loc[:, ["year", "sector_reg", 
                              "industry_restrictions_1_0", "industry_restrictions_2_0"]]

    # sector gdp data
    gdp = pd.read_csv(data_file_path/gdp_path)
    gdp = pd.melt(gdp, id_vars="sector_2", var_name="year", value_name="gdp")
    gdp["year"] = pd.to_numeric(gdp["year"]).astype(np.int64)

    # load BDS dataset by age sector
    df_sec_ag = pd.read_csv(data_file_path/bds_naics_4_path)

    # load BDS dataset by age sector size
    df_sec_sz_ag = pd.read_csv(data_file_path/bds_sector_size_path)
    
    # create size dummy
    df_sec_sz_ag["large_firm"] = 0
    index = (
        (df_sec_sz_ag["fsize"] == "f) 500 to 999")
        | (df_sec_sz_ag["fsize"] == "g) 1000 to 2499")
        | (df_sec_sz_ag["fsize"] == "h) 2500 to 4999")
        | (df_sec_sz_ag["fsize"] == "i) 5000 to 9999")
        | (df_sec_sz_ag["fsize"] == "j) 10000+")
    )
    df_sec_sz_ag.loc[index, "large_firm"] = 1
    
    ####################
    # Create Entry and Exit by Sectors
    ####################
    
    # change variable types
    df_sec_ag["firms"] = pd.to_numeric(df_sec_ag["firms"], errors="coerce", downcast=None)
    df_sec_ag["death"] = pd.to_numeric(df_sec_ag["firmdeath_firms"], errors="coerce", downcast=None)
    df_sec_ag["death_rate"] = df_sec_ag["death"]/df_sec_ag["firms"]
    df_sec_ag["sector"] = df_sec_ag["sector"].astype(str).str.slice(0,2)
    df_sec_ag["sector"] = pd.to_numeric(df_sec_ag["sector"])
    df_sec_ag["year"] = pd.to_numeric(df_sec_ag["year"])
    
    # change variable types
    df_sec_sz_ag["firms"] = pd.to_numeric(df_sec_sz_ag["firms"], errors="coerce", downcast=None)
    df_sec_sz_ag["death"] = pd.to_numeric(df_sec_sz_ag["firmdeath_firms"], errors="coerce", downcast=None)
    df_sec_sz_ag["death_rate"] = df_sec_sz_ag["death"]/df_sec_sz_ag["firms"]
    df_sec_sz_ag["sector"] = df_sec_sz_ag["sector"].astype(str).str.slice(0,2)
    df_sec_sz_ag["sector"] = pd.to_numeric(df_sec_sz_ag["sector"])
    df_sec_sz_ag["year"] = pd.to_numeric(df_sec_sz_ag["year"])
    
    # aggregate entry by sector
    # create aggregation for entry and death measure

    df_age = pd.DataFrame()
    df_age["entry_whole"] = \
        df_sec_ag.loc[df_sec_ag["fage"] == "a) 0", :].groupby(["year", "sector"])["firms"].sum()
    df_age["incumbents_whole"] = \
        df_sec_ag.loc[df_sec_ag["fage"] != "a) 0", :].groupby(["year", "sector"])["firms"].sum()
    df_age = df_age.reset_index()
    # df_age["death"] = \
    #   df_sec_ag.loc[df_sec_ag["fage"] != "a) 0", :].groupby(["year", "sector"])["firmdeath_firms"].sum()

    return df_sec_sz_ag, df_sec_ag, regdata, gdp, df_age


def data_cohort_robust(config):
    """
    data_cohort function
        - merge regulation, gdp, entry at the time of entering for each cohort and observed years
        - create log measurements
        
    Args:
        config [str]: config file
    Returns:
        Merged cohort data
    """
    
    ####################
    # Load data
    ####################
    df, regdata, gdp, df_age= data_load(config)[1:5]

    ####################
    # Merge Datasets at observed years
    ####################

    # merge df and regulation levels at different industry levels
    for naics in range(2, 5):
        # clean and create variables
        sector_name = f"sector_{naics}"
        df[sector_name] = df["sector"].astype(str).str.slice(0,naics)
        df[sector_name] = pd.to_numeric(df[sector_name])

        df = df.merge(regdata, how = "left", left_on=["year", sector_name], right_on=["year", "sector_reg"])
        df = df.rename(columns={"industry_restrictions_1_0": f"industry_restrictions_1_0_{naics}", "industry_restrictions_2_0": f"industry_restrictions_2_0_{naics}"}, errors="raise")
        df = df.drop(columns="sector_reg")
        df[f"log_restriction_1_{naics}"] = np.log(df[f"industry_restrictions_1_0_{naics}"])
        df[f"log_restriction_2_{naics}"] = np.log(df[f"industry_restrictions_2_0_{naics}"])

    # merge with sector gdp
    df = df.merge(gdp, how = "left", left_on=["year", "sector_2"], right_on=["year", "sector_2"])
    
    # merge with Sector Entry and
    df = df.merge(df_age, how = "left", left_on=["year", "sector"], right_on=["year", "sector"])
    
    ####################
    # Define coarse age groups and cohort year variables
    ####################

    # define age groups
    df = df.sort_values(by=["year", "sector", "fage"])
    i = 0
    for fage in df.fage.unique():
        df.loc[df.fage == fage, "age_grp_dummy"] = i
        i = i + 1

    df.loc[df.age_grp_dummy >= 7, "age_grp_dummy"] = 7
    
    # create cohort year data
    df.loc[df.age_grp_dummy <= 5, "cohort"] = \
        df.loc[df.age_grp_dummy <= 5, "year"] - df.loc[df.age_grp_dummy <= 5, "age_grp_dummy"]

    ####################
    # Merge with cohort year variables
    ####################

    # merge reg data at entry year
    regdata = regdata.rename(columns={"year": "cohort"})

    for naics in range(2, 5):
        # clean and create variables
        sector_name = f"sector_{naics}"

        df = df.merge(regdata, how = "left", left_on=["cohort", sector_name], right_on=["cohort", "sector_reg"])
        df = df.rename(columns={"industry_restrictions_1_0": f"industry_restrictions_1_0_{naics}_cohort", 
                                "industry_restrictions_2_0": f"industry_restrictions_2_0_{naics}_cohort"}, 
                                errors="raise")
                                
        df = df.drop(columns="sector_reg")
        
        # create log variables
        df[f"log_restriction_1_{naics}_cohort"] = np.log(df[f"industry_restrictions_1_0_{naics}_cohort"])
        df[f"log_restriction_2_{naics}_cohort"] = np.log(df[f"industry_restrictions_2_0_{naics}_cohort"])

    gdp = gdp.rename(columns={"year": "cohort", 
                              "gdp": "gdp_cohort"})
    gdp["cohort"] = pd.to_numeric(gdp["cohort"]).astype(np.int64)

    df = df.merge(gdp, how = "left", left_on=["cohort", "sector_2"], right_on=["cohort", "sector_2"])

    # merge the entry rate at the cohort level
    df_age_new = df_age.rename(columns={"year": "cohort", 
                                        "entry_whole": "entry_whole_cohort",
                                        "incumbents_whole": "incumbents_whole_cohort"})
    df_age_new = df_age_new[["cohort", "sector", "entry_whole_cohort", "incumbents_whole_cohort"]]
    df = df.merge(df_age_new, 
        how = "left", left_on=["cohort", "sector"], right_on=["cohort", "sector"])

    # create some variables
    with warnings.catch_warnings(): # suppress log zero warnings
        warnings.simplefilter("ignore")
        df["log_gdp"] = np.log(df["gdp"])
        df["log_gdp_cohort"] = np.log(df["gdp_cohort"])

        df["entry_rate_whole"] = df["entry_whole"]/df["incumbents_whole"]  
        df["entry_rate_whole_cohort"] = df["entry_whole_cohort"]/df["incumbents_whole_cohort"]       

    return df


def data_life_path(df_raw, config):
    """
    data_life_path function 
        - merges regulation, gdp, entry on each cohort life path
        - create log and flow measure measurements on each cohort life path
        
    Args:
        df_raw [DataFrame]: Raw BDS data
        config [str]: config file
    Returns:
        Merged cohort data
    """
    ####################
    # Load raw and merged data
    ####################
    df = df_raw
    regdata, gdp, df_age = data_load(config)[2:5]

    ####################
    # Define coarse age groups and cohort year variables
    ####################

    # define age groups
    df = df.sort_values(by=["year", "sector", "fage"])
    i = 0
    for fage in df.fage.unique():
        df.loc[df.fage == fage, "age_grp_dummy"] = i
        i = i + 1

    df.loc[df.age_grp_dummy >= 7, "age_grp_dummy"] = 7
    
    ####################
    # Merge variables at life path
    ####################
    
    # rename the variables to merge (prevent from conflicts and harmonize data types)
    regdata = regdata.rename(columns={"year": "year_reg"})
    gdp = gdp.rename(columns={"year": "year_gdp"})
    gdp["year_gdp"] = pd.to_numeric(gdp["year_gdp"]).astype(np.int64)
    df_age_new = df_age.rename(columns={"year": "year_entry"})
    df_age_new = df_age_new[["year_entry", "sector", "entry_whole", "incumbents_whole"]]
     
    naics = 2
    sector_name = f"sector_{naics}"
    df[sector_name] = df["sector"].astype(str).str.slice(0,naics)
    df[sector_name] = pd.to_numeric(df[sector_name])
    
    # merge by ages
    for age in range(0, 6):
        
        # create lag year data
        df.loc[df.age_grp_dummy <= 5, f"L_{age}_year"] = \
            df.loc[df.age_grp_dummy <= 5, "year"] - age

        ####################
        # Merge with cohort year variables
        ####################

        df = df.merge(regdata, how = "left", left_on=[f"L_{age}_year", sector_name], right_on=["year_reg", "sector_reg"])
        df = df.rename(columns={"industry_restrictions_1_0": f"L_{age}_industry_restrictions_1_0", 
                                "industry_restrictions_2_0": f"L_{age}_industry_restrictions_2_0"}, 
                                errors="raise")
                                
        df = df.drop(columns = {"sector_reg", "year_reg"})

        df = df.merge(gdp, how = "left", left_on=[f"L_{age}_year", "sector_2"], right_on=["year_gdp", "sector_2"])
        
        df = df.rename(columns={"gdp": f"L_{age}_gdp"}, 
                                errors="raise")
        df = df.drop(columns = {"year_gdp"})
        
        # merge the entry rate
        df = df.merge(df_age_new, 
            how = "left", left_on=[f"L_{age}_year", "sector"], right_on=["year_entry", "sector"])
        df = df.rename(columns={"entry_whole": f"L_{age}_entry_whole",
                                "incumbents_whole": f"L_{age}_incumbents_whole"})
        df = df.drop(columns = {"year_entry"})
        
        # create log variables
        with warnings.catch_warnings(): # suppress log zero warnings
            warnings.simplefilter("ignore")
            df[f"L_{age}_log_restriction_1_0"] = np.log(df[f"L_{age}_industry_restrictions_1_0"])
            df[f"L_{age}_log_restriction_2_0"] = np.log(df[f"L_{age}_industry_restrictions_2_0"])
            
            df[f"L_{age}_log_gdp"] = np.log(df[f"L_{age}_gdp"])
            
            df[f"L_{age}_entry_rate_whole"] = df[f"L_{age}_entry_whole"]/df[f"L_{age}_incumbents_whole"]
            
    return df


def data_average(df_raw, config):
    """
    data_average function
        - utilize life path data to create the average regulation of surviving firms
        - create log and flow measure measurements on each cohort life path
        
    Args:
        df [DataFrame]: Raw BDS data
        config [str]: config file
    Returns:
        data with average measure
    """
    
    ####################
    # Load lift path data
    ####################
    
    df = data_life_path(df_raw, config)

    ####################
    # Calculate avg including entry year
    ####################
        
    for age in range(1, 6):
        exo_var_list = []
        for lags in range(1, age + 1):
            exo_var_list.append(f"L_{lags}_log_restriction_2_0")
        
        df.loc[df.age_grp_dummy == age ,"avg_log_restriction_2_0"] = df[exo_var_list].sum(axis=1)/age
        
        exo_var_list = []    
        for lags in range(1, age + 1):    
            exo_var_list.append(f"L_{lags}_entry_rate_whole")
        
        df.loc[df.age_grp_dummy == age ,"avg_entry_rate_whole"] = df[exo_var_list].sum(axis=1)/age
            
        for lags in range(1, age + 1):    
            exo_var_list.append(f"L_{lags}_log_gdp")
        
        df.loc[df.age_grp_dummy == age ,"avg_log_gdp"] = df[exo_var_list].sum(axis=1)/age 
    
    ####################
    # Calculate avg not including entry year
    ####################    
    for age in range(2, 6):
        exo_var_list = []
        for lags in range(1, age):
            exo_var_list.append(f"L_{lags}_log_restriction_2_0")
        
        df.loc[df.age_grp_dummy == age ,"inc_avg_log_restriction_2_0"] = df[exo_var_list].sum(axis=1)/(age -1)
        df.loc[df.age_grp_dummy == age ,"cohort_log_restriction_2_0"] = df[f"L_{age}_log_restriction_2_0"]
        
        exo_var_list = []    
        for lags in range(1, age):    
            exo_var_list.append(f"L_{lags}_entry_rate_whole")
        
        df.loc[df.age_grp_dummy == age ,"inc_avg_entry_rate_whole"] = df[exo_var_list].sum(axis=1)/(age -1)
        df.loc[df.age_grp_dummy == age ,"cohort_entry_rate_whole"] = df[f"L_{age}_entry_rate_whole"]

        for lags in range(1, age):    
            exo_var_list.append(f"L_{lags}_log_gdp")
        
        df.loc[df.age_grp_dummy == age ,"inc_avg_log_gdp"] = df[exo_var_list].sum(axis=1)/(age -1) 
        df.loc[df.age_grp_dummy == age ,"cohort_log_gdp"] = df[f"L_{age}_log_gdp"]
        
    return df


def data_output(config_file):
    """
    data_clean function clean and create the final dataset
    Args:
        config_file [str]: path to config file
    Returns:
        Final data
    """

    ####################
    # Load data and config file
    ####################
    config = parse_config(config_file)
    df_sec_sz_ag, df_sec_ag = data_load(config)[0:2]
    df_cohort_robust = data_cohort_robust(config)
    
    # data using age sector
    df_life_path_sec_ag = data_life_path(df_sec_ag, config)
    df_average_sec_ag = data_average(df_sec_ag, config)
    
    # data using age sector
    df_life_path_sec_sz_ag= data_life_path(df_sec_sz_ag, config)
    df_average_sec_sz_ag = data_average(df_sec_sz_ag, config)
    
    # store cleaned dataset
    cleaned_data_path = Path(config["make_data"]["cleaned_data_path"])
    df_cohort_robust.to_csv(Path.cwd()/cleaned_data_path/"cohort_robust.csv")
    df_life_path_sec_ag.to_csv(Path.cwd()/cleaned_data_path/"life_path_sec_ag.csv")
    df_average_sec_ag.to_csv(Path.cwd()/cleaned_data_path/"average_sec_ag.csv")
    df_life_path_sec_sz_ag.to_csv(Path.cwd()/cleaned_data_path/"life_path_sec_sz_ag.csv")
    df_average_sec_sz_ag.to_csv(Path.cwd()/cleaned_data_path/"average_sec_sz_ag.csv")



@click.command()
@click.argument("config_file", type=str, default="src/config.yaml") 
def data_output_cmd(config_file):
    """
    data_output_cmd use to generate cmd commend
    """
    data_output(config_file)

if __name__ == "__main__":
    data_output_cmd()