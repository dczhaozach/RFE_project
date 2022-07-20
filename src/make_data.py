"""
This script load and clean the raw data to prepare for regression that study
the regulation at the time of entering on firm exit rates
"""

import warnings

from pathlib import Path
import click
import pandas as pd
import numpy as np

from Src.utility import parse_config
from Src.utility import lag_variable

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
    # Load data path
    ####################

    # load data paths
    data_file_path = Path(config["make_data"]["data_file_path"])
    regdata_path = Path(config["make_data"]["regdata_path"])
    bds_naics_4_path = Path(config["make_data"]["bds_naics_4_path"])
    gdp_path = Path(config["make_data"]["gdp_path"])
    bds_sector_size_path = Path(config["make_data"]["bds_sector_size"])

    ####################
    # Load data
    ####################
    
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
    df_sec_ag = df_sec_ag.drop_duplicates(subset=['year', "sector", "fage"])
    
    # load BDS dataset by age sector size
    df_sec_sz_ag = pd.read_csv(data_file_path/bds_sector_size_path)
    df_sec_sz_ag = df_sec_sz_ag.drop_duplicates(subset=['year', "sector", "fage", "fsize"])
    
    return df_sec_sz_ag, df_sec_ag, regdata, gdp


def data_clean(df, id_var, sector_dig, config):
    """
    data_clean function that
        - make preliminary data type changes and data clean for main data file

    Args:
        df [DataFrame]: input main data
        id_var [list]: list of strings contains id variables (sector, year, and fsize)
        sector_dig [int]: sector digits
    Returns:
        Cleaned data

    """
    ####################
    # clean data
    ####################
    
    # change variable types
    dep_var = config["make_data"]["dep_var"]
    
    var_lst = dep_var + [ "emp", "firms", "firmdeath_firms", "year"]
    for var in var_lst:
        df[var] = pd.to_numeric(df[var], errors="coerce", downcast=None)

    df.loc[df.firms <= 0, 'firms'] = np.nan
    
    df["death"] = df["firmdeath_firms"]
    df["log_emp"] = np.log(df["emp"])
    df["death_rate"] = df["death"]/df["firms"]
    
    df["sector"] = df["sector"].astype(str).str.slice(0, sector_dig)
    df["sector"] = pd.to_numeric(df["sector"], errors="coerce", downcast=None)
    
    for naics in range(2, 5):
        if naics > 2 and "fsize" in id_var:
            break
        # clean and create variables
        sector_name = f"sector_{naics}"
        df[sector_name] = df["sector"].astype(str).str.slice(0,naics)
        df[sector_name] = pd.to_numeric(df[sector_name], errors="coerce", downcast=None)

    # recoding sectors
    df.loc[df.sector_2 == 32, "sector_2"] = 31
    df.loc[df.sector_2 == 33, "sector_2"] = 31
    df.loc[df.sector_2 == 45, "sector_2"] = 44
    df.loc[df.sector_2 == 49, "sector_2"] = 48   
    
        
    # lag var
    for age in range(1, 7):
        df = lag_variable(df, ["year"], id_var, ["emp"], age)
        df = lag_variable(df, ["year"], id_var, ["firms"], age)
        df = lag_variable(df, ["year"], id_var, ["death_rate"], age)
        
    df["L_0_emp"] = df["emp"]
    df["L_0_firms"] = df["firms"]
    
   # create size dummy
    if "fsize" in id_var:
        df["large_firm"] = 0
        index = (
            (df["fsize"] == "f) 500 to 999")
            | (df["fsize"] == "g) 1000 to 2499")
            | (df["fsize"] == "h) 2500 to 4999")
            | (df["fsize"] == "i) 5000 to 9999")
            | (df["fsize"] == "j) 10000+")
        )
        df.loc[index, "large_firm"] = 1
        
    return df
    

def data_sector_entry(df, sector):
    """
    data_sector_entry function that
        - load raw bds data
        - create Entry at sector level

    Args:
        df [DataFrame]: bds data
        sector [str]: sector level to aggregate
    Returns:
        Entry data

    """
    df_age = pd.DataFrame()
    df_age["entry_whole"] = \
        df.loc[df["fage"] == "a) 0", :].groupby(["year", sector])["firms"].sum()
    df_age["incumbents_whole"] = \
        df.loc[df["fage"] != "a) 0", :].groupby(["year", sector])["firms"].sum()
    df_age = df_age.reset_index()
    
    return df_age


def data_life_path(df_input):
    """
    data_life_path function 
        - merges regulation, gdp, entry on each cohort life path
        - create log and flow measure measurements on each cohort life path
        
    Args:
        df_input [tuple or list]: A sequence of dataframe
        config [str]: config file
    Returns:
        Merged cohort data
    """
    ####################
    # Load raw and merged data
    ####################
    df, regdata, gdp = df_input
    
    df_age = data_sector_entry(df, "sector")

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
    
    # merge by ages
    for age in range(0, 7):
        # create lag year data
        df.loc[df.age_grp_dummy <= 5, f"L_{age}_year"] = \
            df.loc[df.age_grp_dummy <= 5, "year"] - age

        ####################
        # Merge with cohort year variables
        ####################
        df = df.merge(regdata, how = "left", left_on=[f"L_{age}_year", "sector_2"], right_on=["year_reg", "sector_reg"], validate = "many_to_one")
        df = df.rename(columns={"industry_restrictions_1_0": f"L_{age}_industry_restrictions_1_0", 
                                "industry_restrictions_2_0": f"L_{age}_industry_restrictions_2_0"}, 
                                errors="raise")
                                
        df = df.drop(columns = {"sector_reg", "year_reg"})

        df = df.merge(gdp, how = "left", left_on=[f"L_{age}_year", "sector_2"], right_on=["year_gdp", "sector_2"], validate = "many_to_one")
        
        df = df.rename(columns={"gdp": f"L_{age}_gdp"}, 
                                errors="raise")
        df = df.drop(columns = {"year_gdp"})
        
        # merge the entry rate
        df = df.merge(df_age_new, 
            how = "left", left_on=[f"L_{age}_year", "sector"], right_on=["year_entry", "sector"] , validate = "many_to_one")
        df = df.rename(columns={"entry_whole": f"L_{age}_entry_whole",
                                "incumbents_whole": f"L_{age}_incumbents_whole"})
        df = df.drop(columns = {"year_entry"})
    
        # create log variables
        with warnings.catch_warnings(): # suppress log zero warnings
            warnings.simplefilter("ignore")
            df[f"L_{age}_log_restriction_1_0"] = np.log(df[f"L_{age}_industry_restrictions_1_0"])
            df[f"L_{age}_log_restriction_2_0"] = np.log(df[f"L_{age}_industry_restrictions_2_0"])
            df[f"L_{age}_log_gdp"] = np.log(df[f"L_{age}_gdp"])
            df[f"L_{age}_log_emp"] = np.log(df[f"L_{age}_emp"])
            df[f"L_{age}_entry_rate_whole"] = df[f"L_{age}_entry_whole"]/df[f"L_{age}_incumbents_whole"]
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for age in range(0, 6):
            age_pre = age + 1 
            df[f"L_{age}_chg_restriction_2_0"] = (df[f"L_{age}_log_restriction_2_0"] - 
                                                  df[f"L_{age_pre}_log_restriction_2_0"])
            df[f"L_{age}_chg_gdp"] = (df[f"L_{age}_log_gdp"] - 
                                      df[f"L_{age_pre}_log_gdp"])
            df[f"L_{age}_chg_emp"] = (df[f"L_{age}_log_emp"] - 
                                      df[f"L_{age_pre}_emp"])
            df[f"L_{age}_chg_entry_rate_whole"] = (df[f"L_{age}_entry_rate_whole"] - 
                                                  df[f"L_{age_pre}_entry_rate_whole"])
        
        df["death_rate_chg"] = df["death_rate"] - df["L_1_death_rate"]

        # cohort level data
        for age in range(1, 6):
            age_pre = age + 1
            df.loc[df["age_grp_dummy"] == age, f"full_chg_restriction_2_0"] = \
                (df.loc[df["age_grp_dummy"] == age, f"L_{0}_log_restriction_2_0"] - 
                 df.loc[df["age_grp_dummy"] == age, f"L_{age_pre}_log_restriction_2_0"])
            df.loc[df["age_grp_dummy"] == age, f"pre_cohort_log_restriction_2_0"] = \
                 df.loc[df["age_grp_dummy"] == age, f"L_{age_pre}_log_restriction_2_0"]
                            
            df.loc[df["age_grp_dummy"] == age, "firms_cohort"] = df.loc[df["age_grp_dummy"] == age, f"L_{age}_firms"]
            df.loc[df["age_grp_dummy"] == age, "log_emp_cohort"] = np.log(df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"])
            df.loc[df["age_grp_dummy"] == age, "log_emp_chg"] = np.log(df.loc[df["age_grp_dummy"] == age, "emp"]) - \
                np.log(df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"])
            df.loc[df["age_grp_dummy"] == age,"per_emp_chg"] = 2 * (df.loc[df["age_grp_dummy"] == age, "emp"] - 
                df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"]) /  (df.loc[df["age_grp_dummy"] == age, "emp"] +
                df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"])
                
    return df


def data_patterns(df_input):
    """
    data_patterns function 
        - generate aggregate data to see stylize facts
        
    Args:
        df_input [tuple or list]: A sequence of dataframe
        config [str]: config file
    Returns:
        Merged cohort data
    """
    assert(len(df_input) == 3)
    df, regdata, gdp = df_input
    df_age = data_sector_entry(df, "sector_2")
    
    # aggregate death and firm counts
    df_agg = pd.DataFrame((df.groupby(["year", "sector_2"])[["death", "firms"]].sum()))
        
    regdata = regdata.rename(columns={"sector_reg": "sector_2"})
    df_agg = df_agg.merge(regdata, how="left", on=["sector_2", "year"])
    
    df_agg = df_agg.merge(gdp, how="left", on=["sector_2", "year"])
    
    df_agg = df_agg.merge(df_age, how="left", on=["sector_2", "year"])
    df_agg["entry_rate_whole"] = df_agg[f"entry_whole"]/df_agg["firms"]
    df_agg["death_rate_whole"] = df_agg[f"death"]/df_agg["firms"]
    
    return df_agg


def data_average(df_input):
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
    
    df = df_input
    
    ####################
    # Calculate avg including entry year
    ####################
        
    for age in range(1, 6):
        pre_age = age + 1
        # changes in restrictions in current period
        df.loc[df.age_grp_dummy == age, "curr_chg_restriction_2_0"] = \
            (df.loc[df.age_grp_dummy == age, "L_0_log_restriction_2_0"] - 
            df.loc[df.age_grp_dummy == age, "L_1_log_restriction_2_0"]) 
        
        # changes in restrictions after entering
        df.loc[df.age_grp_dummy == age, "life_chg_restriction_2_0"] = \
            (df.loc[df.age_grp_dummy == age, "L_1_log_restriction_2_0"] - 
            df.loc[df.age_grp_dummy == age, f"L_{age}_log_restriction_2_0"])
        
        # changes in restrictions when entering
        df.loc[df.age_grp_dummy == age, "enter_chg_restriction_2_0"] = \
            (df.loc[df.age_grp_dummy == age, f"L_{age}_log_restriction_2_0"] - 
            df.loc[df.age_grp_dummy == age, f"L_{pre_age}_log_restriction_2_0"]) 
               
        # level in restrictions before entering
        df.loc[df.age_grp_dummy == age, "cohort_log_restriction_2_0"] = \
            df.loc[df.age_grp_dummy == age, f"L_{pre_age}_log_restriction_2_0"]

    return df


def data_cohort_robust(df_input):
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
    df, regdata, gdp = df_input
    df_age = data_sector_entry(df, "sector")
    
    ####################
    # Merge Datasets at observed years
    ####################

    # merge df and regulation levels at different industry levels
    for naics in range(2, 5):
        # clean and create variables
        sector_name = f"sector_{naics}"
        
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
    df.loc[df.age_grp_dummy <= 5, "pre_cohort"] = df.loc[df.age_grp_dummy <= 5, "cohort"] - 1
      
    ####################
    # Merge with cohort year variables
    ####################

    # merge reg data at entry year
    regdata = regdata.rename(columns={"year": "pre_cohort"})

    for naics in range(2, 5):
        # clean and create variables
        sector_name = f"sector_{naics}"

        df = df.merge(regdata, how = "left", left_on=["pre_cohort", sector_name], right_on=["pre_cohort", "sector_reg"])
        df = df.rename(columns={"industry_restrictions_1_0": f"industry_restrictions_1_0_{naics}_pre_cohort", 
                                "industry_restrictions_2_0": f"industry_restrictions_2_0_{naics}_pre_cohort"}, 
                                errors="raise")
                                
        df = df.drop(columns="sector_reg")
        
        # create log variables
        df[f"log_restriction_1_{naics}_pre_cohort"] = np.log(df[f"industry_restrictions_1_0_{naics}_pre_cohort"])
        df[f"log_restriction_2_{naics}_pre_cohort"] = np.log(df[f"industry_restrictions_2_0_{naics}_pre_cohort"])

        # changes in log regulation
        df[f"chg_restriction_2_0_{naics}"] = df[f"log_restriction_2_{naics}"] - df[f"log_restriction_2_{naics}_pre_cohort"]
    
    # controls
    gdp = gdp.rename(columns={"year": "pre_cohort", 
                              "gdp": "gdp_pre_cohort"})
    gdp["pre_cohort"] = pd.to_numeric(gdp["pre_cohort"]).astype(np.int64)

    df = df.merge(gdp, how = "left", left_on=["pre_cohort", "sector_2"], right_on=["pre_cohort", "sector_2"])

    df_emp = df[["year","sector","emp"]]
    df_emp = df_emp.rename(columns={"emp": "emp_pre_cohort", "year": "pre_cohort"})
    df = df.merge(df_emp, how = "left", left_on=["pre_cohort", "sector"], right_on=["pre_cohort", "sector"])
    
    # merge the entry rate at the cohort level
    df_age_new = df_age.rename(columns={"year": "pre_cohort", 
                                        "entry_whole": "entry_whole_pre_cohort",
                                        "incumbents_whole": "incumbents_whole_pre_cohort"})
    df_age_new = df_age_new[["pre_cohort", "sector", "entry_whole_pre_cohort", "incumbents_whole_pre_cohort"]]
    
    df = df.merge(df_age_new, 
        how = "left", left_on=["pre_cohort", "sector"], right_on=["pre_cohort", "sector"])

    # create some variables
    with warnings.catch_warnings(): # suppress log zero warnings
        warnings.simplefilter("ignore")
        df["log_gdp"] = np.log(df["gdp"])
        df["log_gdp_pre_cohort"] = np.log(df["gdp_pre_cohort"])

        df["entry_rate_whole"] = df["entry_whole"]/df["incumbents_whole"]  
        df["entry_rate_whole_pre_cohort"] = df["entry_whole_pre_cohort"]/df["incumbents_whole_pre_cohort"]       
        df["log_emp_pre_cohort"] = np.log(df["emp_pre_cohort"])
        df["log_emp_chg"] = df["log_emp"] - df["log_emp_pre_cohort"]
        df["per_emp_chg"] = 2 * (df["emp"] - df["emp_pre_cohort"])/(df["emp"] + df["emp_pre_cohort"])
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
    print("loading config files")
    config = parse_config(config_file)
    df_sec_sz_ag, df_sec_ag, regdata, gdp = data_load(config)
    
    print("cleaning the data")
    # clean data
    df_sec_ag = data_clean(df_sec_ag, ["sector", "fage"], 4, config)
    df_sec_sz_ag = data_clean(df_sec_sz_ag, ["sector", "fage", "fsize"], 2, config)
    
    print("creating life path and average data")
    # data using age sector
    data_input_sec_ag = (df_sec_ag, regdata, gdp)
    df_life_path_sec_ag = data_life_path(data_input_sec_ag)

    df_average_sec_ag = data_average(df_life_path_sec_ag)
    
    print("creating aggregate pattern data")
    # aggregate data
    data_input_agg = (df_sec_ag, regdata, gdp)
    df_agg = data_patterns(data_input_agg)
    
    print("creating hetero data file")
    # data using age sector size
    data_input_sec_sz_ag = (df_sec_sz_ag, regdata, gdp)
    df_life_path_sec_sz_ag = data_life_path(data_input_sec_sz_ag)
    df_average_sec_sz_ag = data_average(df_life_path_sec_sz_ag)
    
    print("creating robust data file")
    # robust
    df_cohort_robust = data_cohort_robust(data_input_sec_ag)
    
    print("saving data file")
    # store cleaned dataset
    cleaned_data_path = Path(config["make_data"]["cleaned_data_path"])
    df_cohort_robust.to_csv(Path.cwd()/cleaned_data_path/"cohort_robust.csv")
    df_life_path_sec_ag.to_csv(Path.cwd()/cleaned_data_path/"life_path_sec_ag.csv")
    df_agg.to_csv(Path.cwd()/cleaned_data_path/"agg_pattern.csv")
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