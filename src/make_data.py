"""
This script load and clean the raw data to prepare for regression that study
the regulation at the time of entering on firm exit rates
"""

import warnings

import logging
import click
from pathlib import Path
import pandas as pd
import numpy as np

from src.utility import parse_config

# options of pandas
pd.options.mode.use_inf_as_na = True

def data_load(config):
    """
    data_load function that load raw Regdata and BDS data

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

    # reg data
    regdata = pd.read_csv(data_file_path/regdata_path)
    regdata['sector_reg'] = regdata['NAICS']
    regdata = regdata.loc[:,["year", "sector_reg", "industry_restrictions_1_0", "industry_restrictions_2_0"]]

    # sector gdp data
    gdp = pd.read_csv(data_file_path/gdp_path)
    gdp = pd.melt(gdp, id_vars='sector_2', var_name='year', value_name='gdp')
    gdp['year'] = pd.to_numeric(gdp['year']).astype(np.int64)

    # load BDS dataset by age sector
    df = pd.read_csv(data_file_path/bds_naics_4_path)

    ####################
    # Create Entry and Exit by Sectors
    ####################

    # change variable types
    df['firms'] = pd.to_numeric(df['firms'], errors='coerce', downcast=None)
    df['death'] = pd.to_numeric(df['firmdeath_firms'], errors='coerce', downcast=None)

    # aggregate entry by sector
    # creat aggregation for entry and death measure

    df_age = pd.DataFrame()
    df_age['entry_whole'] = df.loc[df['fage'] == 'a) 0', :].groupby(['year', 'sector'])['firms'].sum()
    df_age['incumbents_whole'] = df.loc[df['fage'] != 'a) 0', :].groupby(['year', 'sector'])['firms'].sum()
    df_age = df_age.reset_index()
    # df_age['death'] = df.loc[df['fage'] != 'a) 0', :].groupby(['year', 'sector'])['firmdeath_firms'].sum()

    return df, regdata, gdp, df_age
 
def data_merge(config):
    """
    data_merge function merge Regdata sector GDP, and BDS data
    Args:
        config [str]: config file
    Returns:
        Merged data
    """
    ####################
    # Load data
    ####################
    df, regdata, gdp, df_age= data_load(config)

    ####################
    # Merge Datasets
    ####################

    # merge df and regulation levels at different industry levels
    for naics in range(2, 5):
        # clean and create variables
        sector_name = f"sector_{naics}"
        df[sector_name] = df['sector'].astype(str).str.slice(0,naics)
        df[sector_name] = pd.to_numeric(df[sector_name])

        df = df.merge(regdata, how = "left", left_on=["year", sector_name], right_on=["year", "sector_reg"])
        df = df.rename(columns={"industry_restrictions_1_0": f"industry_restrictions_1_0_{naics}", "industry_restrictions_2_0": f"industry_restrictions_2_0_{naics}"}, errors="raise")
        df = df.drop(columns='sector_reg')
        df[f'log_restriction_1_{naics}'] = np.log(df[f'industry_restrictions_1_0_{naics}'])
        df[f'log_restriction_2_{naics}'] = np.log(df[f'industry_restrictions_2_0_{naics}'])

    # merge with sector gdp
    df = df.merge(gdp, how = "left", left_on=["year", "sector_2"], right_on=["year", "sector_2"])
    
    # merge with Sector Entry and
    df = df.merge(df_age, how = "left", left_on=["year", "sector"], right_on=["year", "sector"])
    
    return df

def data_cohort(config):
    """
    data_cohort function merge regulation, gdp, entry at the time of entering for each cohort
    Args:
        config [str]: config file
    Returns:
        Merged cohort data
    """
    ####################
    # Load raw and merged data
    ####################

    df = data_merge(config)
    df_raw, regdata, gdp, df_age = data_load(config)

    ####################
    # Define coarse age groups and cohort year variables
    ####################

    # define age groups
    df = df.sort_values(by=['year', 'sector', 'fage'])
    i = 0
    for fage in df.fage.unique():
        df.loc[df.fage == fage, 'age_grp_dummy'] = i
        i = i + 1

    df.loc[df.age_grp_dummy >= 7, 'age_grp_dummy'] = 7
    
    # create cohort year data
    df.loc[df.age_grp_dummy <= 5, 'cohort'] = \
        df.loc[df.age_grp_dummy <= 5, 'year'] - df.loc[df.age_grp_dummy <= 5, 'age_grp_dummy']

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
                                
        df = df.drop(columns='sector_reg')
        df[f'log_restriction_1_{naics}_cohort'] = np.log(df[f'industry_restrictions_1_0_{naics}_cohort'])
        df[f'log_restriction_2_{naics}_cohort'] = np.log(df[f'industry_restrictions_2_0_{naics}_cohort'])

    gdp = gdp.rename(columns={"year": "cohort", 
                              "gdp": "gdp_cohort"})
    gdp['cohort'] = pd.to_numeric(gdp['cohort']).astype(np.int64)

    df = df.merge(gdp, how = "left", left_on=["cohort", "sector_2"], right_on=["cohort", "sector_2"])

    # merge the entry rate at the cohort level
    df_age_new = df_age.rename(columns={"year": "cohort", 
                                        "entry_whole": "entry_whole_cohort",
                                        "incumbents_whole": "incumbents_whole_cohort"})
    df_age_new = df_age_new[["cohort", "sector", "entry_whole_cohort", "incumbents_whole_cohort"]]
    df = df.merge(df_age_new, 
        how = "left", left_on=["cohort", 'sector'], right_on=["cohort", "sector"])

    return df


def data_clean(config_file, return_data="N"):
    """
    data_clean function clean and create the final dataset
    Args:
        config_file [str]: path to config file
    Returns:
        Final data
    """
    ####################
    # Load data and config_file
    ####################

    # load config from config file
    config = parse_config(config_file)
    
    df = data_cohort(config)

    ####################
    # Create Variables for USe
    ####################

    # create some variables
    with warnings.catch_warnings(): # supress log zero warnings
        warnings.simplefilter('ignore')
        df['log_entry_whole'] = np.log(df['entry_whole'])
        df['log_death'] = np.log(df['death'])
        df['log_incumbents_whole'] = np.log(df['incumbents_whole'])
        df['entry_rate_whole'] = df['entry_whole']/df['incumbents_whole'] 
        df['death_rate_age'] = df['death']/df['firms'] 
        df['log_gdp'] = np.log(df['gdp'])
        df["entry_rate_whole_cohort"] = df["entry_whole_cohort"]/df["incumbents_whole_cohort"]
        df["log_gdp_cohort"] = np.log(df["gdp_cohort"])
    
    # store cleaned dataset
    cleaned_data_path = Path(config["make_data"]["cleaned_data_path"])
    df.to_csv(Path.cwd()/cleaned_data_path)

    match return_data:
        case "N":
            pass
        case "Y":
            return df
        case _:
            raise Exception("Select 'Y' to output dataframe or 'N' to omit it.")

@click.command()
@click.argument("config_file", type=str, default="src/config.yaml") 
def data_clean_cmd(config_file):
    """
    data_clean_cmd use to generate cmd commend
    """
    data_clean(config_file)

if __name__ == "__main__":
    data_clean_cmd()