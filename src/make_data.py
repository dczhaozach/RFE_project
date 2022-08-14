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
    regdata_path = Path(config["make_data"]["regdata_origin_path"])
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


def data_regdata(config):
    """
    data_regdata function that
        - load raw Regdata
        - create shift share instrument
    Args:
        config [str]: config file
    Returns:
        Reg data
        
    """
    
    ####################
    # Load data
    ####################  
    
    # path
    data_file_path = Path(config["make_data"]["data_file_path"])
    regdata_doc_path = Path(config["make_data"]["regdata_doc_path"])
    regdata_ind_path = Path(config["make_data"]["regdata_ind_path"])
    
    # doc words count
    df_doc = pd.read_csv(data_file_path/regdata_doc_path)
    # ind doc probability
    df_ind = pd.read_csv(data_file_path/regdata_ind_path)
    
    ####################
    # Create merged dataset
    ####################
    
    # clean variables
    df_doc["year"] = pd.to_numeric(df_doc.year.str.slice(0,4))
    df_ind["year"] = pd.to_numeric(df_ind.year.str.slice(0,4))

    # create lag variable
    df_doc = lag_variable(df_doc, ["year"], ["document_reference"], ["restrictions_2_0"], 1)

    # create current measure
    doc_var = [
        "year",
        "document_reference",
        "restrictions_1_0",
        "restrictions_2_0",
        "L_1_restrictions_2_0",
        ]
    
    df_merge = df_ind.merge(
        df_doc[doc_var], 
        how='left', 
        on=["year", "document_reference"], 
        validate="many_to_one",
        )

    # create initial shares
    baseline_year = 1986
    df_init = df_merge.loc[
        df_merge.year == baseline_year,
        ["NAICS", "probability", "document_reference", "restrictions_2_0"]
        ]
    df_init = df_init.rename(columns = {"probability": "probability_init"})
    
    # initial shares
    df_init["reg_s_d_init"] = df_init["probability_init"] * df_init["restrictions_2_0"]
    df_init["reg_s_init"] = df_init.groupby(["NAICS"])["reg_s_d_init"].transform(lambda x: x.sum())
    df_init["share_init"] = np.where(df_init["reg_s_d_init"]>0, df_init["reg_s_d_init"] / df_init["reg_s_init"], 0) 
    
    # merge with main dataset
    init_var = ["NAICS", "document_reference", "reg_s_d_init", "reg_s_init", "share_init"]
    df_merge = df_merge.merge(df_init[init_var], how='left', on=["NAICS", "document_reference"], validate="many_to_one")
    
    # replace non record with zero
    for var in ["reg_s_d_init", "reg_s_init", "share_init"]:
        df_merge[var] = np.where(df_merge[var].isna(), 0, df_merge[var])
    
    # create current share
    df_merge["reg_s_d"] = df_merge["probability"] * df_merge["restrictions_2_0"]
    df_merge["reg_s"] = df_merge.groupby(["NAICS", "year"])["reg_s_d"].transform(lambda x: x.sum())
    df_merge["share"] = np.where(df_merge["reg_s_d"] > 0, df_merge["reg_s_d"] / df_merge["reg_s"], 0)
    
    # average initial log restriction (leave one out)
    df_merge["log_reg_s_d"] = np.where(df_merge["reg_s_d"] > 0, np.log(df_merge["reg_s_d"]), 0)
    #df_merge["log_reg_d_one_out"] = np.where(
    #    df_merge["restrictions_2_0"] - df_merge["reg_s_d"] > 0,
    #    np.log(df_merge["restrictions_2_0"] - df_merge["reg_s_d"]),
    #    0
    #    )
    
    df_merge["sum_log_reg_s_d"] = df_merge.groupby(["document_reference", "year"])["log_reg_s_d"].transform(lambda x: x.sum())
    df_merge["n_doc"] = df_merge.groupby(["document_reference", "year"])["log_reg_s_d"].transform(lambda x: x.size)
    df_merge["avg_log_reg_s_d"] = np.where(
        df_merge["n_doc"] > 1, 
        (df_merge["sum_log_reg_s_d"] - df_merge["log_reg_s_d"])/(df_merge["n_doc"] - 1),
        np.nan
        )
    
    # prepare to aggregate (sum)
    df_merge["bartik_iv"] = df_merge["avg_log_reg_s_d"] * df_merge["share_init"]
    #df_merge["bartik_iv"] = (df_merge["log_reg_d_one_out"]) * df_merge["share_init"]
    df_merge["industry_restrictions_2_0"] = df_merge["reg_s_d"]
    
    # aggregate data and finalize
    regdata = df_merge.groupby(by=["year", "NAICS"])[["industry_restrictions_2_0", "bartik_iv"]].sum()
    regdata = regdata.reset_index()
    regdata = regdata.rename(columns={"NAICS":"sector_reg"})
    
    return regdata

    
def data_clean(df, id_var, sector_dig, config):
    """
    data_clean function that
        - make preliminary data type changes and data clean for main data file

    Args:
        df [DataFrame]: input main data
        id_var [list]: list of strings contains id variables (sector, year, fage, and fsize)
        sector_dig [int]: sector digits
    Returns:
        Cleaned data

    """
    ####################
    # clean data
    ####################
    
    # change variable types
    dep_var = config["make_data"]["dep_var"]
    
    var_lst = dep_var + [
        "firmdeath_firms", "year", "firms", "emp",
        "estabs", "estabs_entry", "estabs_exit",
        "job_creation", "job_destruction",
        "net_job_creation", "denom"
        ]

    for var in var_lst:
        df[var] = pd.to_numeric(df[var], errors="coerce", downcast=None)

    # if age variable included
    if "fage" in id_var:
        # aggregate by age group
        # coarse age group
        conditions = [
            df["fage"] == 'b) 1',
            df["fage"] == 'c) 2',
            df["fage"] == 'd) 3',
            df["fage"] == 'e) 4',
            df["fage"] == 'f) 5',
            df["fage"] == 'g) 6 to 10',
            df["fage"] == 'h) 11 to 15',
            df["fage"] == 'i) 16 to 20',
            df["fage"] == 'j) 21 to 25',
            df["fage"] == 'k) 26+',
            df["fage"] == 'l) Left Censored'
        ]
        
        choices = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06-10",
            "11+",
            "11+",
            "11+",
            "11+",
            "11+",
        ]    
        
        df.loc[:, "age_coarse"] = np.select(conditions, choices, default="00")
        
        # variable selection for new dataset
        id_var.remove("fage") 
        id_var.append("age_coarse") 
    
    # if size variable included
    # create size dummy
    if "fsize" in id_var:
        df.loc[:, "large_firm"] = 0
        index = (
            (df["fsize"] == "f) 500 to 999")
            | (df["fsize"] == "g) 1000 to 2499")
            | (df["fsize"] == "h) 2500 to 4999")
            | (df["fsize"] == "i) 5000 to 9999")
            | (df["fsize"] == "j) 10000+")
        )
        df.loc[index, "large_firm"] = 1
        
        id_var.remove("fsize")
        id_var.append("large_firm")
    
    df = df.dropna(subset=["firms", "emp"])
        
    column_var = [
        'firms', 'estabs', 'emp', 'denom', 
        'estabs_entry', 'estabs_exit', 'job_creation', 
        'job_creation_births', 'job_creation_continuers', 
        'job_destruction', 'job_destruction_deaths', 
        'job_destruction_continuers', 'job_destruction_rate_deaths', 
        'net_job_creation', 'firmdeath_firms', 'firmdeath_estabs', 
        'firmdeath_emp'
        ]
    
    df = df.groupby(["year"] + id_var)[column_var].sum()
    df = df[df.firms > 0]
    df = df[df.emp > 0]
    df = df.reset_index()
    
    # calculate key rates
    df["estabs_entry_rate"] = df["estabs_entry"]/df["estabs"]
    df["estabs_exit_rate"] = df["estabs_exit"]/df["estabs"]
    df["job_creation_rate"] = df["job_creation"]/df["denom"]
    df["job_destruction_rate"] = df["job_destruction"]/df["denom"]
    df["net_job_creation_rate"] = df["net_job_creation"]/df["denom"]
    df["reallocation_rate"] = (
        df["job_creation_rate"] + df["job_destruction_rate"]
        - abs(df["job_creation_rate"] - df["job_destruction_rate"])
    )
    
    df["death"] = df["firmdeath_firms"]
    df["log_emp"] = np.log(df["emp"])
    df["log_avg_emp"] = np.log(df["emp"]) - np.log(df["firms"])
    df["death_rate"] = df["death"]/df["firms"]
    
    # create sector variables
    df["sector"] = df["sector"].astype(str).str.slice(0, sector_dig)
    df["sector"] = pd.to_numeric(df["sector"], errors="coerce", downcast=None)
    
    # create sector at different digit
    for naics in range(2, 5):
        if naics > 2 and "fsize" in id_var:
            break
        # clean and create variables
        sector_name = f"sector_{naics}"
        df[sector_name] = df["sector"].astype(str).str.slice(0,naics)
        df[sector_name] = pd.to_numeric(df[sector_name], errors="coerce", downcast=None)

    # recoding sectors at 2 digit level
    df.loc[df.sector_2 == 32, "sector_2"] = 31
    df.loc[df.sector_2 == 33, "sector_2"] = 31
    df.loc[df.sector_2 == 45, "sector_2"] = 44
    df.loc[df.sector_2 == 49, "sector_2"] = 48 
        
    # lag var change variables
    dep_var = dep_var + ["death_rate", "log_avg_emp", "log_emp", "emp", "firms"]
    
    for var in dep_var:
        for lags in range(0, 3):
            df = lag_variable(df, ["year"], id_var, [var], lags)
  
    for var in dep_var:
        df[f"{var}_chg"] = df[var] - df[f"L_1_{var}"]

    return df
    

def data_sector_entry(df, id_var):
    """
    data_sector_entry function that
        - load raw bds data
        - create Entry at sector level

    Args:
        df [DataFrame]: bds data
        sector [str]: sector variable (level) to aggregate
    Returns:
        Entry data

    """
    df_age = pd.DataFrame()
    df_age["entry"] = \
        df.loc[df["age_coarse"] == "00", :].groupby(["year"] + id_var)["firms"].sum()
    df_age["incumbents"] = \
        df.loc[df["age_coarse"] != "00", :].groupby(["year"] + id_var)["firms"].sum()
    df_age = df_age.reset_index()
    
    return df_age


def data_final(df_input, id_var):
    """
    data_final function 
        - merges regulation, gdp, entry based on sector, age, or even size
        
    Args:
        df_input [tuple or list]: A sequence of dataframe
        config [str]: config file
    Returns:
        Merged cohort data
    """
    ####################
    # Load raw and merged data
    ####################
    df, regdata, gdp, df_age= df_input
    
    ####################
    # Define coarse age groups and cohort year variables
    ####################

    #id_var = ["year", "sector", "age_coarse"]
    # define age groups
    df = df.sort_values(by= ["year"] + id_var)
    
    ####################
    # Merge variables
    ####################
    
    # rename the variables to merge (prevent from conflicts and harmonize data types)
    regdata = regdata.rename(columns={"year": "year_reg"})
    gdp = gdp.rename(columns={"year": "year_gdp"})
    gdp["year_gdp"] = pd.to_numeric(gdp["year_gdp"]).astype(np.int64)
    df_age_new = df_age.rename(columns={"year": "year_entry"})
    
    
    # merge by ages
    for lags in range(0, 3):
        # create lag year data
        df.loc[:, f"L_{lags}_year"] = df.loc[:, "year"] - lags

        ####################
        # Merge with cohort year variables
        ####################
        # regulation
        df = df.merge(
            regdata,
            how="left",
            left_on=[f"L_{lags}_year", "sector_2"], 
            right_on=["year_reg", "sector_reg"],
            validate="many_to_one"
            )
        df = df.rename(
            columns={
                "industry_restrictions_2_0": f"L_{lags}_industry_restrictions_2_0",
                "bartik_iv": f"L_{lags}_bartik_iv",
                },
            errors="raise"
            )                        
        df = df.drop(columns = {"sector_reg", "year_reg"})

        # gdp
        df = df.merge(
            gdp,
            how="left",
            left_on=[f"L_{lags}_year", "sector_2"],
            right_on=["year_gdp", "sector_2"], 
            validate="many_to_one"
            )
        df = df.rename(columns={"gdp": f"L_{lags}_gdp"}, errors="raise")
        df = df.drop(columns = {"year_gdp"})
        
        # merge the entry rate
        merge_var = id_var
        if "age_coarse" in merge_var:
            merge_var.remove("age_coarse")
            
        df = df.merge(
            df_age_new,
            how = "left",
            left_on=[f"L_{lags}_year"] + merge_var,
            right_on=["year_entry"] + merge_var,
            validate = "many_to_one"
            )
        df = df.rename(
            columns={
                "entry": f"L_{lags}_entry",
                "incumbents": f"L_{lags}_incumbents"
                }
            )
        df = df.drop(columns = {"year_entry"})
    
        # create log variables
        with warnings.catch_warnings(): # suppress log zero warnings
            warnings.simplefilter("ignore")
            df[f"L_{lags}_log_restriction_2_0"] = np.log(df[f"L_{lags}_industry_restrictions_2_0"])
            df[f"L_{lags}_log_gdp"] = np.log(df[f"L_{lags}_gdp"])
            df[f"L_{lags}_log_emp"] = np.log(df[f"L_{lags}_emp"])
            df[f"L_{lags}_entry_rate"] = df[f"L_{lags}_entry"]/df[f"L_{lags}_incumbents"]
    
    # create change variables        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lags in range(0, 2):
            lags_pre = lags + 1 
            df[f"L_{lags}_chg_log_restriction_2_0"] = (
                df[f"L_{lags}_log_restriction_2_0"] - df[f"L_{lags_pre}_log_restriction_2_0"]
                )
            df[f"L_{lags}_chg_bartik_iv"] = (
                df[f"L_{lags}_bartik_iv"] - df[f"L_{lags_pre}_bartik_iv"]
                )
            df[f"L_{lags}_chg_log_gdp"] = (
                df[f"L_{lags}_log_gdp"] - df[f"L_{lags_pre}_log_gdp"]
                )
            df[f"L_{lags}_chg_log_emp"] = (
                df[f"L_{lags}_log_emp"] -  df[f"L_{lags_pre}_emp"]
                )
            df[f"L_{lags}_chg_entry_rate"] = (
                df[f"L_{lags}_entry_rate"] - df[f"L_{lags_pre}_entry_rate"]
                )
            df[f"L_{lags}_emp_growth"] = (
                2 * (df[f"L_{lags}_emp"] - df[f"L_{lags_pre}_emp"])
                / (df[f"L_{lags}_emp"] + df[f"L_{lags_pre}_emp"])
                )
                
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
    df_age = data_sector_entry(df, ["sector_2"])
    
    # aggregate death and firm counts
    var_lst = ["death", "firms", "emp", "estabs", "estabs_entry", "estabs_exit", "job_creation", "job_destruction", "net_job_creation", "denom"]
    df_agg = pd.DataFrame(df.groupby(["year", "sector_2"])[var_lst].sum())
    
    regdata = regdata.rename(columns={"sector_reg": "sector_2"})
    df_agg = df_agg.merge(regdata, how="left", on=["sector_2", "year"])
    
    df_agg = df_agg.merge(gdp, how="left", on=["sector_2", "year"])
    
    df_agg = df_agg.merge(df_age, how="left", on=["sector_2", "year"])
    
    df_agg["entry_rate"] = df_agg[f"entry"]/df_agg["firms"]
    df_agg["death_rate"] = df_agg[f"death"]/df_agg["firms"]
    df_agg["estabs_entry_rate"] = df_agg["estabs_entry"]/df_agg["estabs"]
    df_agg["estabs_exit_rate"] = df_agg["estabs_exit"]/df_agg["estabs"]
    df_agg["job_creation_rate"] = df_agg["job_creation"] / df_agg["denom"]
    df_agg["job_destruction_rate"] = df_agg["job_destruction"] / df_agg["denom"]
    df_agg["net_job_creation_rate"] = df_agg["net_job_creation"] / df_agg["denom"]
    
    return df_agg



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
    df_sec_sz_ag_raw, df_sec_ag_raw, regdata, gdp = data_load(config)
    regdata_iv = data_regdata(config)
    
    print("cleaning the data")
    # clean data
    df_sec = data_clean(df_sec_ag_raw, ["sector"], 4, config)
    df_sec_ag = data_clean(df_sec_ag_raw, ["sector", "fage"], 4, config)
    df_sec_sz = data_clean(df_sec_sz_ag_raw, ["sector", "fsize"], 2, config)
    df_sec_sz_ag = data_clean(df_sec_sz_ag_raw, ["sector", "fsize", "fage"], 2, config)
    
    # create entry measures
    df_age_4 = data_sector_entry(df_sec_ag, ["sector"])
    df_age_sz_2 = data_sector_entry(df_sec_sz_ag, ["sector", "large_firm"])
    
    print("creating sector-level data")
    data_input_sec = (df_sec, regdata_iv, gdp, df_age_4)
    data_final_sec = data_final(data_input_sec, ["sector"])
    
    print("creating sector-size-level data")
    data_input_sec_sz = (df_sec_sz, regdata_iv, gdp, df_age_sz_2)
    data_final_sec_sz = data_final(data_input_sec_sz, ["sector","large_firm"])
    
    print("creating sector-age-level data")
    data_input_sec_ag = (df_sec_ag, regdata_iv, gdp, df_age_4)
    data_final_sec_ag = data_final(data_input_sec_ag, ["sector","age_coarse"])
    
    print("creating sector-age-size-level data")
    data_input_sec_sz_ag = (df_sec_sz_ag, regdata_iv, gdp, df_age_sz_2)
    data_final_sec_sz_ag = data_final(data_input_sec_sz_ag, ["sector", "large_firm", "age_coarse"])   
    
    
    print("creating aggregate pattern data")
    # aggregate data
    data_input_agg = (df_sec_ag, regdata_iv, gdp)
    df_agg = data_patterns(data_input_agg)
    
    print("saving data file")
    # store cleaned dataset
    cleaned_data_path = Path(config["make_data"]["cleaned_data_path"])
    
    data_final_sec.to_hdf(Path.cwd()/cleaned_data_path/"sector_panel.h5", key = "data")
    data_final_sec_sz.to_hdf(Path.cwd()/cleaned_data_path/"sector_size_panel.h5", key = "data")
    data_final_sec_ag.to_hdf(Path.cwd()/cleaned_data_path/"sector_age_panel.h5", key = "data")
    data_final_sec_sz_ag.to_hdf(Path.cwd()/cleaned_data_path/"sector_age_size_panel.h5", key = "data")
    df_agg.to_hdf(Path.cwd()/cleaned_data_path/"agg_pattern.h5", key = "data")
    

@click.command()
@click.argument("config_file", type=str, default="src/config.yaml") 
def data_output_cmd(config_file):
    """
    data_output_cmd use to generate cmd commend
    """
    data_output(config_file)

if __name__ == "__main__":
    data_output_cmd()