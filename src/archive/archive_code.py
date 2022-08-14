import os
from dotenv import load_dotenv, find_dotenv

def data_api_bds(config):
    """
    data_api_bds function that
        - load raw BDS data
        - clean and rename variables

    Args:
        config [str]: config file
    Returns:
        BDS data

    """
    # find .env automagically by walking up directories until it's found
    dotenv_path = find_dotenv()

    # load up the entries as environment variables
    load_dotenv(dotenv_path)
    database_url = os.environ.get("DATABASE_URL")
    
    response = requests.get(api_url)
    df_json = response.json()
    df = pd.DataFrame(df_json)
    
    # create header
    new_header = df.iloc[0] 
    df = df[1:]
    df.columns = new_header
    
    # rename 
    
    df['n'].replace({'a': 'x', 'b': 'y', 'c': 'w', 'd': 'z'})


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
    df = df.sort_values(by=["year", "sector", "age_coarse"])
    
    i = 0
    for age_coarse in df.age_coarse.unique():
        df.loc[df.age_coarse == age_coarse, "age_grp_dummy"] = i
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
        df.loc[:, f"L_{age}_year"] = \
            df.loc[:, "year"] - age

        ####################
        # Merge with cohort year variables
        ####################
        df = df.merge(regdata, how = "left", left_on=[f"L_{age}_year", "sector_2"], right_on=["year_reg", "sector_reg"], validate = "many_to_one")
        df = df.rename(columns={
                "industry_restrictions_1_0": f"L_{age}_industry_restrictions_1_0", 
                "industry_restrictions_2_0": f"L_{age}_industry_restrictions_2_0",
                "bartik_iv": f"L_{age}_bartik_iv",
            }, errors="raise")
                                
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
            df[f"L_{age}_chg_log_restriction_2_0"] = (df[f"L_{age}_log_restriction_2_0"] - 
                                                  df[f"L_{age_pre}_log_restriction_2_0"])
            df[f"L_{age}_chg_bartik_iv"] = (df[f"L_{age}_bartik_iv"] - 
                                                  df[f"L_{age_pre}_bartik_iv"])
            df[f"L_{age}_chg_log_gdp"] = (df[f"L_{age}_log_gdp"] - 
                                      df[f"L_{age_pre}_log_gdp"])
            df[f"L_{age}_chg_log_emp"] = (df[f"L_{age}_log_emp"] - 
                                      df[f"L_{age_pre}_emp"])
            df[f"L_{age}_chg_entry_rate_whole"] = (df[f"L_{age}_entry_rate_whole"] - 
                                                  df[f"L_{age_pre}_entry_rate_whole"])
            df[f"L_{age}_emp_growth"] = 2 * ((df[f"L_{age}_emp"] - df[f"L_{age_pre}_emp"]) / 
                                         (df[f"L_{age}_emp"] + df[f"L_{age_pre}_emp"]))

        # cohort level data
        for age in range(1, 6):
            age_pre = age + 1
            df.loc[df["age_grp_dummy"] == age, f"full_chg_restriction_2_0"] = \
                (df.loc[df["age_grp_dummy"] == age, f"L_{0}_log_restriction_2_0"] - 
                 df.loc[df["age_grp_dummy"] == age, f"L_{age_pre}_log_restriction_2_0"])/age
            df.loc[df["age_grp_dummy"] == age, f"pre_cohort_log_restriction_2_0"] = \
                 df.loc[df["age_grp_dummy"] == age, f"L_{age_pre}_log_restriction_2_0"]
                            
            df.loc[df["age_grp_dummy"] == age, "firms_cohort"] = df.loc[df["age_grp_dummy"] == age, f"L_{age}_firms"]
            df.loc[df["age_grp_dummy"] == age, "log_emp_cohort"] = np.log(df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"])
            df.loc[df["age_grp_dummy"] == age, "log_emp_chg_cohort"] = np.log(df.loc[df["age_grp_dummy"] == age, "emp"]) - \
                np.log(df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"])
            df.loc[df["age_grp_dummy"] == age, "log_avg_emp_chg_cohort"] = \
                ( \
                (np.log(df.loc[df["age_grp_dummy"] == age, "emp"]) - np.log(df.loc[df["age_grp_dummy"] == age, "firms"])) - \
                (np.log(df.loc[df["age_grp_dummy"] == age, f"L_{age}_emp"]) - np.log(df.loc[df["age_grp_dummy"] == age, f"L_{age}_firms"])) \
                )
            df.loc[df["age_grp_dummy"] == age,"per_emp_chg_cohort"] = 2 * (df.loc[df["age_grp_dummy"] == age, "emp"] - 
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
    var_lst = ["death", "firms", "emp", "estabs", "estabs_entry", "estabs_exit", "job_creation", "job_destruction", "net_job_creation", "denom"]
    df_agg = pd.DataFrame(df.groupby(["year", "sector_2"])[var_lst].sum())
    
    regdata = regdata.rename(columns={"sector_reg": "sector_2"})
    df_agg = df_agg.merge(regdata, how="left", on=["sector_2", "year"])
    
    df_agg = df_agg.merge(gdp, how="left", on=["sector_2", "year"])
    
    df_agg = df_agg.merge(df_age, how="left", on=["sector_2", "year"])
    
    df_agg["entry_rate_whole"] = df_agg[f"entry_whole"]/df_agg["firms"]
    df_agg["death_rate_whole"] = df_agg[f"death"]/df_agg["firms"]
    df_agg["estabs_entry_rate"] = df_agg["estabs_entry"]/df_agg["estabs"]
    df_agg["estabs_exit_rate"] = df_agg["estabs_exit"]/df_agg["estabs"]
    df_agg["job_creation_rate"] = df_agg["job_creation"] / df_agg["denom"]
    df_agg["job_destruction_rate"] = df_agg["job_destruction"] / df_agg["denom"]
    df_agg["net_job_creation_rate"] = df_agg["net_job_creation"] / df_agg["denom"]
    
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
            df.loc[df.age_grp_dummy == age, f"L_{age}_log_restriction_2_0"])/(age - 1)
        
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
    df = df.sort_values(by=["year", "sector", "age_coarse"])
    i = 0
    for age_coarse in df.age_coarse.unique():
        df.loc[df.age_coarse == age_coarse, "age_grp_dummy"] = i
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
    regdata_iv = data_regdata(config)
    
    print("cleaning the data")
    # clean data
    df_sec_ag = data_clean(df_sec_ag, ["sector", "fage"], 4, config)
    df_sec_sz_ag = data_clean(df_sec_sz_ag, ["sector", "fsize", "fage"], 2, config)
    
    print("creating life path and average data")
    # data using age sector
    data_input_sec_ag = (df_sec_ag, regdata_iv, gdp)
    df_life_path_sec_ag = data_life_path(data_input_sec_ag)

    df_average_sec_ag = data_average(df_life_path_sec_ag)
    
    print("creating aggregate pattern data")
    # aggregate data
    data_input_agg = (df_sec_ag, regdata_iv, gdp)
    df_agg = data_patterns(data_input_agg)
    
    print("creating hetero data file")
    # data using age sector size
    data_input_sec_sz_ag = (df_sec_sz_ag, regdata_iv, gdp)
    df_life_path_sec_sz_ag = data_life_path(data_input_sec_sz_ag)
    df_average_sec_sz_ag = data_average(df_life_path_sec_sz_ag)
    
    print("creating robust data file")
    # robust
    data_input_cohort = (df_sec_ag, regdata, gdp)
    df_cohort_robust = data_cohort_robust(data_input_cohort)
    
    print("saving data file")
    # store cleaned dataset
    cleaned_data_path = Path(config["make_data"]["cleaned_data_path"])
    df_cohort_robust.to_hdf(Path.cwd()/cleaned_data_path/"cohort_robust.h5", key = "data")
    df_life_path_sec_ag.to_hdf(Path.cwd()/cleaned_data_path/"life_path_sec_ag.h5", key = "data")
    df_agg.to_hdf(Path.cwd()/cleaned_data_path/"agg_pattern.h5", key = "data")
    df_average_sec_ag.to_hdf(Path.cwd()/cleaned_data_path/"average_sec_ag.h5", key = "data")
    df_life_path_sec_sz_ag.to_hdf(Path.cwd()/cleaned_data_path/"life_path_sec_sz_ag.h5", key = "data")
    df_average_sec_sz_ag.to_hdf(Path.cwd()/cleaned_data_path/"average_sec_sz_ag.h5", key = "data")
    
    
    
                
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
    results_figs_path = Path(config["model"]["results_figs_path"])
    fig_path = Path.cwd()/results_figs_path
    
    df = pd.read_csv(Path.cwd()/cleaned_data_path) 
    std = df["L_0_log_restriction_2_0"].std()

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
        exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
        v_names = [f"pre_cohort_log_restriction_2_0", f"full_chg_restriction_2_0"]
        coefs_cohort = coef_dict(v_names, res, coefs_cohort, age)

    # save results and figs
    model_name = f"{depend_var}_results_cohort_age_LP"
    v_names = [f"pre_cohort_log_restriction_2_0", f"full_chg_restriction_2_0"]
    var_names = ["Before Entering", "Avg. after Entering"]

    df_coefs_cohort = pd.DataFrame(coefs_cohort)
    df_coefs_cohort = df_coefs_cohort.sort_values(by=['name', 'age'])
    df_coefs_cohort.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{model_name}.csv") 
    plot_lp(v_names, df_coefs_cohort, depend_var, model_name, fig_path, var_names, std)
    
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
            exo_vars.append(f"L_{lags}_chg_log_restriction_2_0")
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        pre_entry_age = age + 1
        exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
        v_names = [f"pre_cohort_log_restriction_2_0"]
        coefs_age = coef_dict(v_names, res, coefs_age, age)

    # save results and figs
    model_name = f"{depend_var}_results_path_age_LP"
    v_names = [f"pre_cohort_log_restriction_2_0"]
    var_names = ["Before Entering"]
    
    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name', 'age'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{model_name}.csv") 
    plot_lp(v_names, df_coefs_age, depend_var, model_name, fig_path, var_names, std)

    
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
    results_figs_path = Path(config["model"]["results_figs_path"])
    fig_path = Path.cwd()/results_figs_path

    df = pd.read_csv(Path.cwd()/cleaned_data_path) 
    std = df["L_0_log_restriction_2_0"].std()
    
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
        data["cross"] = data[f"pre_cohort_log_restriction_2_0"] * data["large_firm"]
        
        # create exogenous variable (list)
        exo_vars = []
        for lags in range(0, age+1):
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"full_chg_restriction_2_0")
        exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
        v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
        coefs_age = coef_dict(v_names, res, coefs_age, age)
    
    # save results and figs
    model_name = f"{depend_var}_results_cohort_age_h_LP"
    v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
    var_names = ["Cross", "Before Entering", "Large"]

    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name', 'age'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{model_name}.csv") 
    plot_lp(v_names, df_coefs_age, depend_var, model_name, fig_path, var_names, std)
    

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
        data["cross"] = data[f"pre_cohort_log_restriction_2_0"] * data["large_firm"]
        
        # create exogenous variable (list)
        exo_vars = []
        for lags in range(0, age + 1):
            exo_vars.append(f"L_{lags}_chg_log_restriction_2_0")
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
        v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
        coefs_age = coef_dict(v_names, res, coefs_age, age)
    
    # save results and figs
    model_name = f"{depend_var}_results_path_age_h_LP"
    v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
    var_names = ["Cross", "Before Entering", "Large"]

    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name', 'age'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{model_name}.csv") 
    plot_lp(v_names, df_coefs_age, depend_var, model_name, fig_path, var_names, std)

            
        
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
    results_figs_path = Path(config["model"]["results_figs_path"])
    fig_path = Path.cwd()/results_figs_path

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    std = df["L_0_log_restriction_2_0"].std()

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
    
    exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
            "enter_chg_restriction_2_0", f"pre_cohort_log_restriction_2_0"]
    coefs_age = coef_dict(v_names, res, coefs_age, age)
    
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
        
        exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
        v_names = ["curr_chg_restriction_2_0", "life_chg_restriction_2_0",
                   "enter_chg_restriction_2_0", f"pre_cohort_log_restriction_2_0"]
        coefs_age = coef_dict(v_names, res, coefs_age, age)

    # save results and figs
    model_name = f"{depend_var}_results_average_age_LP"
    v_names = ["curr_chg_restriction_2_0", "life_chg_restriction_2_0",
                   "enter_chg_restriction_2_0", f"pre_cohort_log_restriction_2_0"]
    var_names = ["Current Change", "After Entering", "When Entering", "Before Entering"]

    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name', 'age'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{model_name}.csv") 
    plot_lp(v_names, df_coefs_age, depend_var, model_name, fig_path, var_names, std)

       
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
    results_figs_path = Path(config["model"]["results_figs_path"])
    fig_path = Path.cwd()/results_figs_path

    df = pd.read_csv(Path.cwd()/cleaned_data_path)
    std = df["L_0_log_restriction_2_0"].std()
    
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
    data["cross"] = data[f"pre_cohort_log_restriction_2_0"] * data["large_firm"]
    
    exo_vars = ["curr_chg_restriction_2_0", "enter_chg_restriction_2_0"]
    
    for lags in range(0, age + 1):
        exo_vars.append(f"L_{lags}_entry_rate_whole")
        exo_vars.append(f"L_{lags}_log_gdp")
    
    exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
    v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
    coefs_age = coef_dict(v_names, res, coefs_age, age)
    
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
        data["cross"] = data[f"pre_cohort_log_restriction_2_0"] * data["large_firm"]
        
        exo_vars = ["curr_chg_restriction_2_0", "life_chg_restriction_2_0", "enter_chg_restriction_2_0"]
        
        for lags in range(0, age + 1):
            exo_vars.append(f"L_{lags}_entry_rate_whole")
            exo_vars.append(f"L_{lags}_log_gdp")
        
        exo_vars.append(f"pre_cohort_log_restriction_2_0")
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
        v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
        coefs_age = coef_dict(v_names, res, coefs_age, age)

    # save results and figs
    model_name = f"{depend_var}_results_life_age_h_LP"
    v_names = ["cross", f"pre_cohort_log_restriction_2_0", "large_firm"]
    var_names = ["Cross", "Before Entering", "Large"]

    df_coefs_age = pd.DataFrame(coefs_age)
    df_coefs_age = df_coefs_age.sort_values(by=['name', 'age'])
    df_coefs_age.to_csv(Path.cwd()/results_tables_path/"key_results"/f"{model_name}.csv") 
    plot_lp(v_names, df_coefs_age, depend_var, model_name, fig_path, var_names, std)
                


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
                     'sector_2', depend_var, 'age_grp_dummy', 'firms']].dropna()
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
                            f'chg_restriction_2_0_{naics_curr}', f'log_restriction_2_{naics_curr}_pre_cohort',
                            'firms', 'sector_2', "log_emp_pre_cohort"]
        y_array = [depend_var]
        all_array = y_array + X_array
        
        data = data.loc[:, all_array].dropna()
        mod = PanelOLS.from_formula(formula = f'{depend_var} ~ chg_restriction_2_0_{naics_curr} + log_restriction_2_{naics_curr}_pre_cohort \
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
    variable_list.sort()
    
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
        
        #print(f"  - robust")
        #model_cohort_robust(config, variable)
    
if __name__ == "__main__":
    model_output()