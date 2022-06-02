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
