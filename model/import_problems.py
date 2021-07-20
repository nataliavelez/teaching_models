import pandas as pd

def df_from_csv(filename):
    '''
    given csv file from teaching_stimuli, turn each grid into array and assemble dataframe
    '''
    df = pd.read_csv(filename, header=1)
    
    # fix row and column names, get rid of the k values 
    df.fillna(method='ffill', inplace=True)
    df.columns = df.columns.to_series().mask(lambda x: x.str.startswith('Unnamed')).ffill()
    df = df.drop(columns=['Designer','k_A','k_B','k_C','k_D'])
    
    # set index labels
    df = df.set_index(['#'])
    
    # make new df, which has the images in matrix form 
    rows = map(int, df.index.unique().tolist())
    columns = df.columns.unique().tolist() # should be ['A', 'B', 'C', 'D']
    new_df = pd.DataFrame(columns = columns).astype(object)
    
    # fill new df
    for row in rows: 
        for column in columns: 
            new_df.loc[row, column] = df.loc[row, column].to_numpy()
    
    return new_df