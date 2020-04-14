import pandas as pd
import os as os
from itertools import product

def generate_hccs(df, version):
    """ Generate Hierarchical Condition Codes (HCCs) for unique recipients from a dataframe of 
    icd9 or icd10 codes. 

    Args:
      df: pandas.DataFrame. Contains 4 columns
        1) recip_id: a unique identifier for each person.
        2) icd: string, diagnosis code (either icd9 or icd10).
        3) version: int {9, 10}, whether the diagnosis code is icd9 or icd10.
        4) date: datetime64, date of the diagnosis
      version: string {'v12', 'v21', 'v22'}
        Determines which versin of CC codes and hierarchies to use.

    Returns:
      pandas.DataFrame with one row per unique recipient and one column per HCC. All DataFrame
        values are True or False for whether that HCC applies to the recipient.
    """
    # Determine the necessary crosswalks, hierarchy and CC list based on the version
    crosswalk_list = [file for file in os.listdir('Crosswalks') if version in file]
    hierarchy = 'ConditionCategory/'+version+'_rules.csv'
    cc_list = 'ConditionCategory/'+version+'_labels.csv'

    # Create icd to cc mapping DataFrame, and read in the rules and list DataFrames.
    lst = []
    for file in crosswalk_list:
        lst.append(pd.read_csv('Crosswalks/'+file))
    df_map = pd.concat(lst)
    df_hier = pd.read_csv(hierarchy)
    df_list = pd.read_csv(cc_list)

    # Bring CCs to the input DataFrame based on diagnosis codes.
    # Drop all helper columns that are no longer useful after the merge.
    df['year'] = df['claim_date'].dt.year

    merged = (df.merge(df_map, on=['diag_code', 'year', 'version'], how='left')
                .drop(columns=['claim_date', 'year']))
    
    # Keep only the subset that was mapped to a CC.
    merged = merged[merged.cc.notnull()]

    # Now convert this to a truth table for whether a CC exists for a recipient
    # Uses the FULL list of CCs as the index.
    merged = (merged.groupby(['recip_id', 'cc'])
                    .size().unstack(fill_value=0)
                    .reindex(df_list.cc, axis=1, fill_value=0).astype(bool))

    # Apply heirarchies. For hierarchical codes, if the column in merged is True, 
    # set the appropriate other column to False. 
    for index, row in df_hier.iterrows():
        merged.loc[merged[row.cc] == True, row.to_zero] = False

    return merged


def extract_hierachy_rules(version_list):
    """ Extract hierarchy rules from SAS scripts into a formatted csv.

    Args:
      version_list: list or strings
        List of all raw hierarchy SAS data. One file for each Condition Category version.
        version_list = ['2010/v12/V12H70H.txt', '2015/v21/V20H87H1.txt', '2017/v22/V22H79H1.txt']

    Output:
      Saves formatted csv to 'ConditionCategory/' with the suffix _rules.csv

    """
    for file in version_list:
        # Determine the version, read in the data.
        _, cc_version, _ = file.split('/')
        df_logic = pd.read_table('Raw/' + file, header=None).rename(columns={0:'text'})

        if cc_version == 'v12':
            # Find the logical if statement in the hierarchy.
            df = pd.DataFrame(df_logic.text.str.split('if h', 1).tolist(), 
                              columns=['junk', 'logic'])

            # Locate the HCC that begins the if statement.
            df['condition'] = df['logic'].str.extract('cc(<?[0-9]*)', expand=True)

            # Find all HCCs to zero based on above 'condition' HCC.
            df['zeros'] = df['logic'].str.extract('(?<=i=)([\d]{1,3}(,\s?[\d]{1,3})*)', 
                                                  expand=True)[0]

        elif cc_version == 'v21' or cc_version == 'v22':
            # Find the logical if statement in the hierarchy.
            df = pd.DataFrame(df_logic.text.str.split(r'%SET0\(', 1).tolist(), 
                              columns=['junk', 'logic'])

            # Locate the HCC that begins the if statement.
            df['condition'] = df['logic'].str.extract('CC=(<?[0-9]*)', expand=True)

            # Find all HCCs to zero based on above 'condition' HCC.
            df['zeros'] = df['logic'].str.extract('(?<=%STR\()([\d]{1,3}(\s?,\s?[\d]{1,3})*)', 
                                                  expand=True)[0]

        # Eplode the above list to a long DataFrame named rules. Rules contains one row per HCC we
        # need to zero.
        smalldf = df.loc[(df.condition.notnull()) & (df.zeros.notnull()), ['zeros', 'condition']]
        rules = (pd.concat([pd.Series(row['condition'], row['zeros'].split(',')) 
                           for _, row in  smalldf.iterrows()])
                   .reset_index()
                   .rename(columns={'index': 'to_zero', 0:'cc'}))

        # Save exploded list so we can use that in the future. 
        rules.to_csv('ConditionCategory/' + cc_version+'_rules.csv', sep=',', index=False)


def extract_cc_table(version_list):
    """ Extract a list of all Conditional Categories and labels for a given version. 

    Args:
      version_list: list or strings
        List of all raw hierarchy SAS data. One file for each Condition Category version.
        version_list =  ['2010/v12/V12H70L.txt','2015/v21/V20H87L1.txt','2017/v22/V22H79L1.txt']

    Output:
      Saves formatted csv to 'ConditionCategory/' with the suffix _rules.csv

    """
    for file in version_list:
        _, cc_version, _ = file.split('/')

        df = pd.read_table('Raw/'+file, header=None).rename(columns={0:'text'})

        # Extract the CC number.
        df['cc'] = df['text'].str.extract('\s?HCC([\d]{1,3})', expand=True)[0]

        #Extract the label
        if cc_version == 'v12':
            df['label'] = df['text'].str.extract('\'(.+?)\s?\'', expand=True)[0]
        elif cc_version == 'v21' or cc_version == 'v22':
            # Extract the label.
            df['label'] = df['text'].str.extract('\"(.+?)\"', expand=True)[0]

        #Save only the subset of rows and columns that atually correspond to a cc and label.
        (df.loc[df.cc.notnull() & df.label.notnull(), ['cc', 'label']]
           .to_csv('ConditionCategory/' + cc_version+'_labels.csv', sep=',', index=False))


def format_crosswaks(icd9_list=None, icd10_list=None):
    """ Formats crosswalks into an easily readable csv with additional information about
    icd version and year. 

    ***If additional mappings are needed outside of 2009-2017, make sure any
    CCs that need to be manually appended to the crosswalks (found in the SAS macros ending in
    M, i.e. V12H70M) are still handled properly, or change the logic loop below that manually
    adds them.

    Args:
      icd9_list: list of strings
        List of all raw icd9 mappings to transform.
        icd9_list = ['2009/v12/F1209F1Y.txt', '2010/v12/F1210F1Y.txt', '2011/v12/F1210F1Y.txt', 
             '2012/v12/F1212H1Y.txt', '2012/v21/F2112H1R.txt', '2013/v12/F1213H1Y.txt',
             '2013/v21/F2113J1R.txt', '2014/v12/F1213H1Y.txt', '2014/v21/F2113J1R.txt',
             '2014/v22/F2213L2P.txt', '2015/v12/F1213H1Y.txt', '2015/v21/F2113J1R.txt',
             '2015/v22/F2213L2P.txt', '2016/v21/F211690R_ICD9.txt', '2016/v22/F221690P_ICD9.txt']
      icd10_list: list of strings
        List of all raw icd10 mappings to transform.
        icd10_list = ['2016/v21/F211690R_ICD10.txt', '2016/v22/F221690P_ICD10.txt', 
              '2017/v21/F2117H1R.txt', '2017/v22/F2217O1P.txt']

    Output: 
      Saves formatted csv to "Crosswalks/"
      
    """
    if icd9_list is None:
        icd9_list = []
    if icd10_list is None:
        icd10_list = []

    # Clean all icd9 crosswalks.
    for file in icd9_list:
        # Input data is horribly formatted, so this works to resolve columns with dangling Ds.
        year, cc_version, name = file.split('/')

        df = pd.read_table('Raw/'+file, header=None)
        df['icd'] = [item[0] for item in df[0].str.split('\s+')]
        df['cc'] = [item[1] for item in df[0].str.split('\s+')]
        
        # Add information on the version and year.
        df['version'] = 9
        df['year'] = int(year)

        # No longer need this column
        df = df.drop(columns=0)

        # Need to manually append the additional CC mappings found in the VXXXXXM files.
        if cc_version == 'v12':
            dictionary1 = {'year': [int(year)], 
                           'icd': ['40403', '40413', '40493'], 
                           'version': [9], 
                           'cc': [80]}
            df_extra = pd.DataFrame([row for row in product(*dictionary1.values())], 
                                    columns=dictionary1.keys())

        elif cc_version == 'v21' and int(year) <= 2015:
            dictionary2 = {'year': [int(year)],
                           'icd': ['3572', '36202'], 
                           'version': [9], 
                           'cc': [18]}
            dictionary3 = {'year': [int(year)],
                           'icd': ['40401', '40403', '40411', '40413', '40491', '40493'],
                           'version': [9], 
                           'cc': [85]}
            df_extra = pd.DataFrame([row for row in product(*dictionary2.values())], 
                                    columns=dictionary2.keys())
            df_extra = df_extra.append(pd.DataFrame([row for row in product(*dictionary3.values())], 
                                                    columns=dictionary3.keys()))

        elif cc_version == 'v22' and int(year) <= 2015:
            df_extra = pd.DataFrame({'year': [int(year)], 'icd': ['36202'], 'version': [9],
                                     'cc': [18]})
            dictionary4 = {'year': [int(year)], 
                'icd': ['40403', '40413', '40493'], 'version': [9], 'cc': [85]}
            df_extra = df_extra.append(pd.DataFrame([row for row in product(*dictionary4.values())], 
                                                    columns=dictionary4.keys()))

        df = df.append(df_extra)

        # Save DataFrame so we don't need to re-run this.
        df.to_csv('Crosswalks/'+year+'_'+cc_version+'_icd9.csv', sep=',', index=False)

    # Clean up all icd10s in the same manner. ICD 10s have no additional mappings, as they already
    # appear in the raw map files with a 'D' in the row. 
    for file in icd10_list:
        year, cc_version, name = file.split('/')
        df = pd.read_table('Raw/'+file, header=None)
        df['icd'] = [item[0] for item in df[0].str.split('\s+')]
        df['cc'] = [item[1] for item in df[0].str.split('\s+')]
        df['version'] = 10
        df['year'] = int(year)
        df = df.drop(columns=0)
        df.to_csv('Crosswalks/'+year+'_'+cc_version+'_icd10.csv', sep=',', index=False)
