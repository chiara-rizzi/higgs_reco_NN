import uproot 
import pandas as pd
import itertools # to make all the combinations of two jets in the event

infile_name = 'files/output_397231_mc16a.root'
tree_name = 'GGMH_500'

file = uproot.open(infile_name)
tree = file[tree_name]
# print(tree.keys())
df1 = tree.pandas.df(["pt_jet*",'jets_n_mini_tree']) 
df1['EventNum'] = df1.index
print(df1.head())
#columns_new = ['pt_jet_1','pt_jet_2']
#df2 = pd.DataFrame(columns=columns_new)
#print(df2.head())
rows_list = []
for index, row in df1.iterrows():
        dict1 = {}
        # get input row in dictionary format
        # key = col_name
        dict1['pt_1'] = -99
        dict1['pt_2'] = -99
        rows_list.append(dict1)

df_new = pd.DataFrame(rows_list)  
print(df_new.head())
