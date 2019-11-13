import uproot 
import pandas as pd
import numpy as np
import itertools # to make all the combinations of two jets in the event
import logging 
from sklearn.preprocessing import StandardScaler # to scale data
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

infile_name = 'files/output_397231_mc16a.root'
tree_name = 'GGMH_500'
# open file and get tree
logger.info(f'Opening: {infile_name}')
file = uproot.open(infile_name)
tree = file[tree_name]
# print(tree.keys())
# make DataFrame with original tree
df1 = tree.pandas.df(["pt_jet*",'jets_n_mini_tree']) 
df1['EventNum'] = df1.index
# print(df1.head())

# Make a new DataFrame with one row for each jet pair
logger.info('Creating DataFrame with one row for each jet pair')
rows_list = [] # list of rows of the new DataFrame
for index, row in df1.iterrows():    
    Njet = min(int(row['jets_n_mini_tree']),5)
    stuff = range(0,Njet)
    # consider all pairs of jets
    for test in itertools.combinations(stuff, 2):
        dict1 = {}
        i_A = test[0]+1 # +1 because jets are named starting from 1, not from 0
        i_B = test[1]+1
        dict1['pt_A'] = row['pt_jet_'+str(i_A)+'_mini_tree']
        dict1['pt_B'] = row['pt_jet_'+str(i_B)+'_mini_tree']
        dict1['EventNum'] = row['EventNum'] # keep track of the event number in the original dataframe
        dict1['i_A'] = i_A # keep track of index of jet A
        dict1['i_B'] = i_B # keep track of index of jet B
        dict1['is_good'] = int(np.random.uniform() > 0.5)
        rows_list.append(dict1)
# finally create the DataFrame from the list
df_new = pd.DataFrame(rows_list, columns=['EventNum','i_A','i_B','pt_A','pt_B','is_good'])  
#print(df_new.head(100))

logger.info('Converting DataFrame into np arrays')
X = df_new.iloc[:,:-1].values
y = df_new.iloc[:,-1].values

# scaling
logger.info('Scaling features')
sc = StandardScaler()
X = sc.fit_transform(X) 

# train and test splitting
logger.info('Train and test splitting')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5)

