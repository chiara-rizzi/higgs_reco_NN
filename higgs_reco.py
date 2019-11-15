import uproot 
import pandas as pd
import numpy as np
import itertools # to make all the combinations of two jets in the event
import logging 

from sklearn.preprocessing import StandardScaler # to scale data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score

# NN model
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import SGD #test keras

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 100)

train = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

infile_name = ['files/create_tree/output_397231_mc16a.root',
               'files/create_tree/output_397231_mc16d.root',
               'files/create_tree/output_397231_mc16e.root',
               'files/create_tree/output_397232_mc16a.root',
               'files/create_tree/output_397232_mc16d.root',
               'files/create_tree/output_397232_mc16e.root'
               ]
tree_name = ['GGMH_500',
             'GGMH_500',
             'GGMH_500',
             'GGMH_1100',
             'GGMH_1100',
             'GGMH_1100'
             ]
             
# open file and get tree
frames = []
for idx,f_name in enumerate(infile_name):    
    logger.info('Opening: '+f_name)
    file = uproot.open(f_name)
    tree = file[tree_name[idx]]
    # print(tree.keys())
    # make DataFrame with original tree
    df1 = tree.pandas.df(["pt_jet*",'eta_jet*','phi_jet*','e_jet*','is_b_jet*','is_from_h*','jets_n','hh_type']) 
    #df1 = df1[(df1['hh_type']==10) | (df1['hh_type']==11)]
    df1 = df1[df1['hh_type']==10]
    print(df1.shape)
    # print(df1.head())
    frames.append(df1)

logger.info('Concatenating DataFrames')        
df1 = pd.concat(frames)
print(df1.shape)
df1['EventNum'] = df1.index
print(df1.shape)

# Make a new DataFrame with one row for each jet pair
logger.info('Creating DataFrame with one row for each jet pair')
rows_list = [] # list of rows of the new DataFrame
for index, row in df1.iterrows():    
    Njet = min(int(row['jets_n']),8)
    stuff = range(0,Njet)
    # consider all pairs of jets
    for test in itertools.combinations(stuff, 2):
        dict1 = {}
        i_A = test[0]+1 # +1 because jets are named starting from 1, not from 0
        i_B = test[1]+1
        dict1['pt_A'] = row['pt_jet_'+str(i_A)]
        dict1['pt_B'] = row['pt_jet_'+str(i_B)]
        dict1['eta_A'] = row['eta_jet_'+str(i_A)]
        dict1['eta_B'] = row['eta_jet_'+str(i_B)]
        dict1['phi_A'] = row['phi_jet_'+str(i_A)]
        dict1['phi_B'] = row['phi_jet_'+str(i_B)]
        dict1['e_A'] = row['e_jet_'+str(i_A)]
        dict1['e_B'] = row['e_jet_'+str(i_B)]
        dict1['is_b_A'] = row['is_b_jet_'+str(i_A)]
        dict1['is_b_B'] = row['is_b_jet_'+str(i_B)]
        dict1['is_h1_A'] = row['is_from_h1_jet_'+str(i_A)]
        dict1['is_h1_B'] = row['is_from_h1_jet_'+str(i_B)]
        dict1['is_h2_A'] = row['is_from_h2_jet_'+str(i_A)]
        dict1['is_h2_B'] = row['is_from_h2_jet_'+str(i_B)]
        dict1['EventNum'] = row['EventNum'] # keep track of the event number in the original dataframe
        dict1['i_A'] = i_A # keep track of index of jet A
        dict1['i_B'] = i_B # keep track of index of jet B
        def is_good(is_h1_A, is_h1_B, is_h2_A, is_h2_B):
            if  is_h1_A>0.1:
                if is_h1_B>0.1:
                    return 1
            if is_h2_A>0.1 :
                if is_h2_B>0.1:
                    return 1
            return 0        
        dict1['is_good'] = is_good(dict1['is_h1_A'], dict1['is_h1_B'], dict1['is_h2_A'], dict1['is_h2_B'])
        rows_list.append(dict1)
# finally create the DataFrame from the list
df_new = pd.DataFrame(rows_list, columns=['EventNum','i_A','i_B','pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B','is_good'])  
#print(df_new.head(100))

df_good =  df_new[df_new['is_good']==1]
df_bad =  df_new[df_new['is_good']==0]

print('Describe')
print(df_new.describe())
print('Size good:',df_good.shape)
print('Size bad:',df_bad.shape)

logger.info('Converting DataFrame into np arrays')
X = df_new[['pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B']].values
y = df_new['is_good'].values

# scaling
logger.info('Scaling features')
sc = StandardScaler()
X = sc.fit_transform(X) 

if train:
    # train and test splitting
    logger.info('Train and test splitting')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
    # Neural network
    model = Sequential() # creating model sequentially (each layer takes as input output of previous layer)
    model.add(Dense(200, input_dim=10, activation='relu')) # Dense: fully connected layer
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(20, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) # chiara: check what's the best activation function for single-value output
    # loss function and optimizer
    #opt = SGD(lr=0.001)
    #model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # training 
    history = model.fit(X_train, y_train, epochs=50, batch_size=50, # it was 100 epochs
                        validation_data = (X_test,y_test)) # show accuracy on test data after every epoch
    
    # Prediction
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_train, tpr_train, label='train')
    plt.plot(fpr_test, tpr_test, label='validation')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    

    #a = accuracy_score(y_pred_test,y_test)
    #logger.info(f'Accuracy is: {a*100}')


    model.save('models/my_model.h5')  # creates a HDF5 file 'my_model.h5'

    """
    print(y_pred_test)
    print('y_train.size: ',y_train.size)
    print('y_test.size: ',y_test.size)
    print('y_pred_test.size: ',y_pred_test.size)
    print('y.size: ', y.size)
    print('X.size: ', X.size)
    print('df_new.shape: ', df_new.shape)
    """
    logger.info('Plotting accuracy')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    logger.info('Plotting loss function')
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()
# end if train

model = load_model('models/my_model.h5')
logger.info('Make column of prediction for DataFrame')
y_pred = model.predict(X) # value of predicted y on train set 
df_new['is_good_pred'] = y_pred

print(df_new.head(10))

df_new_chosen = df_new.loc[df_new.groupby('EventNum')['is_good_pred'].idxmax()]

#print(df_new_chosen.head())
