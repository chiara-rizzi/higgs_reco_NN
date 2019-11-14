
import ROOT
from array import array
from keras.models import load_model
import itertools # to make all the combinations of two jets in the event
import pandas as pd
import numpy as np

print "Chiara"

input_file_name = 'files/big_ntuples/user.cmorenom.397231.mc16e.GGMH.DAOD_SUSY10.e7112_e5984_a875_r10724_r10726_p3756.21.2.67-EWK-0_output_tree.root'
tree_name = 'nominal' 
f_in = ROOT.TFile.Open(input_file_name)
t_in = f_in.Get(tree_name)
print t_in.GetEntries()

output_file_name = 'files/higgs_mass/397231.mc16e.GGMH.root'
f_out = ROOT.TFile.Open(output_file_name,'RECREATE')
t_out = ROOT.TTree(tree_name, tree_name)

mh1 = array( 'f', [ 0 ] )
mh2 = array( 'f', [ 0 ] )
dRh1 = array( 'f', [ 0 ] )
dRh2 = array( 'f', [ 0 ] )
mhh = array( 'f', [ 0 ] )

t_out.Branch( 'm_h1_NN', mh1, 'm_h1_NN/F' )
t_out.Branch( 'm_h2_NN', mh2, 'm_h2_NN/F' )
t_out.Branch( 'dR_h1_NN', dRh1, 'dR_h1_NN/F' )
t_out.Branch( 'dR_h2_NN', dRh2, 'dR_h2_NN/F' )
t_out.Branch( 'm_hh_NN', mhh, 'm_hh_NN/F' )

model = load_model('models/my_model.h5')

def return_pair(df_new, index):
    h = (int(df_new.iloc[index]['i_A']), int(df_new.iloc[index]['i_B']))
    if index == 0:
        return h
    h0 = return_pair(df_new, 0)
    if h[0] in h0 or h[1] in h0:
        return return_pair(df_new, index+1)
    else:
        return h

def choose_index(df_new):
    df_new = df_new.sort_values(by=['is_good_pred'],ascending=False)
    h1 = return_pair(df_new, 0)
    h2 = return_pair(df_new, 1)
    return (h1,h2)

for idx,event in enumerate(t_in):
    # assign value to variables 
    # for now dummy values
    # print type(event.jets_pt)
    # print event.jets_pt
    Njet = event.jets_pt.size()
    # print Njet
    rows_list = [] # list of rows of the new DataFrame
    stuff = range(0,Njet)
    for index in itertools.combinations(stuff, 2):
        dict1 = {}
        dict1['pt_A'] = event.jets_pt[index[0]]
        dict1['pt_B'] = event.jets_pt[index[1]]
        dict1['eta_A'] = event.jets_eta[index[0]]
        dict1['eta_B'] = event.jets_eta[index[1]]
        dict1['phi_A'] = event.jets_phi[index[0]]
        dict1['phi_B'] = event.jets_phi[index[1]]
        dict1['e_A'] = event.jets_e[index[0]]
        dict1['e_B'] = event.jets_e[index[1]]
        dict1['is_b_A'] = event.jets_isb_FixedCutBEff_77[index[0]]
        dict1['is_b_B'] = event.jets_isb_FixedCutBEff_77[index[1]]
        dict1['i_A'] = index[0]
        dict1['i_B'] = index[1]
        rows_list.append(dict1)
    df_new = pd.DataFrame(rows_list, columns=['pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B','i_A','i_B'])
    X = df_new[['pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B']].values
    y_pred = model.predict(X)

    y_pred = np.random.rand(y_pred.size) # chiara: remove!! just for testing!!
    df_new['is_good_pred'] = y_pred
    
    i_h1, i_h2 = choose_index(df_new)

    # print "Original"
    #  print df_new
    #  print "Sorted"
    #  print df_new.sort_values(by=['is_good_pred'],ascending=False)
    #  print "Chosen indexes"
    #  print i_h1
    #  print i_h2
    
    # print "Index:", i_A,i_B
    # chiara: need  to check how to do the scaling with the same values as the training sample
    # X = sc.fit_transform(X)    
    h1_b1 = ROOT.TLorentzVector()
    h1_b2 = ROOT.TLorentzVector()
    h2_b1 = ROOT.TLorentzVector()
    h2_b2 = ROOT.TLorentzVector()

    h1_b1.SetPtEtaPhiE(event.jets_pt[i_h1[0]], event.jets_eta[i_h1[0]], event.jets_phi[i_h1[0]], event.jets_e[i_h1[0]])
    h1_b2.SetPtEtaPhiE(event.jets_pt[i_h1[1]], event.jets_eta[i_h1[1]], event.jets_phi[i_h1[1]], event.jets_e[i_h1[1]])
    h2_b1.SetPtEtaPhiE(event.jets_pt[i_h2[0]], event.jets_eta[i_h2[0]], event.jets_phi[i_h2[0]], event.jets_e[i_h2[0]])
    h2_b2.SetPtEtaPhiE(event.jets_pt[i_h2[1]], event.jets_eta[i_h2[1]], event.jets_phi[i_h2[1]], event.jets_e[i_h2[1]])

    h1 = h1_b1 + h1_b2
    h2 = h2_b1 + h2_b2
    hh = h1 + h2

    mh1[0] = h1.M()
    mh2[0] = h2.M()
    dRh1[0] = h1_b1.DeltaR(h1_b2)
    dRh2[0] = h2_b1.DeltaR(h2_b2)
    mhh[0] = hh.M()

    #  print "mh1:", mh1[0]
    #  print "mh2:", mh2[0]
    #  print "dRh1:", dRh1[0]
    #  print "dRh2:", dRh2[0]
    #  print " "

    # fill output tree
    t_out.Fill()

    if idx%1000 == 0:
        print "Processed: ",idx

f_out.cd()
t_out.Write()
f_out.Close()

#f_in.Close()
