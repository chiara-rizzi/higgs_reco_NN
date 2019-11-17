import ROOT
from array import array
from keras.models import load_model
import itertools # to make all the combinations of two jets in the event
import pandas as pd
import numpy as np
import operator # to get the maximum in dictionary
import math

print "Chiara"

input_file_name = 'files/big_ntuples/user.cmorenom.397231.mc16e.GGMH.DAOD_SUSY10.e7112_e5984_a875_r10724_r10726_p3756.21.2.67-EWK-0_output_tree.root'
tree_name = 'nominal' 
f_in = ROOT.TFile.Open(input_file_name)
t_in = f_in.Get(tree_name)
print t_in.GetEntries()

output_file_name = 'files/higgs_mass/397231.mc16e.GGMH.root'
f_out = ROOT.TFile.Open(output_file_name,'RECREATE')
t_out = ROOT.TTree(tree_name, tree_name)

mh1_v1 = array( 'f', [ 0 ] )
mh2_v1 = array( 'f', [ 0 ] )
dRh1_v1 = array( 'f', [ 0 ] )
dRh2_v1 = array( 'f', [ 0 ] )
mhh_v1 = array( 'f', [ 0 ] )

mh1_v2 = array( 'f', [ 0 ] )
mh2_v2 = array( 'f', [ 0 ] )
dRh1_v2 = array( 'f', [ 0 ] )
dRh2_v2 = array( 'f', [ 0 ] )
mhh_v2 = array( 'f', [ 0 ] )

pt_j1 = array( 'f', [ 0 ])

t_out.Branch( 'm_h1_NN_v1', mh1_v1, 'm_h1_NN_v1/F' )
t_out.Branch( 'm_h2_NN_v1', mh2_v1, 'm_h2_NN_v1/F' )
t_out.Branch( 'dR_h1_NN_v1', dRh1_v1, 'dR_h1_NN_v1/F' )
t_out.Branch( 'dR_h2_NN_v1', dRh2_v1, 'dR_h2_NN_v1/F' )
t_out.Branch( 'm_hh_NN_v1', mhh_v1, 'm_hh_NN_v1/F' )

t_out.Branch( 'm_h1_NN_v2', mh1_v2, 'm_h1_NN_v2/F' )
t_out.Branch( 'm_h2_NN_v2', mh2_v2, 'm_h2_NN_v2/F' )
t_out.Branch( 'dR_h1_NN_v2', dRh1_v2, 'dR_h1_NN_v2/F' )
t_out.Branch( 'dR_h2_NN_v2', dRh2_v2, 'dR_h2_NN_v2/F' )
t_out.Branch( 'm_hh_NN_v2', mhh_v2, 'm_hh_NN_v2/F' )

t_out.Branch( 'pt_j1', pt_j1, 'pt_j1/F' )

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

def to_set (item):
    my_set = set( [val for sublist in item for val in sublist]  )
    return my_set

def choose_index_maxmin(df_new, Njet):
    stuff = range(0,Njet)
    combi2j = itertools.combinations(stuff, 2)
    combi2j_list = [i for i in combi2j]
    combi4j = itertools.combinations(combi2j_list, 2)
    combi4j_list = [i for i in combi4j if len(to_set(i))>3] # consider only combinations with 4 different jets
    pair_scores = {}
    for pair in combi2j_list:
        pair_scores[pair] = df_new[(df_new['i_A']==pair[0]) & (df_new['i_B']==pair[1])]['is_good_pred'].values[0]
        # print('pair_scores[pair]',pair_scores[pair])
    fourj_scores  = {}
    for fourj in combi4j_list:
        # print('pair_scores[fourj[0]]:',pair_scores[fourj[0]])
        # print('pair_scores[fourj[1]]:',pair_scores[fourj[1]])
        fourj_scores[fourj] = min( pair_scores[fourj[0]], pair_scores[fourj[1]] )
    fourj_selected = max(fourj_scores.iteritems(), key=operator.itemgetter(1))[0]
    return (fourj_selected[0], fourj_selected[1])

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
        dict1['run_number'] = event.run_number
        dict1['event_number'] = event.event_number
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
        def compute_dR(eta1, eta2, phi1, phi2):
            dEta = math.fabs(eta1 - eta2)
            dPhi = phi1 - phi2
            dPhi = math.fabs( (dPhi + math.pi) % (2*math.pi) - math.pi  )
            dR = math.sqrt( dEta*dEta + dPhi*dPhi  )
            return dR
        # dict1['dR'] = compute_dR(dict1['eta_A'], dict1['eta_B'], dict1['phi_A'], dict1['phi_B'])
        rows_list.append(dict1)

    df_new = pd.DataFrame(rows_list, columns=['run_number','event_number','pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B','i_A','i_B','dR'])
    # X = df_new[['pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B','dR']].values
    X = df_new[['pt_A','pt_B','eta_A','eta_B','phi_A','phi_B','is_b_A','is_b_B','e_A','e_B']].values
    y_pred = model.predict(X)

    # y_pred = np.random.rand(y_pred.size) # chiara: remove!! just for testing!!
    df_new['is_good_pred'] = y_pred
    
    # print df_new.sort_values(['is_good_pred'], ascending=[False]).head(30), '\n'
    
    """
    max_first = False
    if max_first:
        i_h1, i_h2 = choose_index(df_new)
    else:
        i_h1, i_h2 = choose_index_maxmin(df_new, Njet)
    """
    i_h1, i_h2 = choose_index(df_new)
    i_h1_b, i_h2_b = choose_index_maxmin(df_new, Njet)
    #  print "Original"
    #  print df_new
    #  print "Sorted"
    #  print df_new.sort_values(by=['is_good_pred'],ascending=False)
    #  print "Chosen indexes"
    #  print i_h1
    #  print i_h2
    
    # print "Index:", i_A,i_B
    # chiara: need  to check how to do the scaling with the same values as the training sample
    # X = sc.fit_transform(X)    
    def  compute_var(i_h1, i_h2):
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
        
        if h1.M()<h2.M():
            h_appo = h1
            h1 = h2
            h2 = h_appo
            
        return h1.M(), h2.M(), h1_b1.DeltaR(h1_b2), h2_b1.DeltaR(h2_b2), hh.M() 

    mh1_v1[0], mh2_v1[0], dRh1_v1[0], dRh2_v1[0], mhh_v1[0] = compute_var(i_h1, i_h2)
    mh1_v2[0], mh2_v2[0], dRh1_v2[0], dRh2_v2[0], mhh_v2[0] = compute_var(i_h1_b, i_h2_b)
    #print 'A: ', i_h1, i_h2, '  mh1:',mh1_v1[0], '  mh2:',mh2_v1[0] 


    pt_j1[0] = event.jets_pt[0]

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
