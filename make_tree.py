import ROOT
from array import array
from keras.models import load_model
import itertools # to make all the combinations of two jets in the event

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

for idx,event in enumerate(t_in):
    # assign value to variables 
    # for now dummy values
    print type(event.jets_pt)
    print event.jets_pt



    mh1[0] = 125
    mh2[0] = 125+idx
    dRh1[0] = 0.2
    dRh2[0] = 0.4
    mhh[0] = 10
    # fill output tree
    t_out.Fill()

    if idx>10:
        break
    if idx%1000 == 0:
        print "Processed: ",idx

f_out.cd()
t_out.Write()
f_out.Close()

#f_in.Close()
