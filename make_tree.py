import ROOT

print "Chiara"

input_file_name = 'files/big_ntuples/user.cmorenom.397231.mc16e.GGMH.DAOD_SUSY10.e7112_e5984_a875_r10724_r10726_p3756.21.2.67-EWK-0_output_tree.root'
tree_name = 'nominal' 
f_in = ROOT.TFile.Open(input_file_name)
t_in = f_in.Get(tree_name)
print t_in.GetEntries()

output_file_name = 'files/higgs_mass/397231.mc16e.GGMH.root'
f_out = ROOT.TFile.Open(output_file_name,"RECREATE")
t_out = ROOT.TTree(tree_name, tree_name)

mh1 = array( 'f', [ 0 ] )
mh2 = array( 'f', [ 0 ] )
dRh1 = array( 'f', [ 0 ] )
dRh2 = array( 'f', [ 0 ] )
mhh = array( 'f', [ 0 ] )

for event in t_in:
    print event.jets_pt[1]


