### Notes:
* higgs_reco.py requires keras and uproot --> conda base
* make_tree.py requires keas and ROOT --> conda py2root6


### Workflow
python higgs_reco.py --> train the model
python make_tree.py --> evaluate and builds higgs masses, makes root file
python add_friend.py --> add the HF ntuple as friend
python macros/plot.py --> plotting

 
