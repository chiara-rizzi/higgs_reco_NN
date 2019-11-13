import uproot 

infile_name = 'files/output_397231_mc16a.root'
tree_name = 'GGMH_500'

file = uproot.open(infile_name)
tree = file[tree_name]
print(tree.keys())
