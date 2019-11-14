import ROOT


infile_name = '/Users/crizzi/lavoro/SUSY/susy_EW/higgs_reco/files/higgs_mass/397231.mc16e.GGMH.root'
tree_name = 'nominal'
infile = ROOT.TFile.Open(infile_name)
t = infile.Get(tree_name)

variables = [
    ['mass_h1_dR','m_h1_new_min_dR','m_h1_NN']    
    ]

bins = [40, 50, 250]

colors = [6,5,9,12]

for plot in variables:
    c = ROOT.TCanvas()
    hists = []
    leg =ROOT.TLegend()
    for var in plot:
        print var
        hists.append(ROOT.TH1F('h_'+var,'h_'+var,bins[0],bins[1],bins[2]))
        t.Draw(var+'>>h_'+var, 'hh_type==10', 'goff')

    for idx,h in enumerate(hists):
        print idx
        h.SetLineColor(colors[idx])
        leg.AddEntry(h,plot[idx])
        c.cd()
        if idx==0:
            h.Draw()
        else:
            h.Draw("same")
    leg.Draw()
    c.SaveAs('test.pdf')
