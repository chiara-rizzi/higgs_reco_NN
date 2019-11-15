import ROOT


infile_name = '/Users/crizzi/lavoro/SUSY/susy_EW/higgs_reco/files/higgs_mass/397231.mc16e.GGMH.root'
tree_name = 'nominal'
infile = ROOT.TFile.Open(infile_name)
t = infile.Get(tree_name)

variables = [
    ['mass_h1_dR','m_h1_new_min_dR','m_h1_NN','m_h1_true_match']    ,
    ['mass_h2_dR','m_h2_new_min_dR','m_h2_NN','m_h2_true_match']    
    ]

names = ['mh1','mh2']

bins = [40, 50, 250]

colors = [3,2,4,6,800]
styles = [1,2,3,45,6]

for ipl,plot in enumerate(variables):
    c = ROOT.TCanvas()
    hists = []
    leg =ROOT.TLegend()
    for var in plot:
        print var
        hists.append(ROOT.TH1F('h_'+var,'h_'+var,bins[0],bins[1],bins[2]))
        t.Draw(var+'>>h_'+var, 'hh_type==10', 'goff')

    for idx,h in enumerate(hists):
        print idx
        # add overflow and underflow
        h.SetBinContent(1, h.GetBinContent(1)+h.GetBinContent(0))
        h.SetBinContent(bins[0], h.GetBinContent(bins[0])+h.GetBinContent(bins[0]+1))
        h.SetLineWidth(2)
        h.SetLineColor(colors[idx])
        h.SetLineStyle(styles[idx])
        leg.AddEntry(h,plot[idx])
        c.cd()
        if idx==0:
            h.Draw()
        else:
            h.Draw("same")
    leg.Draw()
    c.SaveAs(names[ipl]+'.pdf')
