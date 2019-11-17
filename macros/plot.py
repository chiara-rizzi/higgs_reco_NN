import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)
ROOT.gStyle.SetOptTitle(False)


infile_name = '/Users/crizzi/lavoro/SUSY/susy_EW/higgs_reco/files/higgs_mass/397231.mc16e.GGMH.root'
tree_name = 'nominal'
infile = ROOT.TFile.Open(infile_name)
t = infile.Get(tree_name)

variables = [
    ['mass_h1_dR','m_h1_new_min_dR','m_h1_NN_v1','m_h1_NN_v2','m_h1_true_match']    ,
    ['mass_h2_dR','m_h2_new_min_dR','m_h2_NN_v1','m_h2_NN_v2','m_h2_true_match']    
    ]

names = ['mh1','mh2']

bins = [40, 50, 250]

colors = [3,2,4,6,800]
styles = [1,2,3,45,6]

for ipl,plot in enumerate(variables):
    c = ROOT.TCanvas()
    hists = []
    leg =ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.SetLineStyle(0)
    leg.SetFillStyle(0)
    for var in plot:
        print var
        hists.append(ROOT.TH1F('h_'+var,'h_'+var,bins[0],bins[1],bins[2]))
        # t.Draw(var+'>>h_'+var, 'hh_type==10 && match_possible>0', 'goff')
        t.Draw(var+'>>h_'+var, 'hh_type==10', 'goff')

    max_hist = 0
    for idx,h in enumerate(hists):
        # add overflow and underflow
        h.SetBinContent(1, h.GetBinContent(1)+h.GetBinContent(0))
        h.SetBinContent(bins[0], h.GetBinContent(bins[0])+h.GetBinContent(bins[0]+1))
        if h.GetMaximum() > max_hist:
            max_hist = h.GetMaximum()        
        h.SetLineWidth(2)
        h.SetLineColor(colors[idx])
        h.SetLineStyle(styles[idx])
        leg.AddEntry(h,plot[idx])        
        c.cd()
    for idx,h in enumerate(hists):   
        h.SetMaximum(1.0*max_hist)
        if idx==0:
            h.GetXaxis().SetTitle('m [GeV]')
            h.Draw()
        else:
            h.Draw("same")
    leg.Draw()
    c.SaveAs(names[ipl]+'.pdf')
