import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl

mpl.use('Agg')


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)   
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)  
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):      
        h = self.conv1(graph, inputs)  
        h = {k: F.relu(v) for k, v in h.items()} 
        h = self.conv2(graph, h)    
        h = {k: F.relu(v) for k, v in h.items()}
        return h

class HeteroClassifier(nn.Module):  
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names) 
        self.classify = nn.Linear(hidden_dim, n_classes)   
    def forward(self, g):
        h = g.ndata['feat']  
     
        h = self.rgcn(g, h)  
        with g.local_scope():  
            g.ndata['hfeat'] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'hfeat', ntype=ntype)   
            return self.classify(hg)

