import dgl
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from utils import get_ntype_hetero_nids_to_homo_nids, get_homo_nids_to_ntype_hetero_nids, get_ntype_pairs_to_cannonical_etypes
from utils import hetero_src_tgt_khop_in_subgraph, get_neg_path_score_func, k_shortest_paths_with_max_length
from utils import power_link_p_on, get_inverse_score_func

def get_edge_mask_dict(ghetero):
    '''
    Create a dictionary mapping etypes to learnable edge masks 
            
    Parameters
    ----------
    ghetero : heterogeneous dgl graph.

    Return
    ----------
    edge_mask_dict : dictionary
        key=etype, value=torch.nn.Parameter with size number of etype edges
    '''
    device = ghetero.device
    edge_mask_dict = {}
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()

        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)
    return edge_mask_dict

def remove_edges_of_high_degree_nodes(ghomo, max_degree=10, always_preserve=[]):
    '''
    For all the nodes with degree higher than `max_degree`, 
    except nodes in `always_preserve`, remove their edges. 
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    max_degree : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    low_degree_ghomo : dgl homogeneous graph
        Pruned graph with edges of high degree nodes removed

    '''
    d = ghomo.in_degrees()
    high_degree_mask = d > max_degree
    
    # preserve nodes
    high_degree_mask[always_preserve] = False    

    high_degree_nids = ghomo.nodes()[high_degree_mask]
    u, v = ghomo.edges()
    high_degree_edge_mask = torch.isin(u, high_degree_nids) | torch.isin(v, high_degree_nids)
    high_degree_u, high_degree_v = u[high_degree_edge_mask], v[high_degree_edge_mask]
    high_degree_eids = ghomo.edge_ids(high_degree_u, high_degree_v)
    low_degree_ghomo = dgl.remove_edges(ghomo, high_degree_eids)
    
    return low_degree_ghomo


def remove_edges_except_k_core_graph(ghomo, k, always_preserve=[]):
    '''
    Find the `k`-core of `ghomo`.
    Only isolate the low degree nodes by removing theirs edges
    instead of removing the nodes, so node ids can be kept.
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    k : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    k_core_ghomo : dgl homogeneous graph
        The k-core graph
    '''
    k_core_ghomo = ghomo
    degrees = k_core_ghomo.in_degrees()
    k_core_mask = (degrees > 0) & (degrees < k)
    k_core_mask[always_preserve] = False
    
    while k_core_mask.any():
        k_core_nids = k_core_ghomo.nodes()[k_core_mask]
        
        u, v = k_core_ghomo.edges()
        k_core_edge_mask = torch.isin(u, k_core_nids) | torch.isin(v, k_core_nids)
        k_core_u, k_core_v = u[k_core_edge_mask], v[k_core_edge_mask]
        k_core_eids = k_core_ghomo.edge_ids(k_core_u, k_core_v)

        k_core_ghomo = dgl.remove_edges(k_core_ghomo, k_core_eids)
        
        degrees = k_core_ghomo.in_degrees()
        k_core_mask = (degrees > 0) & (degrees < k)
        k_core_mask[always_preserve] = False

    return k_core_ghomo

def get_eids_on_paths(paths, ghomo):
    '''
    Collect all edge ids on the paths
    
    Note: The current version is a list version. An edge may be collected multiple times
    A different version is a set version where an edge can only contribute one time 
    even it appears in multiple paths
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    Returns
    -------
    paths: list of lists
        Each list contains (source node ids, target node ids)
        
    '''
    row, col = ghomo.edges()
    eids = []
    for path in paths:
        for i in range(len(path)-1):
            eids += [((row == path[i]) & (col == path[i+1])).nonzero().item()]
            
    return torch.LongTensor(eids)

def comp_g_paths_to_paths(comp_g, comp_g_paths):
    paths = []
    g_nids = comp_g.ndata[dgl.NID]
    for comp_g_path in comp_g_paths:
        path = []
        for can_etype, u, v in comp_g_path:
            u_ntype, _, v_ntype = can_etype
            path += [(can_etype, g_nids[u_ntype][u].item(), g_nids[v_ntype][v].item())]
        paths += [path]
    return paths


class TripletEdgeScorer(nn.Module):
    """Triplet Edge Scorer (TES), Power-Link Eq 1-3 (arXiv:2401.02290).

    Scores every edge (h_i, r_j, t_k) in the computation graph by an MLP
    over the concatenation of its own (frozen, from the trained KGC/GNN
    encoder) triplet embedding with the target triplet (h_hat, r_hat,
    t_hat) being explained -- the paper's "concatenation" combination
    strategy (Eq 2), used in all of its main experiments. TES is retrained
    from scratch for each explained instance via backprop, same as the
    edge mask it replaces (Algorithm 1).

    This graph has no native per-relation embedding (its GNN encoder only
    has per-canonical-etype message-passing weights, not a relation
    embedding table like TransE/DistMult/ConvE in the paper's KGC setting),
    so `rel_emb` is a small learnable table trained jointly with the MLP.
    """
    def __init__(self, emb_dim, num_etypes, hidden_dim=64):
        super().__init__()
        self.rel_emb = nn.Embedding(num_etypes, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_i, etype_ids, t_k, h_hat, r_hat_id, t_hat):
        """
        h_i, t_k : (E, emb_dim) local edge triplet endpoint embeddings
        etype_ids : (E,) local edge triplet relation ids
        h_hat, t_hat : (1, emb_dim) target triplet endpoint embeddings
        r_hat_id : scalar tensor, target triplet relation id

        Returns
        -------
        (E,) raw score logits (apply .sigmoid() to get the TES probability)
        """
        e = h_i.shape[0]
        r_j = self.rel_emb(etype_ids)
        r_hat = self.rel_emb(r_hat_id).expand(e, -1)
        x = torch.cat([h_i, r_j, t_k, h_hat.expand(e, -1), r_hat, t_hat.expand(e, -1)], dim=-1)
        return self.mlp(x).squeeze(-1)


class PaGELink(nn.Module):
    """Path-based GNN Explanation for Heterogeneous Link Prediction (PaGELink)
    
    Some methods are adapted from the DGL GNNExplainer implementation
    https://docs.dgl.ai/en/0.8.x/_modules/dgl/nn/pytorch/explain/gnnexplainer.html#GNNExplainer
    
    Parameters
    ----------
    model : nn.Module
        The GNN-based link prediction model to explain.

        * The required arguments of its forward function are source node id, target node id,
          graph, and feature ids. The feature ids are for selecting input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by the weights during message passing.
        * The output of its forward function is the logits in (-inf, inf) for the 
          predicted link.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more discrete by decreasing
        the entropy of the edge mask.
    alpha : float, optional
        A higher value will make edges on high-quality paths to have higher weights
    beta : float, optional
        A higher value will make edges off high-quality paths to have lower weights
    log : bool, optional
        If True, it will log the computation process, default to True.
    """
    def __init__(self,
                 model,
                 lr=0.001,
                 num_epochs=100,
                 alpha=1.0,
                 beta=1.0,
                 log=False,
                 path_method='dijkstra',
                 pred_etype='likes',
                 gamma=1e-3):
        super(PaGELink, self).__init__()
        self.model = model
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype

        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.log = log

        assert path_method in ('dijkstra', 'power'), \
            "path_method must be 'dijkstra' (original PaGE-Link) or " \
            "'power' (Power-Link, arXiv:2401.02290: TES + graph-powering path loss)"
        self.path_method = path_method
        # the relation type being predicted/explained -- Power-Link's target
        # triplet relation r_hat (Sec 5.1); also used by `path_loss`'s
        # power-method branch, set per-call in `explain`
        self.pred_etype = pred_etype
        # weight of the ||M||^2 regularizer on the TES score matrix (Eq 12)
        self.gamma = gamma
        self._path_loss_max_length = 5

        self.all_loss = defaultdict(list)

    def _init_masks(self, ghetero):
        """Initialize the learnable edge mask.

        Parameters
        ----------
        graph : DGLGraph
            Input graph.

        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        return get_edge_mask_dict(ghetero)
    

    def _prune_graph(self, ghetero, prune_max_degree=-1, k_core=2, always_preserve=[]):
        # Prune edges by (optionally) removing edges of high degree nodes and extracting k-core
        # The pruning is computed on the homogeneous graph, i.e., ignoring node/edge types
        ghomo = dgl.to_homogeneous(ghetero)
        device = ghetero.device
        ghomo.edata['eid_before_prune'] = torch.arange(ghomo.num_edges()).to(device)
        
        if prune_max_degree > 0:
            max_degree_pruned_ghomo = remove_edges_of_high_degree_nodes(ghomo, prune_max_degree, always_preserve)
            k_core_ghomo = remove_edges_except_k_core_graph(max_degree_pruned_ghomo, k_core, always_preserve)
            
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = max_degree_pruned_ghomo
            else:
                pruned_ghomo = k_core_ghomo
        else:
            k_core_ghomo = remove_edges_except_k_core_graph(ghomo, k_core, always_preserve)
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = ghomo
            else:
                pruned_ghomo = k_core_ghomo
        
        pruned_ghomo_eids = pruned_ghomo.edata['eid_before_prune']
        pruned_ghomo_eid_mask = torch.zeros(ghomo.num_edges()).bool()
        pruned_ghomo_eid_mask[pruned_ghomo_eids] = True

        # Apply the pruning result on the heterogeneous graph
        etypes_to_pruned_ghetero_eid_masks = {}
        pruned_ghetero = ghetero
        cum_num_edges = 0
        for etype in ghetero.canonical_etypes:
            num_edges = ghetero.num_edges(etype=etype)
            pruned_ghetero_eid_mask = pruned_ghomo_eid_mask[cum_num_edges:cum_num_edges+num_edges]
            etypes_to_pruned_ghetero_eid_masks[etype] = pruned_ghetero_eid_mask

            remove_ghetero_eids = (~ pruned_ghetero_eid_mask).nonzero().view(-1).to(device)
            pruned_ghetero = dgl.remove_edges(pruned_ghetero, eids=remove_ghetero_eids, etype=etype)

            cum_num_edges += num_edges
                
        return pruned_ghetero, etypes_to_pruned_ghetero_eid_masks
        
        
    def path_loss(self, src_nid, tgt_nid, g, eweights, num_paths=5):
        """Compute the (original PaGE-Link) path loss for the 'dijkstra'
        baseline: find k-shortest paths under the current mask, then push
        their edge weights up and all other edges' weights down.

        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        g : dgl graph

        eweights : Tensor
            Edge weights with shape equals the number of edges.

        num_paths : int
            Number of paths to compute path loss on

        Returns
        -------
        loss : Tensor
            The path loss
        """
        neg_path_score_func = get_neg_path_score_func(g, 'eweight', [src_nid, tgt_nid])
        paths = k_shortest_paths_with_max_length(g,
                                                 src_nid,
                                                 tgt_nid,
                                                 weight=neg_path_score_func,
                                                 k=num_paths)

        eids_on_path = get_eids_on_paths(paths, g)

        if eids_on_path.nelement() > 0:
            loss_on_path = - eweights[eids_on_path].mean()
        else:
            loss_on_path = 0

        eids_off_path_mask = ~torch.isin(torch.arange(eweights.shape[0]), eids_on_path)
        if eids_off_path_mask.any():
            loss_off_path = eweights[eids_off_path_mask].mean()
        else:
            loss_off_path = 0

        loss = self.alpha * loss_on_path + self.beta * loss_off_path 

        self.all_loss['loss_on_path'] += [float(loss_on_path)]
        self.all_loss['loss_off_path'] += [float(loss_off_path)]

        return loss   

    
    def get_edge_mask(self, 
                      src_nid, 
                      tgt_nid, 
                      ghetero, 
                      feat_nids, 
                      prune_max_degree=-1,
                      k_core=2, 
                      prune_graph=True,
                      with_path_loss=True):

        """Learning the edge mask dict.   
        
        Parameters
        ----------
        see the `explain` method.
        
        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """

        self.model.eval()
        device = ghetero.device

        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]

        # Get the initial prediction.
        with torch.no_grad():
            score = self.model(src_nid, tgt_nid, ghetero, feat_nids)
            pred = (score > 0).int().item()

        if prune_graph:
            # The pruned graph for mask learning
            ml_ghetero, etypes_to_pruned_ghetero_eid_masks = self._prune_graph(ghetero,
                                                                               prune_max_degree,
                                                                               k_core,
                                                                               [homo_src_nid, homo_tgt_nid])
        else:
            # The original graph for mask learning
            ml_ghetero = ghetero

        if self.path_method == 'power':
            ml_edge_mask_dict = self._train_tes_power(src_nid, tgt_nid, ml_ghetero, feat_nids,
                                                       pred, homo_src_nid, homo_tgt_nid, with_path_loss)
        else:
            ml_edge_mask_dict = self._train_mask_dijkstra(src_nid, tgt_nid, ml_ghetero, feat_nids,
                                                           pred, homo_src_nid, homo_tgt_nid, with_path_loss)

        edge_mask_dict_placeholder = self._init_masks(ghetero)
        edge_mask_dict = {}

        if prune_graph:
            # remove pruned edges
            for etype in ghetero.canonical_etypes:
                edge_mask = edge_mask_dict_placeholder[etype].data + float('-inf')
                pruned_ghetero_eid_mask = etypes_to_pruned_ghetero_eid_masks[etype]
                edge_mask[pruned_ghetero_eid_mask] = ml_edge_mask_dict[etype]
                edge_mask_dict[etype] = edge_mask

        else:
            edge_mask_dict = ml_edge_mask_dict

        edge_mask_dict = {k : v.detach() for k, v in edge_mask_dict.items()}
        return edge_mask_dict

    def _train_mask_dijkstra(self, src_nid, tgt_nid, ml_ghetero, feat_nids,
                              pred, homo_src_nid, homo_tgt_nid, with_path_loss):
        """Original PaGE-Link mask learning: a free scalar per edge, path
        loss found via Dijkstra/Yen's k-shortest-paths every epoch."""
        ml_edge_mask_dict = self._init_masks(ml_ghetero)
        optimizer = torch.optim.Adam(ml_edge_mask_dict.values(), lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)

        eweight_norm = 0
        EPS = 1e-3
        for e in range(self.num_epochs):

            # Apply sigmoid to edge_mask to get eweight
            ml_eweight_dict = {etype: ml_edge_mask_dict[etype].sigmoid() for etype in ml_edge_mask_dict}

            score = self.model(src_nid, tgt_nid, ml_ghetero, feat_nids, ml_eweight_dict)
            pred_loss = (-1) ** pred * score.sigmoid().log()
            self.all_loss['pred_loss'] += [pred_loss.item()]

            ml_ghetero.edata['eweight'] = ml_eweight_dict
            ml_ghomo = dgl.to_homogeneous(ml_ghetero, edata=['eweight'])
            ml_ghomo_eweights = ml_ghomo.edata['eweight']

            # Check for early stop
            curr_eweight_norm = ml_ghomo_eweights.norm()
            if abs(eweight_norm - curr_eweight_norm) < EPS:
                break
            eweight_norm = curr_eweight_norm

            # Update with path loss
            if with_path_loss:
                path_loss = self.path_loss(homo_src_nid, homo_tgt_nid, ml_ghomo, ml_ghomo_eweights)
            else:
                path_loss = 0

            loss = pred_loss + path_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        return ml_edge_mask_dict

    def _frozen_homo_node_emb(self, ghetero, feat_nids):
        """Frozen entity embeddings from the trained (post-hoc, not
        modified) KGC/GNN encoder Phi, reindexed onto the homogeneous
        node numbering -- the entity half of the TES's Combine() input
        (Eq 2/3)."""
        with torch.no_grad():
            emb_dict = self.model.encode(ghetero, feat_nids)
        ghomo = dgl.to_homogeneous(ghetero)
        ntype_ids = ghomo.ndata[dgl.NTYPE]
        hetero_nids = ghomo.ndata[dgl.NID]
        emb_dim = next(iter(emb_dict.values())).shape[1]
        homo_emb = torch.zeros(ghomo.num_nodes(), emb_dim, device=ghetero.device)
        for i, ntype in enumerate(ghetero.ntypes):
            mask = ntype_ids == i
            if mask.any():
                homo_emb[mask] = emb_dict[ntype][hetero_nids[mask]]
        return ghomo, homo_emb

    def _train_tes_power(self, src_nid, tgt_nid, ml_ghetero, feat_nids,
                          pred, homo_src_nid, homo_tgt_nid, with_path_loss):
        """Power-Link Path-Enforcing Learning (Sec 5.2, Algorithm 1): train
        a Triplet Edge Scorer (TES) per instance by backprop through the
        fully-differentiable graph-powering path loss (Eq 10) plus the
        prediction/fidelity loss (Eq 11) and a ||M||^2 regularizer (Eq 12).
        Entity embeddings come from the frozen encoder and are computed
        once; only the TES's own parameters (MLP + relation embedding
        table) are optimized here."""
        device = ml_ghetero.device
        ghomo_struct, homo_node_emb = self._frozen_homo_node_emb(ml_ghetero, feat_nids)
        homo_src_e, homo_dst_e = ghomo_struct.edges()
        homo_etype_ids = ghomo_struct.edata[dgl.ETYPE]
        homo_eid_within_etype = ghomo_struct.edata[dgl.EID]

        canonical_etypes = ml_ghetero.canonical_etypes
        num_etypes = len(canonical_etypes)
        emb_dim = homo_node_emb.shape[1]
        tes = TripletEdgeScorer(emb_dim, num_etypes, hidden_dim=max(32, emb_dim)).to(device)
        optimizer = torch.optim.Adam(tes.parameters(), lr=self.lr)

        pred_etype_ids = [i for i, et in enumerate(canonical_etypes) if et[1] == self.pred_etype]
        pred_etype_id = torch.tensor(pred_etype_ids[0] if pred_etype_ids else 0, device=device)

        h_i = homo_node_emb[homo_src_e]
        t_k = homo_node_emb[homo_dst_e]
        h_hat = homo_node_emb[homo_src_nid].unsqueeze(0)
        t_hat = homo_node_emb[homo_tgt_nid].unsqueeze(0)

        def scores_to_hetero_eweight_dict(scores):
            eweight_dict = {}
            for i, etype in enumerate(canonical_etypes):
                mask = homo_etype_ids == i
                n = ml_ghetero.num_edges(etype)
                w = torch.zeros(n, device=device, dtype=scores.dtype)
                if mask.any():
                    w = w.scatter(0, homo_eid_within_etype[mask], scores[mask])
                eweight_dict[etype] = w
            return eweight_dict

        if self.log:
            pbar = tqdm(total=self.num_epochs)

        eweight_norm = 0
        EPS = 1e-3
        last_logits = None
        for e in range(self.num_epochs):
            logits = tes(h_i, homo_etype_ids, t_k, h_hat, pred_etype_id, t_hat)
            eweights_homo = logits.sigmoid()
            last_logits = logits

            ml_eweight_dict = scores_to_hetero_eweight_dict(eweights_homo)
            score = self.model(src_nid, tgt_nid, ml_ghetero, feat_nids, ml_eweight_dict)
            pred_loss = (-1) ** pred * score.sigmoid().log()
            self.all_loss['pred_loss'] += [pred_loss.item()]

            curr_eweight_norm = eweights_homo.norm()
            if abs(eweight_norm - curr_eweight_norm) < EPS:
                break
            eweight_norm = curr_eweight_norm

            if with_path_loss:
                p_on = power_link_p_on(ghomo_struct, homo_src_nid, homo_tgt_nid,
                                        eweights_homo, max_length=self._path_loss_max_length)
                path_loss = -torch.log(p_on.clamp(min=1e-12)) if p_on is not None else 0
            else:
                path_loss = 0

            reg_loss = self.gamma * eweights_homo.pow(2).sum()

            loss = pred_loss + path_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        with torch.no_grad():
            final_logits = last_logits if last_logits is not None else \
                tes(h_i, homo_etype_ids, t_k, h_hat, pred_etype_id, t_hat)
            ml_edge_mask_dict = {}
            for i, etype in enumerate(canonical_etypes):
                mask = homo_etype_ids == i
                n = ml_ghetero.num_edges(etype)
                w = torch.full((n,), float('-inf'), device=device)
                if mask.any():
                    w = w.scatter(0, homo_eid_within_etype[mask], final_logits[mask])
                ml_edge_mask_dict[etype] = w

        return ml_edge_mask_dict

    def get_paths(self,
                  src_nid, 
                  tgt_nid, 
                  ghetero,
                  edge_mask_dict,
                  num_paths=1, 
                  max_path_length=3):

        """A postprocessing step that turns the `edge_mask_dict` into actual paths.
        
        Parameters
        ----------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges

        Others: see the `explain` method.
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)
        """
        ntype_pairs_to_cannonical_etypes = get_ntype_pairs_to_cannonical_etypes(ghetero)
        eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
        ghetero.edata['eweight'] = eweight_dict

        # convert ghetero to ghomo and find paths
        ghomo = dgl.to_homogeneous(ghetero, edata=['eweight'])
        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]

        if self.path_method == 'power':
            # Power-Link Path Generation (Eq 13): invert the trained TES
            # score matrix M into a cost matrix and run Dijkstra once.
            inverse_score_func = get_inverse_score_func(ghomo, 'eweight')
            homo_paths = k_shortest_paths_with_max_length(ghomo,
                                                           homo_src_nid,
                                                           homo_tgt_nid,
                                                           weight=inverse_score_func,
                                                           k=num_paths,
                                                           max_length=max_path_length)
        else:
            neg_path_score_func = get_neg_path_score_func(ghomo, 'eweight', [src_nid.item(), tgt_nid.item()])
            homo_paths = k_shortest_paths_with_max_length(ghomo,
                                                           homo_src_nid,
                                                           homo_tgt_nid,
                                                           weight=neg_path_score_func,
                                                           k=num_paths,
                                                           max_length=max_path_length)

        paths = []
        homo_nids_to_ntype_hetero_nids = get_homo_nids_to_ntype_hetero_nids(ghetero)
    
        if len(homo_paths) > 0:
            for homo_path in homo_paths:
                hetero_path = []
                for i in range(1, len(homo_path)):
                    homo_u, homo_v = homo_path[i-1], homo_path[i]
                    hetero_u_ntype, hetero_u_nid = homo_nids_to_ntype_hetero_nids[homo_u] 
                    hetero_v_ntype, hetero_v_nid = homo_nids_to_ntype_hetero_nids[homo_v] 
                    can_etype = ntype_pairs_to_cannonical_etypes[(hetero_u_ntype, hetero_v_ntype)]    
                    hetero_path += [(can_etype, hetero_u_nid, hetero_v_nid)]
                paths += [hetero_path]

        else:
            # A rare case, no paths found, take the top edges
            cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
            M = len(cat_edge_mask)
            k = min(num_paths * max_path_length, M)
            threshold = cat_edge_mask.topk(k)[0][-1].item()
            path = []
            for etype in edge_mask_dict:
                u, v = ghetero.edges(etype=etype)  
                topk_edge_mask = edge_mask_dict[etype] >= threshold
                path += list(zip([etype] * topk_edge_mask.sum().item(), u[topk_edge_mask].tolist(), v[topk_edge_mask].tolist()))                
            paths = [path]
        return paths
    
    def explain(self,  
                src_nid, 
                tgt_nid, 
                ghetero,
                num_hops=2,
                prune_max_degree=-1,
                k_core=2, 
                num_paths=1, 
                max_path_length=3,
                prune_graph=True,
                with_path_loss=True,
                return_mask=False):
        
        """Return a path explanation of a predicted link
        
        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        ghetero : dgl graph

        num_hops : int
            Number of hops to extract the computation graph, i.e. GNN # layers
            
        prune_max_degree : int
            If positive, prune the edges of graph nodes with degree larger than `prune_max_degree`
            If  -1, do nothing
            
        k_core : int 
            k for the the k-core graph extraction
            
        num_paths : int
            Number of paths for the postprocessing path extraction
            
        max_path_length : int
            Maximum length of paths for the postprocessing path extraction
        
        prune_graph : bool
            If true apply the max_degree and/or k-core pruning. For ablation. Default True.
            
        with_path_loss : bool
            If true include the path loss. For ablation. Default True.
            
        return_mask : bool
            If true return the edge mask in addition to the path. For AUC evaluation. Default False
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)

        (optional) edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        # Used by path_loss()'s power-method branch during mask learning below.
        self._path_loss_max_length = max_path_length

        # Extract the computation graph (k-hop subgraph)
        (comp_g_src_nid,
         comp_g_tgt_nid,
         comp_g,
         comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(self.src_ntype,
                                                             src_nid,
                                                             self.tgt_ntype,
                                                             tgt_nid,
                                                             ghetero,
                                                             num_hops)
        # Learn the edge mask on the computation graph
        comp_g_edge_mask_dict = self.get_edge_mask(comp_g_src_nid, 
                                                   comp_g_tgt_nid, 
                                                   comp_g, 
                                                   comp_g_feat_nids,
                                                   prune_max_degree,
                                                   k_core,
                                                   prune_graph,
                                                   with_path_loss)

        # Extract paths 
        comp_g_paths = self.get_paths(comp_g_src_nid,
                                      comp_g_tgt_nid, 
                                      comp_g, 
                                      comp_g_edge_mask_dict, 
                                      num_paths, 
                                      max_path_length)    
        
        
        # Covert the node id in computation graph to original graph
        paths = comp_g_paths_to_paths(comp_g, comp_g_paths)
        
        if return_mask:
            # return masks for easier evaluation
            return paths, comp_g_edge_mask_dict
        else:
            return paths 



