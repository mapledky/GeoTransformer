import torch
import torch.nn as nn
import torch.nn.functional as F


def log_optimal_transport(scores, alpha, iters, src_mask, tgt_mask):
    b, m, n = scores.shape

    if src_mask is None:
        ms = m
        ns = n
    else:
        ms = src_mask.sum(dim=1, keepdim=True)
        ns = tgt_mask.sum(dim=1, keepdim=True)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    Z = torch.cat([torch.cat([scores, bins0], -1),
                   torch.cat([bins1, alpha], -1)], 1)

    # Convert ms and ns to float
    ms = ms.float()
    ns = ns.float()

    norm = -(ms + ns).log()  # [b, 1]

    log_mu = torch.cat([norm.repeat(1, m), ns.log() + norm], dim=1)
    log_nu = torch.cat([norm.repeat(1, n), ms.log() + norm], dim=1)

    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    Z = Z + u.unsqueeze(2) + v.unsqueeze(1)
    Z = Z - norm.view(-1, 1, 1)

    return Z



class Matching(nn.Module):

    def __init__(self, f_dim=256, thr=0.2, bin_score=1., iter=3, ):
        super().__init__()

        self.confidence_threshold = thr

        d_model = f_dim

        self.src_proj = nn.Linear(d_model, d_model, bias=False)
        self.tgt_proj = nn.Linear(d_model, d_model, bias=False)

        # Sinkhorn algorithm
        self.skh_init_bin_score = bin_score
        self.skh_iters = iter
        self.bin_score = nn.Parameter(
            torch.tensor(self.skh_init_bin_score, requires_grad=True))


    def forward(self, src_feats, tgt_feats, src_mask, tgt_mask):
        '''
        @param src_feats: [B, S, C]
        @param tgt_feats: [B, T, C]
        @param src_mask: [B, S]
        @param tgt_mask: [B, T]
        @return:
        '''
        src_feats = self.src_proj(src_feats)
        tgt_feats = self.tgt_proj(tgt_feats)

        src_feats, tgt_feats = map(lambda feat: feat / feat.shape[-1] ** .5, [src_feats, tgt_feats])

        # Optimal transport sinkhorn
        sim_matrix = torch.einsum("bsc,btc->bst", src_feats, tgt_feats)

        if src_mask is not None:
            sim_matrix.masked_fill_(
                ~(src_mask[..., None] * tgt_mask[:, None]).bool(), float('-inf'))

        log_assign_matrix = log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters, src_mask, tgt_mask)

        assign_matrix = log_assign_matrix.exp()
        conf_matrix = assign_matrix[:, :-1, :-1].contiguous()

        return conf_matrix
