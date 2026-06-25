import torch
import torch.nn as nn
import torch.nn.functional as F

class FKANLayer(nn.Module):
    """
    Implements harmonic functional decomposition as a temporal deviation operator.
    Leverages sinusoidal basis functions to learn frequency-aware transformations.
    """
    def __init__(self, in_features, out_features, num_frequencies=8):
        super(FKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # Trainable coefficients for the harmonic components
        self.fourier_coeffs = nn.Parameter(
            torch.randn(num_frequencies, in_features, out_features) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            batch_size, seq_len, in_feats = orig_shape
            x = x.reshape(-1, in_feats)
        
        out = torch.zeros(x.size(0), self.out_features, device=x.device)
        
        # Apply harmonic functional terms: sin(k * x)
        for k in range(1, self.num_frequencies + 1):
            harmonic = torch.sin(k * x)
            out += torch.matmul(harmonic, self.fourier_coeffs[k-1])
            
        out += self.bias
        
        if len(orig_shape) == 3:
            return out.reshape(batch_size, orig_shape[1], self.out_features)
        return out


class TransformerFKANBranch(nn.Module):
    """
    Processes a single view via a joint Transformer and FKAN pipeline.
    Isolates deviations using residual modeling between embeddings.
    """
    def __init__(self, input_dim, embed_dim=64, nhead=4, num_layers=2):
        super(TransformerFKANBranch, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fkan_operator = FKANLayer(embed_dim, embed_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )

    def forward(self, x):
        proj = self.input_projection(x)
        transformer_out = self.transformer_encoder(proj)
        
        # Generate harmonic baseline baseline
        harmonic_baseline = self.fkan_operator(transformer_out)
        
        # Extract deviation component via structural residual
        deviation_residual = transformer_out - harmonic_baseline
        
        # Gated combination (Section V)
        g = self.gate(torch.cat([transformer_out, deviation_residual], dim=-1))
        z = g * transformer_out + (1 - g) * deviation_residual
        
        z_fused = z.mean(dim=1)
        reconstruction = self.decoder(z_fused)
        
        return z_fused, reconstruction


class MultiViewFKANIDS(nn.Module):
    """
    Unified Multi-View Cyber-Physical Intrusion Detection System.
    """
    def __init__(self, inst_dim=12, struct_dim=120, deriv_dim=12, embed_dim=64):
        super(MultiViewFKANIDS, self).__init__()
        self.inst_branch = TransformerFKANBranch(inst_dim, embed_dim)
        self.struct_branch = TransformerFKANBranch(struct_dim, embed_dim)
        self.deriv_branch = TransformerFKANBranch(deriv_dim, embed_dim)
        
        self.proj_inst = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.proj_struct = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.proj_deriv = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, x_inst, x_struct, x_deriv):
        z_i, recon_i = self.inst_branch(x_inst)
        z_s, recon_s = self.struct_branch(x_struct)
        z_d, recon_d = self.deriv_branch(x_deriv)
        
        p_i = F.normalize(self.proj_inst(z_i), p=2, dim=-1)
        p_s = F.normalize(self.proj_struct(z_s), p=2, dim=-1)
        p_d = F.normalize(self.proj_deriv(z_d), p=2, dim=-1)
        
        # Dynamic Attention Fusion (Section VII)
        stacked_z = torch.stack([z_i, z_s, z_d], dim=1)
        attn_logits = self.attention_weights(stacked_z).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        z_global = torch.sum(attn_weights * stacked_z, dim=1)
        
        return {
            "embeddings": (z_i, z_s, z_d),
            "projections": (p_i, p_s, p_d),
            "reconstructions": (recon_i, recon_s, recon_d),
            "global_embedding": z_global
        }


class MultiViewContrastiveLoss(nn.Module):
    """
    InfoNCE Multi-View Alignment Loss combined with reconstruction regularizers.
    """
    def __init__(self, temperature=0.07):
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature

    def pairwise_contrastive(self, p1, p2):
        batch_size = p1.size(0)
        representations = torch.cat([p1, p2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        mask = (~torch.eye(batch_size * 2, dtype=torch.bool, device=p1.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity_matrix / self.temperature)
        
        return -torch.log(numerator / torch.sum(denominator, dim=1)).mean()

    def forward(self, outputs, inputs, alpha=1.0, beta=0.5):
        p_i, p_s, p_d = outputs["projections"]
        recon_i, recon_s, recon_d = outputs["reconstructions"]
        x_inst, x_struct, x_deriv = inputs
        
        loss_contrastive = (
            self.pairwise_contrastive(p_i, p_s) +
            self.pairwise_contrastive(p_s, p_d) +
            self.pairwise_contrastive(p_i, p_d)
        ) / 3.0
        
        loss_recon = (
            F.mse_loss(recon_i, x_inst.mean(dim=1)) +
            F.mse_loss(recon_s, x_struct.mean(dim=1)) +
            F.mse_loss(recon_d, x_deriv.mean(dim=1))
        ) / 3.0
        
        return alpha * loss_contrastive + beta * loss_recon