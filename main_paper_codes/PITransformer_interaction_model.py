import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhysicsParamEstimator(nn.Module):

    def __init__(self, n_channels=18, hidden=256, use_dc=False):
        super().__init__()
        self.use_dc = use_dc                                      # NEW
        d = n_channels * n_channels + (9 if use_dc else 0)        # NEW
        self.feat_norm = nn.LayerNorm(d)
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.inertia_operator   = nn.Linear(hidden, 3).to(device)
        self.damping_operator   = nn.Linear(hidden, 3).to(device)
        self.stiffness_operator = nn.Linear(hidden, 3).to(device)
        self.random_operator    = nn.Linear(hidden, 3).to(device)
        # self.offset_operator    = nn.Linear(hidden, 3).to(device)            # NEW

    def summary(self, x):
        xc = x - x.mean(dim=1, keepdim=True)
        C = torch.einsum('bti,btj->bij', xc, xc) / xc.size(1)     # (B, 18, 18)
        C = torch.sign(C) * torch.log1p(C.abs())                  # compress the dynamic range
        feats = C.flatten(1)                                      # (B, 324)
        return self.feat_norm(feats)

    def forward(self, x):
        h = self.trunk(self.summary(x))
        J = F.softplus(self.inertia_operator(h))
        b = (self.damping_operator(h))
        k = F.softplus(self.stiffness_operator(h))
        R = (self.random_operator(h))
        # c = self.offset_operator(h) 
        return J, b, k, R


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.15, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        

class PhysicsBasedLoss(nn.Module):
    def __init__(self,lambda_phy, lambda_smooth = 0.2, lambda_traj = 0.1, initial_params=None, lower_bounds=None, device = device):
        super(PhysicsBasedLoss, self).__init__()
        
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss(1.0)
        self.lambda_phy = lambda_phy
        self.lambda_smooth = lambda_smooth
        self.lambda_traj = lambda_traj

        # Getters to share the parameters with other classes
    

    def forward(self,predicted_torque, actual_torque, position, target_positions, velocity, target_velocity, acceleration, J, b, R, k):
        
        J = J.unsqueeze(1)
        b = b.unsqueeze(1)
        R = R.unsqueeze(1)
        k = k.unsqueeze(1)
        
        mse_loss = self.mse_loss(predicted_torque, actual_torque).to(device)

        phsyics_force =    J * (acceleration)  + k * (target_positions - position) + R*(velocity) + b * torch.sign(velocity)

            

        physics_loss = self.mse_loss(predicted_torque, phsyics_force.detach())# + self.huber_loss(predicted_torque,phsyics_force) 
        param_loss = self.mse_loss(actual_torque, phsyics_force)
        

        total_loss = mse_loss + self.lambda_phy*(param_loss)# + (1-self.lambda_phy)*param_loss    #+ hubber_loss + self.lambda_smooth * smoothness_loss +
        #return [mse_loss, physics_loss, penalty]
        return total_loss
    

class PhysicsInformedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, physics_dim=18, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, bias=bias, dropout=dropout, batch_first=True)
        self.physics_projection = nn.Linear(physics_dim, d_model, bias=False)  # Project physics features
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, physics_features):
        """
        x: Tensor of shape (batch_size, seq_length, d_model)
        physics_features: Tensor of shape (batch_size, seq_length, physics_dim)
        """
        seq_len = x.shape[1]

        # Compute physics bias and apply it to the keys
        physics_bias = self.physics_projection(physics_features)  # (batch_size, seq_length, d_model)
        key_with_physics = x + physics_bias
        
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(key_with_physics, key_with_physics, key_with_physics, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(key_with_physics, key_with_physics, key_with_physics, is_causal=False)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y
    
class PhysicsInformedSelfAttentionDecoder(nn.Module):
    def __init__(self, d_model, n_heads, physics_dim=9, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, bias=bias, dropout=dropout, batch_first=True)
        self.physics_projection = nn.Linear(physics_dim, d_model, bias=False)  # Project physics features
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, physics_features):
        """
        x: Tensor of shape (batch_size, seq_length, d_model)
        physics_features: Tensor of shape (batch_size, seq_length, physics_dim)
        """
        seq_len = x.shape[1]

        # Compute physics bias and apply it to the keys
        physics_bias = self.physics_projection(physics_features)  # (batch_size, seq_length, d_model)
        key_with_physics = x + physics_bias
        
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(key_with_physics, key_with_physics, key_with_physics, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(key_with_physics, key_with_physics, key_with_physics, is_causal=False)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class PhysicsInformedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, physics_dim=18, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        # Physics-informed projections
        self.physics_projection_encoder = nn.Linear(physics_dim, d_model, bias=False)  # For encoder physics bias
        self.physics_projection_decoder = nn.Linear(physics_dim - 9, d_model, bias=False)  # For decoder physics bias

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, x, physics_features, physics_features_decoder):
        """
        x: Decoder input (batch_size, seq_length, d_model)
        encoder_output: Encoder output (batch_size, encoder_seq_length, d_model)
        physics_features: Physics features for encoder (batch_size, encoder_seq_length, physics_dim)
        physics_features_decoder: Physics features for decoder (batch_size, seq_length, physics_dim - 9)
        """

        # Compute physics-based biases
        physics_bias_encoder = self.physics_projection_encoder(physics_features)  # (batch_size, encoder_seq_length, d_model)
        physics_bias_decoder = self.physics_projection_decoder(physics_features_decoder)  # (batch_size, seq_length, d_model)

        # Add physics biases to encoder keys and values
        key_with_physics = x + physics_bias_decoder
        # value_with_physics = encoder_output + physics_bias_encoder  # Values are also affected
        key_with_physics_encoder = encoder_output + physics_bias_encoder
        # Add physics bias to the decoder queries
        # query_with_physics = x + physics_bias_decoder
        

        # Compute cross-attention
        # attn_output, _ = self.mha(query_with_physics, key_with_physics, value_with_physics)
        x = self.mha(key_with_physics, key_with_physics_encoder, key_with_physics_encoder, is_causal=self.causal)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout =0.0, bias = False):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = PhysicsInformedSelfAttention(embed_dim, num_heads, dropout= dropout, causal=False, bias= bias )
        # self.attention = SelfAttention(embed_dim, num_heads, dropout= dropout,causal= False, bias= bias)
        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias= bias)
        self.mlp = MLP(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )


    def forward(self, x, physics_features):
        norm1 = self.norm1(x)
        attention = self.attention(norm1,physics_features)
        # attention = self.attention(norm1)
        x = attention+x
        norm2 = self.norm2(x)
        mlp1 = self.mlp(norm2)
        x = x + mlp1

        return x
    
def generate_causal_mask(seq_len):
    
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # Upper triangular matrix
    return mask.to(device)  # Ensure the mask is on the same device as the model


# Transformer Decoder Layer with Cross-Attention
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout = 0.0, bias = False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = PhysicsInformedSelfAttentionDecoder(embed_dim, num_heads, dropout= dropout, causal= True)  # Self-attention in decoder
        self.cross_attention = PhysicsInformedCrossAttention(embed_dim, num_heads, dropout= dropout, causal= False)  # Cross-attention with encoder output

        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias=bias)
        self.norm3 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

    def forward(self, x, decoder_input, physics_features, physics_features_decoder):
        # Self-attention within the decoder layer
        norm1= self.norm1(decoder_input)  # Apply normalization before self-attention
        self_attention = self.self_attention(norm1, physics_features_decoder)
        decoder_input= self_attention + decoder_input
        norm2 = self.norm2(decoder_input)

        cross_attention = self.cross_attention(x, norm2, physics_features, physics_features_decoder)
        decoder_output = cross_attention  + decoder_input
        
        # MLP
        norm3 = self.norm3(decoder_output)  # Apply normalization before MLP
        mlp1 = self.mlp(norm3)
        
        decoder_output = mlp1 + decoder_output

        
        return decoder_output


        
class PhysicsAwareEmbedding(nn.Module):
    def __init__(self, input_dim, n_embd, device=device):
        """
        Physics-aware embedding layer with explicit physics-based features.
        
        Parameters:
        - input_dim: Number of raw input features (positions, velocities, accelerations, forces).
        - n_embd: Output embedding dimension.
        - device: Computation device.
        """
        super(PhysicsAwareEmbedding, self).__init__()
        self.device = device
        self.n_embd = n_embd

        # Learnable weights for each input feature (dimension-specific scaling)
        self.weights = nn.Parameter(torch.ones(input_dim), requires_grad=True)

        # Linear transformation to project physics-enhanced features to embedding space
        self.embedding_layer = nn.Linear(input_dim + 12, n_embd)  # Extra 4 for physics features

        # Physics-informed activation
        self.activation = nn.Tanh()

    def forward(self, x, positions, target_positions, velocities, target_velocities, accelerations, interaction_forces, J, b, R, k):

        # Apply learnable scaling to raw input features
        x_weighted = x * self.weights  # Element-wise scaling
        
        J = J.unsqueeze(1)
        b = b.unsqueeze(1)
        R = R.unsqueeze(1)
        k = k.unsqueeze(1)

        # Compute explicit physics-informed features
        kinetic_energy =   J * (accelerations)  # (B, T, D)
        damping_force = b * torch.sign(velocities) + R*(velocities)# (B, T, D)
        elastic_force = k * (target_positions - positions)  # (B, T, D)
        residual_force = interaction_forces - ( elastic_force + kinetic_energy + damping_force)  # (B, T, D)

        # Concatenate raw inputs with physics-based features
        physics_features = torch.cat([
            residual_force,
            kinetic_energy,  # Convert (B, T, 3) → (B, T, 1)
            damping_force ,  
            elastic_force ,
            x_weighted  # Full 15 features
        ], dim=-1).to(device=device)  # (B, T, 15 + 4 = 19)
    

        # Apply linear transformation
        embeddings = self.embedding_layer(physics_features)

        # Apply physics-informed activation
        # embeddings = self.activation(embeddings)

        return embeddings

        

# Updated EnhancedTransformer class
class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, n_embd, forward_expansion,seq_len,
                 dropout = 0.0, bias = False, device = device):
        super(EnhancedTransformer, self).__init__()
        # self.embedding = nn.Linear(input_dim, n_embd)
        
        self.embedding = PhysicsAwareEmbedding(input_dim, n_embd, device)
        self.embedding_output = PhysicsAwareEmbedding(input_dim-9, n_embd,device)

        self.encoder_wte = nn.Linear(input_dim, n_embd).to(device)
        self.encoder_wpe = nn.Embedding(seq_len, n_embd).to(device)
        
        self.decoder_wte = nn.Linear(input_dim -9, n_embd).to(device)
        self.decoder_wpe = nn.Embedding(seq_len, n_embd).to(device)
                
        self.norm1 = LayerNorm(n_embd, bias=bias).to(device)
        self.norm2 = LayerNorm(n_embd, bias=bias).to(device)
        
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        
        self.paremeter_estimator = PhysicsParamEstimator(input_dim).to(device=device)
        
        self.decoder_output = nn.Linear(n_embd, 3, bias= True).to(device)  # Output layer for torque prediction
        
    def DecoderEmbedding(self, decoder_input):
        
        device = decoder_input.device
        seq_len_decoder = decoder_input.size(1)
        
        # Positional embedding
        pos_decoder = torch.arange(0, seq_len_decoder, dtype=torch.long, device=device).unsqueeze(0)
        
        pos_emb_decoder = self.decoder_wpe(pos_decoder)
        tok_emb_decoder = self.decoder_wte(decoder_input)
        

        # Combine physics-aware embedding and positional embedding
        return  pos_emb_decoder 

    def EncoderEmbeding(self, x):
        
        device = x.device
        seq_len_encoder = x.size(1)
        
        # Positional embedding
        pos_encoder= torch.arange(0, seq_len_encoder, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb_encoder = self.encoder_wpe(pos_encoder)
        tok_emb_encoder = self.encoder_wte(x)


        # Combine physics-aware embedding and positional embedding
        return   pos_emb_encoder #+ tok_emb_encoder

    def forward(self, x, decoder_input, positions, target_positions, velocities, target_velocities, accelerations, torques,
                positions_next, velocities_next, accelerations_next):
        
        physics_features = torch.cat(
            [positions, target_positions, velocities, target_velocities, accelerations, torques], dim=-1
        ) 
        
        physics_features_decoder = torch.cat(
            [ positions_next, velocities_next, accelerations_next], dim=-1
        )
        

        J, b, k, R = self.paremeter_estimator(x)
        
        # physics_emb_encoder = self.embedding(x,torques, accelerations, target_velocities, velocities, target_positions, positions).to(device)
        physics_emb_encoder = self.embedding(x,positions, target_positions, velocities, target_velocities, accelerations, torques,J, b, R, k).to(device)
        # physics_emb_decoder = self.embedding_output(decoder_input, torques, accelerations_next, target_velocities, velocities, target_positions, positions).to(device)
        physics_emb_decoder = self.embedding_output(decoder_input, positions, target_positions, velocities, target_velocities, accelerations, torques,J, b, R, k).to(device)
        
        x = self.EncoderEmbeding(x) + physics_emb_encoder 
        decoder_input = self.DecoderEmbedding(decoder_input) + physics_emb_decoder

        for layer in self.encoder_layers:
            x = layer(x, physics_features)
      
        x = self.norm1(x)

        

        
        # decoder_input = self.norm1(decoder_input)
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(x, decoder_output, physics_features, physics_features_decoder)

        decoder_output = self.norm2(decoder_output)
                
        return self.decoder_output(decoder_output),J, b, R, k  # Predict based on the last time step
