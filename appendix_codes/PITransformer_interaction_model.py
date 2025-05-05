import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x), a smooth non-linearity often used in place of ReLU.
    """
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class LayerNorm(nn.Module):
    """ 
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False 
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MLP(nn.Module):
    """
    Feed-forward sub-layer in Transformer blocks: expands, applies GELU, projects back, and adds dropout.
    """

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
    """
    Custom loss integrating physics-informed constraints into training.
    Includes:
    - MSE between predicted and actual forces
    - Physics-consistent force computation using estimated J, b, k, R
    - A penalty on physical parameters if they fall below defined bounds
    """


    def __init__(self,lambda_phy, lambda_smooth = 0.2, lambda_traj = 0.1, initial_params=None, lower_bounds=None, device = device):
        super(PhysicsBasedLoss, self).__init__()
        
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss(1.0)
        self.lambda_phy = lambda_phy
        self.lambda_smooth = lambda_smooth
        self.lambda_traj = lambda_traj

        self.J = nn.Parameter(torch.tensor(initial_params['inertia'], dtype=torch.float32, device= self.device), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(initial_params['damping'], dtype=torch.float32, device= self.device), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(initial_params['stiffness'], dtype=torch.float32, device= self.device), requires_grad=False)
        self.R = nn.Parameter(torch.tensor(initial_params['random'], dtype=torch.float32, device= self.device), requires_grad = False)
        self.lower_bounds = {k: torch.tensor(v, dtype=torch.float32, device= self.device) for k, v in lower_bounds.items()}

        # Getters to share the parameters with other classes
    

    def forward(self,predicted_force, actual_force, position, target_positions, velocity, target_velocity, acceleration, J, b, k, R):
        mse_loss = self.mse_loss(predicted_force, actual_force).to(device)
        phsyics_force =    J * (acceleration)  + k * (position - target_positions) + R*target_velocity + b * torch.sign(target_velocity)
        with torch.no_grad():
            self.J.copy_(J)
            self.b.copy_(b)
            self.k.copy_(k)
            self.R.copy_(R)
        
        physics_loss = self.mse_loss(predicted_force, phsyics_force)

        # Penalty: enforce lower bounds on the physical parameters to avoid non-physical values
        penalty = sum(torch.sum(torch.relu(self.lower_bounds[param] - getattr(self, param)))
                      for param in ['J', 'b', 'k', 'R'])
        
        total_loss = mse_loss + self.lambda_phy*(physics_loss + penalty )
        return total_loss
    
    @property
    def J_val(self):
        return self.J

    @property
    def b_val(self):
        return self.b

    @property
    def k_val(self):
        return self.k

    @property
    def R_val(self):
        return self.R
    

class PhysicsInformedSelfAttention(nn.Module):
    """Self-attention layer enhanced with physics context:
    - Adds a learned projection of physics features to modify keys/values.
    - Standard MHA used, but with physics-informed biases.
    """

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
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class PhysicsInformedCrossAttention(nn.Module):
    """
    Cross-attention layer integrating separate physics context from encoder and decoder:
    - Uses different projections for encoder and decoder physics inputs
    - Adds biases before attention computation
    """
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
        key_with_physics_encoder = encoder_output + physics_bias_encoder
 
        # Compute cross-attention
        x = self.mha(key_with_physics, key_with_physics_encoder, key_with_physics_encoder, is_causal=self.causal)[0]
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class TransformerEncoderLayer(nn.Module):
    """
    A full encoder block:
    - Physics-informed self-attention
    - LayerNorm and MLP block
    - Skip connections
    """
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout =0.0, bias = False):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = PhysicsInformedSelfAttention(embed_dim, num_heads, dropout= dropout, causal=False, bias= bias )
        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias= bias)
        self.mlp = MLP(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )


    def forward(self, x, physics_features):
        # Self-attention within the encoder layer
        norm1 = self.norm1(x)  # Apply normalization before self-attention
        attention = self.attention(norm1,physics_features)
        x = attention+x
        # MLP
        norm2 = self.norm2(x)  # Apply normalization before MLP
        mlp1 = self.mlp(norm2)
        x = x + mlp1
        return x
    
def generate_causal_mask(seq_len):
    
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # Upper triangular matrix
    return mask.to(device)  # Ensure the mask is on the same device as the model


# Transformer Decoder Layer with Cross-Attention
class TransformerDecoderLayer(nn.Module):
    """
    Decoder block with:
    - Physics-aware self-attention and cross-attention
    - LayerNorms and MLP for final processing
    """
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout = 0.0, bias = False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = PhysicsInformedSelfAttention(embed_dim, num_heads, dropout= dropout, causal= True)  # Self-attention in decoder
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
        self_attention = self.self_attention(norm1, physics_features)
        decoder_input= self_attention + decoder_input
        # Cross-attention
        norm2 = self.norm2(decoder_input)  # Apply normalization before cross-attention
        cross_attention = self.cross_attention(x, norm2, physics_features, physics_features_decoder)
        decoder_output = cross_attention  + decoder_input
        # MLP
        norm3 = self.norm3(decoder_output)  # Apply normalization before MLP
        mlp1 = self.mlp(norm3)
        decoder_output = mlp1 + decoder_output
        
        return decoder_output


class DataAwareLearnablePositionalEncoding(nn.Module):
    """
    Positional encoding initialized and normalized based on data statistics (mean and std).
    Helps bridge input data scale with positional representations for more stable learning.
    """

    def __init__(self, embed_dim, seq_length, mean, std, max_len=500, init_std=1e-6):
        super(DataAwareLearnablePositionalEncoding, self).__init__()
        
        # Ensure mean and std are tensors and match embed_dim
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32).to(device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32).to(device)
        
        # Expand mean and std to match embed_dim
        mean = mean.mean().expand(embed_dim)
        std = std.mean().expand(embed_dim)
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.mean = mean
        self.std = std

        # Learnable positional encodings
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_len, embed_dim).to(device)
        )

        # Data-aware initialization
        nn.init.normal_(self.positional_encoding, mean=mean.mean().item(), std=init_std)

    def forward(self, x):
        # Slice positional encodings to match input sequence length
        pos_enc = self.positional_encoding[:, :x.size(1)]
        
        # Normalize positional encodings to match training data scale
        pos_enc = (pos_enc - self.mean) / self.std
        
        return x + pos_enc


# Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    """
    Standard learnable positional encoding layer.
    Adds trainable position-dependent vectors to token embeddings.
    """

    def __init__(self, d_model, max_len=200):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_embeddings[:, :seq_len]

class PhysicsPositionalEncoding(nn.Module):
    """
    Composite positional encoding:
    - Adds learnable encodings
    - Adds data-aware encodings (normalized)
    - Pads physics features to match dimension and includes them too
    """

    def __init__(self, d_model,mean , std, max_len=352):
        super(PhysicsPositionalEncoding, self).__init__()
        self.learnable_positional_encoding = LearnablePositionalEncoding(d_model, max_len)
        self.Dataawar_encoding = DataAwareLearnablePositionalEncoding(d_model, seq_length= max_len, mean= mean, std= std, max_len= max_len)
        self.d_model = d_model

    def forward(self, x, physics_features):
        # Standard positional encoding
        learnable_encoding = self.learnable_positional_encoding(x)
        data_encoding = self.Dataawar_encoding(x)
       
        physics_encoding = torch.nn.functional.pad(
            physics_features, (0, self.d_model - physics_features.size(-1)), mode='constant', value=0
        )   # Pad to match d_model
        
        # Combine learnable and physics-based encodings
        return   physics_encoding + data_encoding + learnable_encoding # maybe - 2*x #+ physics_encoding + 
        
class PhysicsAwareEmbedding(nn.Module):
    """
    Physics-aware embedding:
    - Combines raw features with engineered physics features:
    * Kinetic energy
    * Damping force
    * Elastic force
    * Residuals (force - modeled force)
    - Applies learnable scaling, projection, and non-linearity
    """

    def __init__(self, input_dim, n_embd, physics_params, device=device):
        """
        Physics-aware embedding layer with explicit physics-based features.
        
        Parameters:
        - input_dim: Number of raw input features (positions, velocities, accelerations, forces).
        - n_embd: Output embedding dimension.
        - physics_params: Dictionary of learnable physics parameters (J, b, k, R).
        - device: Computation device.
        """
        super(PhysicsAwareEmbedding, self).__init__()
        self.device = device
        self.n_embd = n_embd

        # Learnable weights for each input feature (dimension-specific scaling)
        self.weights = nn.Parameter(torch.ones(input_dim), requires_grad=True)

        # Physics parameters (shared with the loss function)
        self.J = physics_params["J"]
        self.b = physics_params["b"]
        self.k = physics_params["k"]
        self.R = physics_params["R"]

        # Linear transformation to project physics-enhanced features to embedding space
        self.embedding_layer = nn.Linear(input_dim + 4, n_embd)  # Extra 4 for physics features

        # Physics-informed activation
        self.activation = nn.Tanh()

    def forward(self, x, positions, target_positions, velocities, target_velocities, accelerations, interaction_forces):

        # Apply learnable scaling to raw input features
        x_weighted = x * self.weights  # Element-wise scaling

        # Compute explicit physics-informed features
        kinetic_energy =   self.J * (accelerations)  # (B, T, D)
        damping_force = self.b * torch.sign(target_velocities) + self.R*target_velocities# (B, T, D)
        elastic_force = self.k * (positions - target_positions)  # (B, T, D)
        residual_force = interaction_forces - ( elastic_force + kinetic_energy + damping_force)  # (B, T, D)

        # Concatenate raw inputs with physics-based features
        physics_features = torch.cat([
            x_weighted,  # Full 15 features
            kinetic_energy.mean(dim=-1, keepdim=True),  # Convert (B, T, 3) → (B, T, 1)
            damping_force.mean(dim=-1, keepdim=True),  
            elastic_force.mean(dim=-1, keepdim=True),
            residual_force.mean(dim=-1, keepdim=True)  
        ], dim=-1)  # (B, T, 15 + 4 = 19)

        # Apply linear transformation
        embeddings = self.embedding_layer(physics_features)

        # Apply physics-informed activation
        embeddings = self.activation(embeddings)

        return embeddings
    

        

# Updated EnhancedTransformer class
class EnhancedTransformer(nn.Module):    
    """
    Main physics-informed Transformer model.
    Architecture:
    - Encoder: Encodes input trajectory and physical info
    - Decoder: Predicts future interaction forces
    - Positional encodings are physics-informed
    - Final layers estimate physical parameters (J, b, k, R)
    """

    def __init__(self, input_dim, n_heads, n_layers, n_embd, forward_expansion,seq_len,seq_len_dec, mean, std, physics_params,
                 dropout = 0.0, bias = False, device = device):
        super(EnhancedTransformer, self).__init__()
        
        self.embedding = PhysicsAwareEmbedding(input_dim, n_embd, physics_params, device)
        self.embedding_output = PhysicsAwareEmbedding(input_dim-9, n_embd, physics_params,device)
        
        self.encoder_wte = nn.Linear(input_dim, n_embd).to(device)
        self.encoder_wpe = nn.Embedding(seq_len, n_embd).to(device)
        
        self.decoder_wte = nn.Linear(input_dim -9, n_embd).to(device)
        self.decoder_wpe = nn.Embedding(seq_len, n_embd).to(device)
        
        self.positional_encoding_enc = PhysicsPositionalEncoding(n_embd, mean, std).to(device)
        self.positional_encoding_dec = PhysicsPositionalEncoding(n_embd, mean, std).to(device)
        
        self.norm1 = LayerNorm(n_embd, bias=bias).to(device)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)

        self.stiffness_operator = nn.Linear(n_embd, 3, bias=True).to(device) 
        self.inertia_operator = nn.Linear(n_embd, 3, bias=True).to(device)  
        self.damping_operator = nn.Linear(n_embd, 3, bias=True).to(device) 
        self.random_operator = nn.Linear(n_embd, 3, bias=True).to(device)  
        
        self.decoder_output = nn.Linear(n_embd, 3, bias= True).to(device)  # Output layer for force prediction
        
    def DecoderEmbedding(self, decoder_input):
        
        device = decoder_input.device
        seq_len_decoder = decoder_input.size(1)
        
        # Positional embedding
        pos_decoder = torch.arange(0, seq_len_decoder, dtype=torch.long, device=device).unsqueeze(0)
        
        pos_emb_decoder = self.decoder_wpe(pos_decoder)
        tok_emb_decoder = self.decoder_wte(decoder_input)

        # Combine physics-aware embedding and positional embedding
        return   tok_emb_decoder + pos_emb_decoder

    def EncoderEmbeding(self, x):
        
        device = x.device
        seq_len_encoder = x.size(1)
        
        # Positional embedding
        pos_encoder= torch.arange(0, seq_len_encoder, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb_encoder = self.encoder_wpe(pos_encoder)
        tok_emb_encoder = self.encoder_wte(x)

        # Combine physics-aware embedding and positional embedding
        return  tok_emb_encoder + pos_emb_encoder

    def forward(self, x, decoder_input, positions, target_positions, velocities, target_velocities, accelerations, forces,
                positions_next, velocities_next, accelerations_next):

        physics_features = torch.cat(
            [forces, accelerations, target_velocities, velocities, target_positions, positions], dim=-1
        ) 
        
        physics_features_decoder = torch.cat(
            [accelerations_next, velocities_next, positions_next], dim=-1
        )
        
        physics_emb_encoder = self.embedding(x, positions, target_positions, velocities, target_velocities, accelerations, forces).to(device)
        physics_emb_decoder = self.embedding_output(decoder_input, positions, target_positions, velocities, target_velocities, accelerations_next, forces).to(device)
        
        x = self.EncoderEmbeding(x) + physics_emb_encoder
        decoder_input = self.DecoderEmbedding(decoder_input) + physics_emb_decoder
        
        x = self.positional_encoding_enc(x, physics_features)
        decoder_input = self.positional_encoding_dec(decoder_input, physics_features)
        
        for layer in self.encoder_layers:
            x = layer(x, physics_features)
        x_params = x
        x = self.norm1(x)

        for layer in self.decoder_layers:
            decoder_output = layer(x, decoder_input, physics_features, physics_features_decoder)
            decoder_output_params = layer(x_params, decoder_input, physics_features, physics_features_decoder)
        
        estimated_stiffness = F.softplus(torch.mean(self.stiffness_operator(torch.mean(decoder_output_params, dim=1)), dim=0))
        estimated_inertia = F.softplus(torch.mean(self.inertia_operator(torch.mean(decoder_output_params, dim=1)), dim=0))
        estimated_damping = F.softplus(torch.mean(self.damping_operator(torch.mean(decoder_output_params, dim=1)), dim=0))
        estimated_random = F.softplus(torch.mean(self.random_operator(torch.mean(decoder_output_params, dim=1)), dim=0))
        
        decoder_output = self.norm1(decoder_output)
        
        
        
        
        return self.decoder_output(decoder_output),estimated_inertia, estimated_damping,estimated_stiffness, estimated_random  # Predict based on the last time step
