import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
    
class SelfAttention(nn.Module):
    """
    Self-attention mechanism using PyTorch's MultiheadAttention with optional causal masking.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(x, x, x, is_causal=False)[0]
        y = self.resid_dropout(x)  
        return y
    
class CrossAttention(nn.Module):
    """
    Cross-attention module for Transformer decoders.
    Allows decoder to attend over encoder outputs (memory) to integrate contextual information.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, mem, x):
        x = self.mha(x, mem, mem, is_causal=self.causal)[0]
        y = self.resid_dropout(x)
        return y
    
# Shared base module for meta-learning style models
class BaseInteractionModel(nn.Module):
    def __init__(self, input_dim, emb_dim=64, max_len=704, mean=0.0, std=1.0):
        super(BaseInteractionModel, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # Shared embedding projection
        self.input_projection = nn.Linear(input_dim + 3, emb_dim).to(device)

        # Use fair positional encoding
        self.pos_encoding = StdPositionalEncoding(emb_dim, mean, std,max_len = max_len).to(device)

    def combine_context_and_query(self, context_x, context_y, query_x):
        B, T_q, D_q = query_x.shape
        pad_size = self.input_dim - D_q
        padded_query_x = nn.functional.pad(query_x, pad=(0, pad_size), mode='constant', value=0)

        context = torch.cat([context_x, context_y], dim=-1)
        query_y_dummy = torch.zeros(B, T_q, 3, device=query_x.device)
        query = torch.cat([padded_query_x, query_y_dummy], dim=-1)
        return torch.cat([context, query], dim=1)#[context,query]

    def embed(self, combined_sequence):
        x_proj = self.input_projection(combined_sequence)
        return self.pos_encoding(x_proj)

# LSTM-based model
class LSTMInteractionModel(BaseInteractionModel):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMInteractionModel, self).__init__(input_dim)
        self.lstm = nn.LSTM(self.emb_dim, hidden_dim, num_layers=num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim, 3).to(device)

    def forward(self, context_x, context_y, query_x):
        combined = self.combine_context_and_query(context_x, context_y, query_x)
        embedded = self.embed(combined)
        output, _ = self.lstm(embedded)
        return self.fc(output[:, context_x.size(1):])
    
   
# Decoder-only 
class DecoderonlyDecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_heads,
                                       dropout=dropout, causal=True, bias=bias)
        
        self.ln_3 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)


    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))

        x = x + self.mlp(self.ln_3(x))
        return x


class DecoderOnlyInteractionModel(BaseInteractionModel):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, mean, std, dropout=0.0, bias=False):
        super(DecoderOnlyInteractionModel, self).__init__(input_dim, emb_dim=embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [DecoderonlyDecoderLayer(embed_dim, num_heads, dropout, bias) for _ in range(num_layers)]
        )
        self.norm_output = LayerNorm(embed_dim, bias=bias).to(device)
        self.decoder_output = nn.Linear(embed_dim, 3).to(device)


    def forward(self,context_y, context_x,  query_x):
        combined = self.combine_context_and_query(context_x, context_y, query_x)  # (B, T_total, D)
        x = self.embed(combined)

        for block in self.decoder_blocks:
            x = block(x)  

        x = self.norm_output(x)
        return self.decoder_output(x[:, context_x.size(1):])

# TCN
class TCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation, padding='same').to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.relu(self.conv(x)))
    
class TCNInteractionModel(BaseInteractionModel):
    def __init__(self, input_dim, hidden_dim=128, num_layers=6):  # NEW: more layers
        super(TCNInteractionModel, self).__init__(input_dim)
        self.tcn_layers = nn.ModuleList([
            TCNBlock(hidden_dim if i > 0 else self.emb_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_layers)
        ]).to(device)
        self.fc = nn.Linear(hidden_dim, 3).to(device)

    def forward(self, context_x, context_y, query_x):
        combined = self.combine_context_and_query(context_x, context_y, query_x)
        x = self.embed(combined).permute(0, 2, 1)
        for tcn in self.tcn_layers:
            x = tcn(x)
        x = x.permute(0, 2, 1)
        return self.fc(x[:, context_x.size(1):])

# DeepSets
class DeepSetInteractionModel(BaseInteractionModel):
    def __init__(self, input_dim, emb_dim=128, hidden_dim=128, output_dim=3):
        super(DeepSetInteractionModel, self).__init__(input_dim, emb_dim)

        self.phi = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(device)

    def forward(self, context_x, context_y, query_x):
        combined = self.combine_context_and_query(context_x, context_y, query_x)
        embedded = self.embed(combined)  # (B, T, D)

        phi_out = self.phi(embedded)  # (B, T, H)
        pooled = phi_out.mean(dim=1)  # (B, H)

        output = self.rho(pooled)     # (B, 3)
        query_len = query_x.size(1)
        return output.unsqueeze(1).expand(-1, query_len, -1)

    
class TransformerEncoderLayer(nn.Module):
    """
    A standard Transformer encoder block.
    Includes:
    - Pre-normalized self-attention (non-causal)
    - Feed-forward MLP
    - Residual connections
    """

    def __init__(self, embed_dim, num_heads, forward_expansion, dropout =0.0, bias = False):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout= dropout,causal= False, bias= bias) # Self-attention in encoder
        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias= bias)
        self.mlp = MLP(embed_dim)


    def forward(self, x):
        # Self-attention within the encoder layer
        norm1 = self.norm1(x)  # Apply normalization before self-attention
        attention = self.attention(norm1)
        x = attention+x
        # MLP
        norm2 = self.norm2(x)  # Apply normalization before MLP
        mlp1 = self.mlp(norm2)
        x = x + mlp1
        return x
    
def generate_causal_mask(seq_len):
    
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  
    return mask.to(device) 

# Transformer Decoder Layer with Cross-Attention
class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer combining:
    - Causal self-attention on decoder inputs
    - Cross-attention with encoder outputs
    - Feed-forward network
    Includes normalization and residual connections.
    """

    def __init__(self, embed_dim, num_heads, forward_expansion, dropout = 0.0, bias = False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads, dropout= dropout, causal = True, bias = bias)  # Self-attention in decoder
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout= dropout, causal = False, bias =bias)  # Cross-attention with encoder output
        self.norm1 = LayerNorm(embed_dim, bias=bias)
        self.norm2 = LayerNorm(embed_dim, bias=bias)
        self.norm3 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim)


    def forward(self, x, decoder_input): 
        # Self-attention within the decoder layer
        norm1= self.norm1(decoder_input)  # Apply normalization before self-attention
        self_attention = self.self_attention(norm1)
        decoder_input= self_attention + decoder_input
        # Cross-attention
        norm2 = self.norm2(decoder_input)  # Apply normalization before cross-attention
        cross_attention = self.cross_attention(x, norm2)  
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

# Standard positional Encoding
class StdPositionalEncoding(nn.Module):
    """
    Combines standard learnable positional encoding with a data-aware variant.
    Helps align positional embeddings with data statistics (mean and std) to improve performance and stability.
    """

    def __init__(self, d_model, mean, std,max_len=352):
        super(StdPositionalEncoding, self).__init__()
        self.learnable_positional_encoding = LearnablePositionalEncoding(d_model, max_len)
        self.Dataawar_encoding = DataAwareLearnablePositionalEncoding(d_model, seq_length= max_len, mean= mean, std= std, max_len= max_len)
        self.d_model = d_model

    def forward(self, x):
        # Standard positional encoding
        learnable_encoding = self.learnable_positional_encoding(x)  ## added only for performance point of view
        data_encoding = self.Dataawar_encoding(x)
        return learnable_encoding  + data_encoding 


# Updated EnhancedTransformer class
class EnhancedTransformerData(nn.Module):
    """
    Main transformer model for data-driven interaction force prediction.
    Combines encoder/decoder embeddings, data-aware positional encoding,
    and multi-layer Transformer blocks.
    """

    def __init__(self, input_dim, n_heads, n_layers, n_embd, forward_expansion,seq_len,seq_len_dec, mean, std,dropout = 0.0, bias = False):
        super(EnhancedTransformerData, self).__init__()

        self.encoder_wte = nn.Linear(input_dim, n_embd).to(device)
        self.encoder_wpe = nn.Embedding(seq_len, n_embd).to(device)
        
        self.decoder_wte = nn.Linear(input_dim -9, n_embd).to(device)
        self.decoder_wpe = nn.Embedding(seq_len_dec, n_embd).to(device)   ## From the physics informed ones and maybe useless, but comes from the physics informed where it has been added energy and so on.
        
        self.positional_encoding_enc = StdPositionalEncoding(n_embd, mean, std,max_len = seq_len).to(device)
        self.positional_encoding_dec = StdPositionalEncoding(n_embd, mean, std,max_len = seq_len_dec).to(device)
        
        self.norm1 = LayerNorm(n_embd, bias=bias).to(device) 

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(n_embd, n_heads, forward_expansion, dropout, bias) for _ in range(n_layers)]
        ).to(device)
        
        self.decoder_output = nn.Linear(n_embd, 3, bias= True).to(device)  # Output layer for torque prediction
        
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

    def forward(self, x, decoder_input):
        
        x = self.EncoderEmbeding(x)
        decoder_input = self.DecoderEmbedding(decoder_input)
                
        x = self.positional_encoding_enc(x)
        decoder_input = self.positional_encoding_dec(decoder_input)
        
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm1(x)

        for layer in self.decoder_layers:
            decoder_output = layer(x, decoder_input)
        decoder_output = self.norm1(decoder_output)
        
        return self.decoder_output(decoder_output)  # Predict based on the last time step
