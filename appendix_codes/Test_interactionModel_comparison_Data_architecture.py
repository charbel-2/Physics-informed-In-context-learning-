import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.signal import savgol_filter

from DataDriven_interaction_model import  DecoderOnlyInteractionModel,LSTMInteractionModel,DeepSetInteractionModel,TCNInteractionModel,EnhancedTransformerData

SEED = 42


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 


# Set the device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
# file_path= '../Datasets/Test_Sponge1.csv'  
# file_path= '../Datasets/Test_Sponge2.csv'  
# file_path= '../Datasets/Test_Sponge3.csv'  
file_path= '../Datasets/Test_Table.csv'  # Table csv
df = pd.read_csv(file_path)

start = 0

# Select relevant columns for modeling: positions, velocities, and torques
joint_x = ['franka_ee_pose_x']
joint_y = ['franka_ee_pose_y']
joint_z = ['franka_ee_pose_z']
target_x = ['target_position_x']
target_y = ['target_position_y']
target_z = ['target_position_z']
speed_x = ['velocity_x']
speed_y = ['velocity_y']
speed_z = ['velocity_z']
torque_x = ['force_x']
torque_y = ['force_y']
torque_z = ['force_z']
# time = ['time']

relevant_columns = joint_x + joint_y + joint_z + target_x + target_y + target_z + speed_x + speed_y + speed_z
relevant_outputs = torque_x + torque_y + torque_z


# Filter the DataFrame for relevant columns and drop NaN values
data = df[relevant_columns].dropna()
data_outputs = df[relevant_outputs].dropna()

# Downsampling factor (e.g., keep every 5th sample)
downsampling_factor = 1

# Downsample the data
downsampled_data = data.iloc[::downsampling_factor, :].reset_index(drop=True)
downsampled_output = data_outputs.iloc[::downsampling_factor, :].reset_index(drop=True)
downsampled_relevant = downsampled_data.copy()
print('-----------------')
print('relevant:')
print(downsampled_data.shape)
print(downsampled_relevant.shape)
print('--------------------')


# Calculate acceleration using central difference
delta_t = 0.001 * downsampling_factor  # Effective time step after downsampling
velocity_x = downsampled_data['velocity_x'].to_numpy()
velocity_y = downsampled_data['velocity_y'].to_numpy()
velocity_z = downsampled_data['velocity_z'].to_numpy()
acceleration_x = np.zeros_like(velocity_x)
acceleration_y = np.zeros_like(velocity_x)
acceleration_z = np.zeros_like(velocity_x)

# Compute central difference for acceleration
acceleration_x[1:-1] = (velocity_x[2:] - velocity_x[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
acceleration_x[0] = (velocity_x[1] - velocity_x[0]) / (delta_t)
acceleration_x[-1] = (velocity_x[-1] - velocity_x[-2]) / (delta_t)

acceleration_y[1:-1] = (velocity_y[2:] - velocity_y[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
acceleration_y[0] = (velocity_y[1] - velocity_y[0]) / (delta_t)
acceleration_y[-1] = (velocity_y[-1] - velocity_y[-2]) / (delta_t)

acceleration_z[1:-1] = (velocity_z[2:] - velocity_z[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
acceleration_z[0] = (velocity_z[1] - velocity_z[0]) / (delta_t)
acceleration_z[-1] = (velocity_z[-1] - velocity_z[-2]) / (delta_t)

# Add acceleration to the DataFrame

downsampled_data = downsampled_data.iloc[:,3:6]

position_x = downsampled_data['target_position_x'].to_numpy()
position_y = downsampled_data['target_position_y'].to_numpy()
position_z = downsampled_data['target_position_z'].to_numpy()
target_velocity_x = np.zeros_like(position_x)
target_velocity_y = np.zeros_like(position_x)
target_velocity_z = np.zeros_like(position_x)

# Compute central difference for acceleration
target_velocity_x[1:-1] = (position_x[2:] - position_x[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
target_velocity_x[0] = (position_x[1] - position_x[0]) / (delta_t)
target_velocity_x[-1] = (position_x[-1] - position_x[-2]) / (delta_t)

target_velocity_y[1:-1] = (position_y[2:] - position_y[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
target_velocity_y[0] = (position_y[1] - position_y[0]) / (delta_t)
target_velocity_y[-1] = (position_y[-1] - position_y[-2]) / (delta_t)

target_velocity_z[1:-1] = (position_z[2:] - position_z[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
target_velocity_z[0] = (position_z[1] - position_z[0]) / (delta_t)
target_velocity_z[-1] = (position_z[-1] - position_z[-2]) / (delta_t)
    
downsampled_data['target_velocity_x'] = savgol_filter(target_velocity_x, window_length=500, polyorder=3)
downsampled_data['target_velocity_y'] = savgol_filter(target_velocity_y, window_length=500, polyorder=3)
downsampled_data['target_velocity_z'] = savgol_filter(target_velocity_z, window_length=500, polyorder=3)

downsampled_relevant['target_velocity_x'] = savgol_filter(target_velocity_x, window_length=500, polyorder=3)
downsampled_relevant['target_velocity_y'] = savgol_filter(target_velocity_y, window_length=500, polyorder=3)
downsampled_relevant['target_velocity_z'] = savgol_filter(target_velocity_z, window_length=500, polyorder=3)

downsampled_relevant['acceleration_x'] = acceleration_x
downsampled_relevant['acceleration_y'] = acceleration_y
downsampled_relevant['acceleration_z'] = acceleration_z
downsampled_relevant['torque_x'] = downsampled_output['force_x'].copy()
downsampled_relevant['torque_y'] = downsampled_output['force_y'].copy()
downsampled_relevant['torque_z'] = downsampled_output['force_z'].copy()

target_acceleration_x = np.zeros_like(position_x)
target_acceleration_y = np.zeros_like(position_x)
target_acceleration_z = np.zeros_like(position_x)

# Compute central difference for acceleration
target_acceleration_x[1:-1] = (target_velocity_x[2:] - target_velocity_x[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
target_acceleration_x[0] = (target_velocity_x[1] - target_velocity_x[0]) / (delta_t)
target_acceleration_x[-1] = (target_velocity_x[-1] - target_velocity_x[-2]) / (delta_t)

target_acceleration_y[1:-1] = (target_velocity_y[2:] - target_velocity_y[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
target_acceleration_y[0] = (target_velocity_y[1] - target_velocity_y[0]) / (delta_t)
target_acceleration_y[-1] = (target_velocity_y[-1] - target_velocity_y[-2]) / (delta_t)

target_acceleration_z[1:-1] = (target_velocity_z[2:] - target_velocity_z[:-2]) / (2 * delta_t)
# Forward and backward differences for edges
target_acceleration_z[0] = (target_velocity_z[1] - target_velocity_z[0]) / (delta_t)
target_acceleration_z[-1] = (target_velocity_z[-1] - target_velocity_z[-2]) / (delta_t)
    
downsampled_data['target_acceleration_x'] = savgol_filter(target_acceleration_x, window_length=500, polyorder=3)
downsampled_data['target_acceleration_y'] = savgol_filter(target_acceleration_y, window_length=500, polyorder=3)
downsampled_data['target_acceleration_z'] = savgol_filter(target_acceleration_z, window_length=500, polyorder=3)

print('Data check')
print(downsampled_data.head())
print('------------------')
print(downsampled_relevant.head())


print('-----------------')
print('new relevant:')
print(downsampled_data.shape)
print(downsampled_relevant.shape)
print(type(downsampled_data))
print('--------------------')

mean = np.load('train_mean.npy')
std = np.load('train_std.npy')
mean_output= np.load('output_mean.npy')
std_output = np.load('output_std.npy')
relevant_mean = np.load('relevant_mean.npy')
relevant_std = np.load('relevant_std.npy')

joint_data = downsampled_data
joint_outputs = downsampled_output
joint_relevant = downsampled_relevant 
n_total = joint_relevant.shape[0]

# Create decoder inputs from ground truth outputs
def create_decoder_inputs(data, seq_length, output_len,q):  #before q was 352, that is the length of the decoder, in order to have the same ctx len and prediction len
    # Create sequences
    sequences = []
    for i in range(output_len+seq_length-q, len(data) - q, output_len):
        sequences.append(data[i:i + q])
    return np.array(sequences)

# Create decoder inputs from ground truth outputs
def create_encoder_inputs(data, seq_length, output_len): #better to not overlap the sequences, try to use continuous sequences
    sequences = []
    for i in range(0,len(data) - seq_length, output_len):
        sequences.append(data[i:i +seq_length])
    return np.array(sequences)


# Updated model output sequence length
output_seq_len = 240
seq_length = 352
len_dec = 352
X = create_encoder_inputs(joint_relevant, seq_length, output_seq_len)
Y = create_decoder_inputs(joint_outputs, seq_length, output_seq_len,len_dec)
X_decoder = create_decoder_inputs(joint_data, seq_length, output_seq_len,len_dec)

print(X.shape)
print(X_decoder.shape)
print(Y.shape)

X = X[:Y.shape[0], :, :]
X_decoder = X_decoder[:Y.shape[0],:,:]

# 80% of training
X_train, X_valts, Y_train, Y_valts, X_decoder_train, X_decoder_valts = train_test_split(
        X, Y, X_decoder, test_size=0.2, random_state=42, shuffle= False)
    
# 10% validation & 10% test
X_val, X_test, Y_val, Y_test, X_decoder_val, X_decoder_test = train_test_split(
    X_valts, Y_valts, X_decoder_valts, test_size=0.5, random_state=42, shuffle= False)

test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.FloatTensor(X_decoder_test),
    torch.FloatTensor(Y_test)
)

batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Reinitialize the model with the same architecture
embed_dim = 128
num_heads = 4

# Different architecture chosen for the architecture comparison
models = {
    "Transformer": EnhancedTransformerData(input_dim= 18, n_heads= 4, n_layers= 8, n_embd= 128, forward_expansion= 6,
                                     seq_len= seq_length,seq_len_dec = len_dec, mean= relevant_mean, std= relevant_std).to(device),  # Plug your transformer here
    "Decoder-only": DecoderOnlyInteractionModel(input_dim=15,  embed_dim= embed_dim, num_heads=num_heads,num_layers= 8, mean= relevant_mean, std= relevant_std).to(device),
    "LSTM": LSTMInteractionModel(input_dim=15, hidden_dim=256, num_layers = 4).to(device),
    "DeepLSTM": LSTMInteractionModel(input_dim=15, hidden_dim=256, num_layers=8).to(device),
    "DeepSets": DeepSetInteractionModel(input_dim=15,hidden_dim=256).to(device),
    "TCN": TCNInteractionModel(input_dim=15, hidden_dim=256, num_layers= 8).to(device),
}

# coulurs for the plots
colour = ['r','g','c','m']

# index
i = 0

# vectors of the all predictions for each model
predictions_data = []
for name, model_data in models.items():
    total_mse, total_rmse, total_mae, total_r2 = 0.0, 0.0, 0.0, 0.0
    total_samples = 0
    i+=1
    
    model_data.load_state_dict(torch.load(f'comparison_Data_architecture_ckpts/Interaction_metamodel_data_{name}_test_right_all4.pth', weights_only= True))
    model_data.eval()

    predictions = []
    y_test_all = []

    with torch.no_grad():
        total_loss = 0.0
        total_loss_data = 0.0
        
        for X_batch, X_decoder_batch, Y_batch in test_loader:
        
            X_batch = X_batch.to(device)
            X_decoder_batch = X_decoder_batch.to(device)
            Y_batch = Y_batch.to(device)
             
            seq_mean = X_batch.mean(dim=1, keepdims=True)
            seq_std = X_batch.std(dim=1, keepdims = True) +1e-8
            seq_mean_decoder = X_decoder_batch.mean(dim=1, keepdims=True)
            seq_std_decoder = X_decoder_batch.std(dim=1, keepdims = True)+1e-8
            
            X_batch_norm = (X_batch- seq_mean)/seq_std 
            X_decoder_batch_norm = (X_decoder_batch - seq_mean_decoder)/seq_std_decoder 
            Y_batch_norm = (Y_batch - seq_mean[:,:,-3:])/ seq_std[:,:,-3:]
            
            # Prepare decoder inputs
            if name == "Transformer":
                output_data = model_data(X_batch_norm, X_decoder_batch_norm)
            else:
                output_data = model_data(X_batch_norm[:,:,:-3],X_batch_norm[:,:,-3:], X_decoder_batch_norm)
            
            output_data = output_data[:, len_dec-output_seq_len:, :] * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
            Y_batch = Y_batch_norm[:, len_dec-output_seq_len:, :] * seq_std[:, :, -3:] + seq_mean[:,:, -3:]
            predictions.append(output_data.cpu().numpy())
            y_test_all.append(Y_batch.cpu().numpy())

            
    print('------------------------------------------')
    predictions = np.concatenate(predictions, axis=0)
    y_test_np = np.concatenate(y_test_all, axis=0)
    print(predictions.shape)
    
    def stitch_predictions(preds, stride=240):
        """
        Given an array preds of shape (n_sequences, seq_length, features),
        reconstruct the original continuous time series by averaging the overlapping predictions.
    
        Parameters:
        - preds: np.ndarray of shape (n_sequences, seq_length, features)
        - stride: int, the shift between the starting points of consecutive sequences
    
        Returns:
        - stitched: np.ndarray of shape (reconstructed_length, features)
        """
        n_seq, seq_length, n_features = preds.shape
        L = (n_seq - 1) * stride + seq_length  # Total length of the reconstructed sequence
    
        stitched = np.zeros((L, n_features))
        counts = np.zeros((L, n_features))
    
        for i in range(n_seq):
            start = i * stride
            end = start + seq_length
            stitched[start:end] += preds[i]
            counts[start:end] += 1
    
        # Avoid division by zero (in case some positions never got filled)
        counts[counts == 0] = 1
    
        return stitched / counts

    predictions_data.append(stitch_predictions(predictions))
    stitched_y_test = stitch_predictions(y_test_np)


    print('--- Metrics on Stitched Time Series (each time step unique) ---')

    j = 2 # for the z-axis
    mse_val = mean_squared_error(stitched_y_test[start:, j], stitch_predictions(predictions)[start:, j])
    rmse_val = root_mean_squared_error(stitched_y_test[start:, j], stitch_predictions(predictions)[start:, j])
    mae_val = mean_absolute_error(stitched_y_test[start:, j], stitch_predictions(predictions)[start:, j])
    r2_val  = r2_score(stitched_y_test[start:, j], stitch_predictions(predictions)[start:, j])
    print(f"{file_path} {name}  Data-Driven  - Z Axis: MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
    print("-----------------------------------------------------------")

    

    if name != 'Transformer':
        plt.figure(figsize=(14,5))
        plt.plot(stitched_y_test[start:, 2], '--', label='Ground Truth', c = 'k')

        plt.plot(predictions_data[0][start:, 2], label=f'Transformer')
        plt.plot(predictions_data[-1][start:, 2], label=f'{name}')
        plt.xlabel('Time Step', fontsize = 18)
        plt.ylabel('Force', fontsize = 18)
        plt.title('Interaction Forces- Z Axis', fontsize = 22)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.savefig(f'figures/comparison_Data_architecture/{file_path[12:-4]}_{name}_Comparison_prediction.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
