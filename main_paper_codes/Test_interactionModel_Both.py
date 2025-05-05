import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.signal import savgol_filter
from PITransformer_interaction_model import EnhancedTransformer , PhysicsBasedLoss # Import the model definition
from DataDriven_interaction_model import EnhancedTransformerData

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
file_path = 'test_IDSIA7.csv'  # Adjust based on the actual filename

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
force_x = ['force_x']
force_y = ['force_y']
force_z = ['force_z']

relevant_columns = joint_x + joint_y + joint_z + target_x + target_y + target_z + speed_x + speed_y + speed_z
relevant_outputs = force_x + force_y + force_z

# Filter the DataFrame for relevant columns and drop NaN values
data = df[relevant_columns].dropna()
data_outputs = df[relevant_outputs].dropna()
data_relevant = data.copy()

# Calculate acceleration using central difference
delta_t = 0.001  # Effective time step after downsampling
velocity_x = data['velocity_x'].to_numpy()
velocity_y = data['velocity_y'].to_numpy()
velocity_z = data['velocity_z'].to_numpy()
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


# downsampled_data = pd.concat([downsampled_data.iloc[:,:3], downsampled_data.iloc[:,9:]], ignore_index=True)
data = data.iloc[:,3:6]

position_x = data['target_position_x'].to_numpy()
position_y = data['target_position_y'].to_numpy()
position_z = data['target_position_z'].to_numpy()
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
    
# Smoothen the calculated velocities
data['target_velocity_x'] = savgol_filter(target_velocity_x, window_length=500, polyorder=3)
data['target_velocity_y'] = savgol_filter(target_velocity_y, window_length=500, polyorder=3)
data['target_velocity_z'] = savgol_filter(target_velocity_z, window_length=500, polyorder=3)

data_relevant['target_velocity_x'] = savgol_filter(target_velocity_x, window_length=500, polyorder=3)
data_relevant['target_velocity_y'] = savgol_filter(target_velocity_y, window_length=500, polyorder=3)
data_relevant['target_velocity_z'] = savgol_filter(target_velocity_z, window_length=500, polyorder=3)

data_relevant['acceleration_x'] = acceleration_x
data_relevant['acceleration_y'] = acceleration_y
data_relevant['acceleration_z'] = acceleration_z
data_relevant['force_x'] = data_outputs['force_x'].copy()
data_relevant['force_y'] = data_outputs['force_y'].copy()
data_relevant['force_z'] = data_outputs['force_z'].copy()

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

# Smoothen the calculated accelerations   
data['target_acceleration_x'] = savgol_filter(target_acceleration_x, window_length=500, polyorder=3)
data['target_acceleration_y'] = savgol_filter(target_acceleration_y, window_length=500, polyorder=3)
data['target_acceleration_z'] = savgol_filter(target_acceleration_z, window_length=500, polyorder=3)


relevant_mean = np.load('relevant_mean.npy')
relevant_std = np.load('relevant_std.npy')

joint_data = data
joint_outputs = data_outputs
joint_relevant = data_relevant 

# Create sequences with n_future prediction
def create_sequences(data, seq_length, output_seq_len):
    sequences = []
    for i in range(0,len(data) - seq_length, 30):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def create_output_sequences(data, seq_length, output_len):
    sequences = []
    for i in range(output_len,len(data) - seq_length, 30):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Updated model output sequence length
output_seq_len = 240 # Predict next 10 timesteps
seq_length = 352
X = create_sequences(joint_relevant, seq_length, output_seq_len)
X_decoder = create_output_sequences(joint_data, seq_length, output_seq_len)
Y = create_output_sequences(joint_outputs, seq_length,output_seq_len)

X = X[:Y.shape[0], :, :]
X_decoder = X_decoder[:Y.shape[0],:,:]

#comment the following if you want to test on the complete dataset
X, c, Y,d, X_decoder,e = train_test_split(
    X, Y, X_decoder, test_size=0.3, random_state=42, shuffle= False)


# Load learned parameters from the JSON file
with open('learned_params_saved.json', 'r') as f:
    learned_params = json.load(f)

lower_bounds = {
    'J': [1e-6, 1e-6, 1e-6],  # Lower bounds for inertia across X, Y, Z
    'b': [1e-6, 1e-6, 1e-6],  # Lower bounds for damping across X, Y, Z
    'k': [1e-6, 1e-6, 1e-6],  # Lower bounds for stiffness across X, Y, Z
    'R': [1e-6, 1e-6, 1e-6]   # Lower bounds for random offset across X, Y, Z
}

test_dataset = TensorDataset(
    torch.FloatTensor(c),
    torch.FloatTensor(e),
    torch.FloatTensor(d)
)

batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Reinitialize the model with the same architecture
embed_dim = 128
num_heads = 4

criterion = PhysicsBasedLoss(lambda_phy=0.1, initial_params=learned_params, lower_bounds=lower_bounds).to(device)

physics_params = {
    "J": criterion.J,
    "b": criterion.b,
    "k": criterion.k,
    "R": criterion.R
}

model = EnhancedTransformer(input_dim=18, n_heads=num_heads, n_layers=8, n_embd=embed_dim, forward_expansion=6, 
                            seq_len= seq_length,seq_len_dec= seq_length, mean= relevant_mean, std= relevant_std, physics_params= physics_params) .to(device)

model_data = EnhancedTransformerData(input_dim= 18, n_heads= num_heads, n_layers= 8, n_embd= embed_dim, forward_expansion= 6,
                                     seq_len= seq_length, mean= relevant_mean, std= relevant_std).to(device)

# Load the model weights
model.load_state_dict(torch.load('Interaction_metamodel_physics_test.pth', weights_only= True)) 
model_data.load_state_dict(torch.load('Interaction_metamodel_data_test.pth', weights_only= True))


predictions = []
predictions_data = []
y_test_all = []
inertia = []
stiffness = []
damping = []
random = []

with torch.no_grad():
    total_loss = 0.0
    total_loss_data = 0.0
       
    for X_batch, X_decoder_batch, Y_batch in test_loader:
      
        X_batch = X_batch.to(device)
        X_decoder_batch = X_decoder_batch.to(device)
        Y_batch = Y_batch.to(device)
        Forces = X_batch[:,:,15:18].to(device)
        
        
        seq_mean = X_batch.mean(dim=1, keepdims=True)
        seq_std = X_batch.std(dim=1, keepdims = True) +1e-8
        seq_mean_decoder = X_decoder_batch.mean(dim=1, keepdims=True)
        seq_std_decoder = X_decoder_batch.std(dim=1, keepdims = True)+1e-8
        
        seq_mean_out = Forces.mean(dim=1, keepdims= True)
        seq_std_out = Forces.std(dim=1, keepdims= True) + 1e-8
        
        std_positions = seq_std[:,:,0:3].mean(dim = 0, keepdims = True)
        std_target_positions = seq_std[:,:,3:6].mean(dim = 0, keepdims = True)
        std_forces = seq_std[:,:,15:18].mean(dim = 0, keepdims = True)
        
        X_batch_norm = (X_batch- seq_mean)/seq_std 
        X_decoder_batch_norm = (X_decoder_batch - seq_mean_decoder)/seq_std_decoder 
        Y_batch_norm = (Y_batch - seq_mean[:,:,-3:])/ seq_std[:,:,-3:]
        
        # Prepare decoder inputs
        positions = X_batch_norm[:, :, 0:3]
        target_positions = X_batch_norm[:, :, 3:6]
        velocities = X_batch_norm[:, :, 6:9]
        target_velocities = X_batch_norm[:,:,9:12]
        accelerations = X_batch_norm[:, :, 12:15]
        torques = X_batch_norm[:,:,15:18]
        
        positions_next = X_decoder_batch_norm[:,:,0:3].to(device)
        velocities_next = X_decoder_batch_norm[:,:,3:6].to(device)
        accelerations_next = X_decoder_batch_norm[:,:,6:9].to(device)
        # Forward pass
        output, J,b,k,R = model(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities, target_velocities, accelerations, torques,
                       positions_next, velocities_next, accelerations_next)
                
        stiffness.append(k.detach().cpu().numpy())
        inertia.append(J.detach().cpu().numpy())
        damping.append(b.detach().cpu().numpy())
        random.append(R.detach().cpu().numpy())

        output_data = model_data(X_batch_norm, X_decoder_batch_norm)

        output = output * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
        output_data = output_data * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
        Y_batch = Y_batch_norm * seq_std[:, :, -3:] + seq_mean[:,:, -3:]
        
        # Collect predictions and ground truth
        predictions.append(output[:, seq_length-output_seq_len:, :].cpu().numpy())
        predictions_data.append(output_data[:, seq_length-output_seq_len:, :].cpu().numpy())
        y_test_all.append(Y_batch[:, seq_length-output_seq_len:, :].cpu().numpy())
        
print('------------------------------------------')


# Convert results to NumPy
predictions = np.concatenate(predictions, axis=0)
predictions_data = np.concatenate(predictions_data, axis=0)
y_test_np = np.concatenate(y_test_all, axis=0)

learned_params_new = {
       'inertia': J.detach().cpu().tolist(),  # Convert to list for JSON compatibility
        'damping': b.detach().cpu().tolist(),
        'stiffness': k.detach().cpu().tolist(),
        'random': R.detach().cpu().tolist()
    }


print('-----------')
print('new physical parameters')
print(learned_params_new)
print('---------------------')

def stitch_predictions(preds, stride=30):
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

# Stitch predictions for each model:
stitched_pred_model1 = stitch_predictions(predictions)
stitched_pred_data   = stitch_predictions(predictions_data)
stitched_y_test = stitch_predictions(y_test_np)


print('--- Metrics on Stitched Time Series (each time step unique) ---')
for i, axis in enumerate(['X','Y','Z']):
    mse_val = mean_squared_error(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    rmse_val = root_mean_squared_error(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    mae_val = mean_absolute_error(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    r2_val  = r2_score(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    print(f"Physics-Driven - {axis} Axis: MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
    
    mse_val_data = mean_squared_error(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    rmse_val_data = root_mean_squared_error(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    mae_val_data = mean_absolute_error(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    r2_val_data  = r2_score(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    print(f"Data-Driven  - {axis} Axis: MSE: {mse_val_data:.4f}, RMSE: {rmse_val_data:.4f}, MAE: {mae_val_data:.4f}, R²: {r2_val_data:.4f}")
    print("-----------------------------------------------------------")

plt.figure(figsize=(14,5))
plt.plot(stitched_y_test[start:, 2], '--', label='Ground Truth')
plt.plot(stitched_pred_model1[start:, 2], label='Physics-Based')
plt.plot(stitched_pred_data[start:, 2], label='Data-Driven')
plt.xlabel('Time Step', fontsize = 18)
plt.ylabel('Force', fontsize = 18)
plt.title('Interaction Forces- Z Axis', fontsize = 22)
plt.legend(fontsize=16, loc = 'lower right')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.savefig('interaction_forces_z.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

