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
from DataDrivenInteractionModel import EnhancedTransformerData

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
file_path = 'test_IDSIA3.csv'  # Adjust based on the actual filename
ekf_path = 'figure200_data_IDSIA3.csv'
df = pd.read_csv(file_path)
df_ekf = pd.read_csv(ekf_path)

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

ekf_output = ['pred']

relevant_columns = joint_x + joint_y + joint_z + target_x + target_y + target_z + speed_x + speed_y + speed_z
# relevant_targets = target_x + target_y + target_z
relevant_outputs = torque_x + torque_y + torque_z

ekf_outputs = ekf_output

# time_df = time


# Filter the DataFrame for relevant columns and drop NaN values
data = df[relevant_columns].dropna()
data_outputs = df[relevant_outputs].dropna()
data_ekf = df_ekf[ekf_output].dropna()

# Downsampling factor (e.g., keep every 5th sample)
downsampling_factor = 1

# Downsample the data
downsampled_data = data.iloc[::downsampling_factor, :].reset_index(drop=True)
downsampled_output = data_outputs.iloc[::downsampling_factor, :].reset_index(drop=True)
downsampled_ekf = data_ekf.iloc[::downsampling_factor, :].reset_index(drop=True)
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

print(downsampled_data.shape)
##relevant data cleaning

# downsampled_data = pd.concat([downsampled_data.iloc[:,:3], downsampled_data.iloc[:,9:]], ignore_index=True)
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

# Normalize the data (normalize sequences individually)
# mean = (downsampled_data.mean(axis=0)).to_numpy()
# std = (downsampled_data.std(axis=0)).to_numpy()
joint_data = downsampled_data

# mean_output = (downsampled_output.mean(axis=0)).to_numpy()
# std_output = (downsampled_output.std(axis=0)).to_numpy()
joint_outputs = downsampled_output

# relevant_mean = (downsampled_relevant.mean(axis=0)).to_numpy()
# relevant_std = (downsampled_relevant.std(axis=0)).to_numpy()
joint_relevant = downsampled_relevant 

# print(Train_relevant.shape)
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

# Create decoder inputs from ground truth outputs
def create_decoder_inputs(data, seq_length, output_len): #better to not overlap the sequences, try to use continuous sequences
    # Create sequences
    sequences = []
    for i in range(output_len, len(data) - seq_length, seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Create decoder inputs from ground truth outputs
def create_encoder_inputs(data, seq_length, output_len): #better to not overlap the sequences, try to use continuous sequences
    sequences = []
    for i in range(0,len(data) - seq_length, seq_length):
        sequences.append(data[i:i +seq_length])
    return np.array(sequences)

# Updated model output sequence length
output_seq_len = 240 # Predict next 10 timesteps
seq_length = 352
X = create_sequences(joint_relevant, seq_length, output_seq_len)
X_decoder = create_output_sequences(joint_data, seq_length, output_seq_len)
Y = create_output_sequences(joint_outputs, seq_length,output_seq_len)
# ekf_sequences = create_sequences(downsampled_ekf)
print(X.shape)
print(X_decoder.shape)
print(Y.shape)

X = X[:Y.shape[0], :, :]
X_decoder = X_decoder[:Y.shape[0],:,:]

X, c, Y,d, X_decoder,e = train_test_split(
    X, Y, X_decoder, test_size=0.3, random_state=42, shuffle= False)


# Load learned parameters from the JSON file
with open('learned_params_IDSIA_2_4_5_6__again.json', 'r') as f:
    learned_params = json.load(f)

lower_bounds = {
    'J': [1e-6, 1e-6, 1e-6],  # Lower bounds for inertia across X, Y, Z
    'b': [1e-6, 1e-6, 1e-6],  # Lower bounds for damping across X, Y, Z
    'k': [1e-6, 1e-6, 1e-6],  # Lower bounds for stiffness across X, Y, Z
    'R': [1e-6, 1e-6, 1e-6]   # Lower bounds for random offset across X, Y, Z
}

print(learned_params)

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

criterion = PhysicsBasedLoss(lambda_phy=0.6, initial_params=learned_params, lower_bounds=lower_bounds).to(device)

physics_params = {
    "J": criterion.J,
    "b": criterion.b,
    "k": criterion.k,
    "R": criterion.R
}

model = EnhancedTransformer(input_dim=18, n_heads=num_heads, n_layers=8, n_embd=embed_dim, forward_expansion=6, 
                            seq_len= seq_length, mean= relevant_mean, std= relevant_std, physics_params= physics_params) .to(device)

# model2 = EnhancedTransformer(input_dim=18, n_heads=num_heads, n_layers=8, n_embd=embed_dim, forward_expansion=6, 
#                             seq_len= seq_length, mean= relevant_mean, std= relevant_std, physics_params= physics_params) .to(device)

model_data = EnhancedTransformerData(input_dim= 18, n_heads= num_heads, n_layers= 8, n_embd= embed_dim, forward_expansion= 6,
                                     seq_len= seq_length, mean= relevant_mean, std= relevant_std).to(device)



# Load the model weights
#try physics without the physics loss
model.load_state_dict(torch.load('Interaction_metamodel_physics_test.pth', weights_only= True)) #Interaction_metamodel_physicsIDSIA_2_4_5_6_again_lambdaPhys0_1
# model2.load_state_dict(torch.load('Interaction_model_physicsIDSIA2_actual.pth', weights_only= True))
model_data.load_state_dict(torch.load('Interaction_metamodel_data_IDSIA_2_4_5_6_again.pth', weights_only= True))

# model.eval()
# model2.eval()
model_data.eval()

predictions = []
predictions2 = []
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
        # target_positions_next = X_decoder_batch[:,:,3:6].to(device)
        velocities_next = X_decoder_batch_norm[:,:,3:6].to(device)
        accelerations_next = X_decoder_batch_norm[:,:,6:9].to(device)
        # start_time = time.time()
        # Forward pass
        output, J,b,k,R = model(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities, target_velocities, accelerations, torques,
                       positions_next, velocities_next, accelerations_next)
        # print(f'Inference time - Physics = {time.time() - start_time}')
        
        # copy_torques = output.clone()
        
        stiffness.append(k.detach().cpu().numpy())
        inertia.append(J.detach().cpu().numpy())
        damping.append(b.detach().cpu().numpy())
        random.append(R.detach().cpu().numpy())
        # output2 = model2(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities, target_velocities, accelerations, torques,
        #                positions_next, velocities_next, accelerations_next)
        start_time_data = time.time()
        output_data = model_data(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities, accelerations)
        print(f'Inference time - Data = {time.time() - start_time_data}')
        loss = criterion(output[:, seq_length-output_seq_len:, :].squeeze(), Y_batch_norm[:, seq_length-output_seq_len:, :].squeeze(), positions[:, seq_length-output_seq_len:, :], target_positions[:, seq_length-output_seq_len:, :],
                         velocities[:, seq_length-output_seq_len:, :], target_velocities[:, seq_length-output_seq_len:, :], accelerations[:, seq_length-output_seq_len:, :],
                         J, b, k, R).to(device)
        
        output = output * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
        # output2 = output2 * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
        output_data = output_data * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
        Y_batch = Y_batch_norm * seq_std[:, :, -3:] + seq_mean[:,:, -3:]
        
        # Collect predictions and ground truth
        predictions.append(output[:, seq_length-output_seq_len:, :].cpu().numpy())
        # predictions2.append(output2.cpu().numpy())
        predictions_data.append(output_data[:, seq_length-output_seq_len:, :].cpu().numpy())
        y_test_all.append(Y_batch[:, seq_length-output_seq_len:, :].cpu().numpy())
        
print('------------------------------------------')


# Convert results to NumPy
predictions = np.concatenate(predictions, axis=0)
# predictions2 = np.concatenate(predictions2, axis=0)
predictions_data = np.concatenate(predictions_data, axis=0)
y_test_np = np.concatenate(y_test_all, axis=0)
print(y_test_np.shape)
print(f'EKF Shape = {downsampled_ekf.shape}')

learned_params_loss = {
       'inertia': criterion.J.detach().cpu().tolist(),  # Convert to list for JSON compatibility
        'damping': criterion.b.detach().cpu().tolist(),
        'stiffness': criterion.k.detach().cpu().tolist(),
        'random': criterion.R.detach().cpu().tolist()
    }

learned_params_new = {
       'inertia': J.detach().cpu().tolist(),  # Convert to list for JSON compatibility
        'damping': b.detach().cpu().tolist(),
        'stiffness': k.detach().cpu().tolist(),
        'random': R.detach().cpu().tolist()
    }


print('-----------')
print('new physical parameters')
print(learned_params_new)
print('-----------')
print('new physical parameters - Loss')
print(learned_params_loss)
print('---------------------')
print('denormalized stiffness')
# print(((k*std_forces)/std_positions))
print(((k*seq_std_out[:, :, -3:])/(seq_std[:,:,0:3] - seq_std[:,:,3:6])).mean(dim=0)[0,-1])



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
# stitched_pred_model2 = stitch_predictions(predictions2)
stitched_pred_data   = stitch_predictions(predictions_data)

# If you have the original continuous ground truth,
# you can stitch it the same way (if you built it from overlapping sequences).
stitched_y_test = stitch_predictions(y_test_np)


print('--- Metrics on Stitched Time Series (each time step unique) ---')
for i, axis in enumerate(['X','Y','Z']):
    mse_val = mean_squared_error(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    rmse_val = root_mean_squared_error(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    mae_val = mean_absolute_error(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    r2_val  = r2_score(stitched_y_test[start:, i], stitched_pred_model1[start:, i])
    print(f"Physics-Driven - {axis} Axis: MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
    
    # mse_val2 = mean_squared_error(stitched_y_test[start:, i], stitched_pred_model2[start:, i])
    # mae_val2 = mean_absolute_error(stitched_y_test[start:, i], stitched_pred_model2[start:, i])
    # r2_val2  = r2_score(stitched_y_test[start:, i], stitched_pred_model2[start:, i])
    # print(f"Physics 2 - {axis} Axis: MSE: {mse_val2:.4f}, MAE: {mae_val2:.4f}, R²: {r2_val2:.4f}")
    
    mse_val_data = mean_squared_error(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    rmse_val_data = root_mean_squared_error(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    mae_val_data = mean_absolute_error(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    r2_val_data  = r2_score(stitched_y_test[start:, i], stitched_pred_data[start:, i])
    print(f"Data-Driven  - {axis} Axis: MSE: {mse_val_data:.4f}, RMSE: {rmse_val_data:.4f}, MAE: {mae_val_data:.4f}, R²: {r2_val_data:.4f}")
    
    mse_val_ekf = mean_squared_error(stitched_y_test[start:len(downsampled_ekf), i], downsampled_ekf)
    rmse_val_ekf = root_mean_squared_error(stitched_y_test[start:len(downsampled_ekf), i], downsampled_ekf)
    mae_val_ekf = mean_absolute_error(stitched_y_test[start:len(downsampled_ekf), i], downsampled_ekf)
    r2_val_ekf  = r2_score(stitched_y_test[start:len(downsampled_ekf), i], downsampled_ekf)
    print(f"EKF  - {axis} Axis: MSE: {mse_val_ekf:.4f}, RMSE: {rmse_val_ekf:.4f}, MAE: {mae_val_ekf:.4f}, R²: {r2_val_ekf:.4f}")
    print("-----------------------------------------------------------")

z_stiffness = [axis_stiffness[2] for axis_stiffness in stiffness]
plt.figure(figsize=(14,5))
plt.plot(z_stiffness, label='Predicted stiffness - (Z)')
# plt.plot(stiffness_loss[2], label='Predicted stiffness Grad - (Z)')
# plt.plot(stitched_y_test[3000:, 2] - stitched_pred_data[3000:, 2], label='Data-Driven (Z)')
plt.xlabel('Time Step')
plt.ylabel('Force')
# plt.ylim(0,50)
plt.title('Prediction Error Percentage - Z Axis')
plt.legend()
plt.grid(True)
plt.show()

print(f'stitched shape = {stitched_y_test[:len(downsampled_ekf),2].shape}')

plt.figure(figsize=(14,5))
plt.plot(stitched_y_test[start:, 2], '--', label='Ground Truth')
plt.plot(stitched_pred_model1[start:, 2], label='Physics-Based')
plt.plot(stitched_pred_data[:, 2], label='Data-Driven')
plt.plot(downsampled_ekf, label='EKF')
plt.xlabel('Time Step', fontsize = 18)
plt.ylabel('Force', fontsize = 18)
plt.title('Interaction Forces- Z Axis', fontsize = 22)
# Setting legend with specified font size
plt.legend(fontsize=16, loc = 'lower right')
# Adjusting tick label font sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim(5.7,12.2) #for idsia7
# plt.legend()
plt.grid(True)
plt.savefig('interaction_forces_z.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(14,5))
plt.plot(stitched_y_test[start:, 2], '--', label='Ground Truth (Z)')
plt.plot(stitched_pred_model1[start:, 2], label='Physics-Based (Z)')
# plt.plot(stitched_pred_model2[start:, 2], label='Physics-Based - lambda =0.5(Z)')
# plt.plot(stitched_pred_data[start:, 2], label='Data-Driven (Z)')
plt.xlabel('Time Step')
plt.ylabel('Force')
plt.title('Interaction Forces- Z Axis')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,5))
plt.plot(stitched_y_test[start:, 2], '--', label='Ground Truth (Z)')
# plt.plot(stitched_pred_model1[start:, 2], label='Physics-Based - lambda =0.8 (Z)')
# plt.plot(stitched_pred_model2[start:, 2], label='Physics-Based - lambda =0.5(Z)')
plt.plot(stitched_pred_data[start:, 2], label='Data-Driven (Z)')
plt.xlabel('Time Step')
plt.ylabel('Force')
plt.title('Interaction Forces - Z Axis')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,5))
plt.plot([],[] , label='_nolegend_')
plt.plot(stitched_y_test[start:len(downsampled_ekf), 2] - stitched_pred_model1[start:len(downsampled_ekf), 2], label='Physics-based')
plt.plot(stitched_y_test[start:len(downsampled_ekf), 2] - stitched_pred_data[start:len(downsampled_ekf), 2], label='Data-Driven')
plt.plot(stitched_y_test[:downsampled_ekf.squeeze().shape[0], 2] - downsampled_ekf.squeeze(), label='EKF')
plt.xlabel('Time Step', fontsize=18)
plt.ylabel('Force', fontsize=18)
plt.title('Prediction Error - Z Axis', fontsize=22)
plt.legend(fontsize=16, loc= 'lower right')
# Adjusting tick label font sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.savefig('estimation_error_ramp.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()


plt.figure(figsize=(14,5))
plt.plot((stitched_y_test[start:, 2] - stitched_pred_model1[start:, 2])*100/stitched_y_test[start:,2], label='Predicted Force Error Percentage - Physics-Based (Z)')
# plt.plot(stitched_y_test[3000:, 2] - stitched_pred_data[3000:, 2], label='Data-Driven (Z)')
plt.xlabel('Time Step')
plt.ylabel('Force')
plt.ylim(-30,30)
plt.title('Prediction Error Percentage - Z Axis')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,5))
plt.plot((stitched_y_test[start:, 2] - stitched_pred_data[start:, 2])*100/stitched_y_test[start:,2], label='Predicted Force Error Percentage - Data-Based (Z)')
# plt.plot(stitched_y_test[3000:, 2] - stitched_pred_data[3000:, 2], label='Data-Driven (Z)')
plt.xlabel('Time Step')
plt.ylabel('Force')
plt.ylim(-30,30)
plt.title('Prediction Error Percentage - Z Axis')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,5))
plt.plot((stitched_y_test[start:, 2] - stitched_pred_model1[start:, 2])*100/stitched_y_test[start:,2], label='Predicted Force Error Percentage - Physics-Based (Z)')
plt.plot((stitched_y_test[start:, 2] - stitched_pred_data[start:, 2])*100/stitched_y_test[start:,2], label='Predicted Force Error Percentage - Data-Based (Z)')
plt.xlabel('Time Step',  fontsize = 14)
plt.ylabel('Force',  fontsize = 14)
plt.ylim(-40,40)
plt.title('Prediction Error Percentage - Z Axis', fontsize = 16)
# Setting legend with specified font size
plt.legend(fontsize=14)
# Adjusting tick label font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('estimation_error_percentage_OOD.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()


# # # Calculate metrics for Data-Driven model
metrics_data = {
    'R-squared': r2_val_data,
    'MSE':mse_val_data,
    'RMSE': rmse_val_data,
    'MAE': mae_val_data
}

# Calculate metrics for Physics-Based model
metrics_physics = {
    'R-squared': r2_val,
    'MSE': mse_val,
    'RMSE': rmse_val,
    'MAE': mae_val
}

# Create a grouped bar chart for comparison
metrics_names = list(metrics_data.keys())
data_values = list(metrics_data.values())
physics_values = list(metrics_physics.values())

x = np.arange(len(metrics_names))  # the label locations
width = 0.15  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, data_values, width, label='Data-Driven')
bars2 = ax.bar(x + width/2, physics_values, width, label='Physics-Based')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Metric Value')
ax.set_title('Comparison of Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

# Optionally, add text labels above the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()