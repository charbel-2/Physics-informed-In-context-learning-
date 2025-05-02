import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.signal import savgol_filter
from PITransformer_interaction_model import EnhancedTransformer , PhysicsBasedLoss

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 


# Set the device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
file_path= 'Datasets/Test_Sponge4.csv'  
# file_path= 'Datasets/Test_Table.csv'  # Table csv
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

def create_sequences(data, seq_length, output_seq_len):
    sequences = []
    for i in range(0,len(data) - seq_length,30):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def create_output_sequences(data, seq_length, output_len, q):
    # Create sequences
    sequences = []
    for i in range(output_len+seq_length-q, len(data) - q, 30):
        sequences.append(data[i:i + q])
    return np.array(sequences)

# Updated model output sequence length
output_seq_len = 240 # Predict next 10 timesteps
seq_length = 352
len_dec = 352
X = create_sequences(joint_relevant, seq_length, output_seq_len)
X_decoder = create_output_sequences(joint_data, seq_length, output_seq_len, len_dec)
Y = create_output_sequences(joint_outputs, seq_length,output_seq_len, len_dec)
print(X.shape)
print(X_decoder.shape)
print(Y.shape)

X = X[:Y.shape[0], :, :]
X_decoder = X_decoder[:Y.shape[0],:,:]

# creation of the test subset, as last 30% of the whole dataset
X, X_test, Y,Y_test, X_decoder,X_decoder_test = train_test_split(
    X, Y, X_decoder, test_size=0.3, random_state=42, shuffle= False)



lower_bounds = {
    'J': [1e-6, 1e-6, 1e-6],  # Lower bounds for inertia across X, Y, Z
    'b': [1e-6, 1e-6, 1e-6],  # Lower bounds for damping across X, Y, Z
    'k': [1e-6, 1e-6, 1e-6],  # Lower bounds for stiffness across X, Y, Z
    'R': [1e-6, 1e-6, 1e-6]   # Lower bounds for random offset across X, Y, Z
}


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

# the vector of the different prediction for each choice of lambda
predictions_phy = []
lam = []
for lam_phy in np.arange(0.0,0.8,0.1):

    lam_phy = round(lam_phy,1)
    # Load learned parameters from the JSON file
    with open(f'fine_tune_phy_Lam_ckpts/{lam_phy}learned_params_IDSIA_2_5_6__again.json', 'r') as f:
        learned_params = json.load(f)
    criterion = PhysicsBasedLoss(lambda_phy=lam_phy, initial_params=learned_params, lower_bounds=lower_bounds).to(device)

    physics_params = {
        "J": criterion.J,
        "b": criterion.b,
        "k": criterion.k,
        "R": criterion.R
    }

    model = EnhancedTransformer(input_dim=18, n_heads=num_heads, n_layers=8, n_embd=embed_dim, forward_expansion=6, 
                                seq_len= seq_length,seq_len_dec= len_dec, mean= relevant_mean, std= relevant_std, physics_params= physics_params) .to(device)


    model.load_state_dict(torch.load(f'fine_tune_phy_Lam_ckpts/Interaction_metamodel_physics_{lam_phy}_test_right.pth', weights_only= True, map_location= torch.device('cpu'))) 

    model.eval()

    predictions = []
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
            X_decoder_batch2 = X_decoder_batch.to(device)
            X_decoder_batch = X_decoder_batch.to(device)
            Y_batch = Y_batch.to(device)
            Forces = X_batch[:,:,15:18].to(device)
            
            
            seq_mean = X_batch.mean(dim=1, keepdims=True)
            seq_std = X_batch.std(dim=1, keepdims = True) +1e-8
            seq_mean_decoder = X_decoder_batch.mean(dim=1, keepdims=True)
            seq_std_decoder = X_decoder_batch.std(dim=1, keepdims = True)+1e-8
            seq_mean_decoder2 = X_decoder_batch2.mean(dim=1, keepdims=True)
            seq_std_decoder2 = X_decoder_batch2.std(dim=1, keepdims = True)+1e-8
            seq_mean_out = Forces.mean(dim=1, keepdims= True)
            seq_std_out = Forces.std(dim=1, keepdims= True) + 1e-8
            
            std_positions = seq_std[:,:,0:3].mean(dim = 0, keepdims = True)
            std_target_positions = seq_std[:,:,3:6].mean(dim = 0, keepdims = True)
            std_forces = seq_std[:,:,15:18].mean(dim = 0, keepdims = True)
            
            
            X_batch_norm = (X_batch- seq_mean)/seq_std 
            X_decoder_batch_norm = (X_decoder_batch - seq_mean_decoder)/seq_std_decoder 
            X_decoder_batch_norm2 = (X_decoder_batch2 - seq_mean_decoder2)/seq_std_decoder2 
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
            
            # Forward pass
            output, J,b,k,R = model(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities, target_velocities, accelerations, torques,
                        positions_next, velocities_next, accelerations_next)
            
            stiffness.append(k.detach().cpu().numpy())
            inertia.append(J.detach().cpu().numpy())
            damping.append(b.detach().cpu().numpy())
            random.append(R.detach().cpu().numpy())
            
            output = output[:, len_dec-output_seq_len:, :] * seq_std[:, :, -3:] + seq_mean[:, :, -3:]
            Y_batch = Y_batch_norm[:, len_dec-output_seq_len:, :] * seq_std[:, :, -3:] + seq_mean[:,:, -3:]
            
            # Collect predictions and ground truth
            predictions.append(output.cpu().numpy())
            y_test_all.append(Y_batch.cpu().numpy())
            
    print('------------------------------------------')


    # Convert results to NumPy
    predictions = np.concatenate(predictions, axis=0)
    y_test_np = np.concatenate(y_test_all, axis=0)
    
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
    print(((k*seq_std_out[:, :, -3:])/(seq_std[:,:,3:6]- seq_std[:,:,0:3])).mean(dim=0)[0,-1])

    
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
    predictions_phy.append(stitch_predictions(predictions))
    stitched_y_test = stitch_predictions(y_test_np)


    print('--- Metrics on Stitched Time Series (each time step unique) ---')
    j = 2
    mse_val = mean_squared_error(stitched_y_test[start:, j], predictions_phy[-1][start:, j])
    rmse_val = root_mean_squared_error(stitched_y_test[start:, j], predictions_phy[-1][start:, j])
    mae_val = mean_absolute_error(stitched_y_test[start:, j], predictions_phy[-1][start:, j])
    r2_val  = r2_score(stitched_y_test[start:, j], predictions_phy[-1][start:, j])
    print(f"Physics-Driven {lam_phy} - Z Axis: MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
    print("-----------------------------------------------------------")


plt.figure(figsize=(14,5))
plt.plot(stitched_y_test[start:, 2], '--', label='Ground Truth (Z)')
plt.plot(predictions_phy[0][start:, 2], label=r'$\lambda_{\mathrm{physics}} = 0$')
plt.plot(predictions_phy[1][start:, 2], label=r'$\lambda_{\mathrm{physics}} = 0.1$')
plt.plot(predictions_phy[-1][start:, 2], label=r'$\lambda_{\mathrm{physics}} = 0.7$')
plt.xlabel('Time Step', fontsize = 18)
plt.ylabel('Force', fontsize = 18)
plt.title('Interaction Forces- Z Axis', fontsize = 22)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.savefig(f'figures/fine_tune_Phy_Lam/{file_path[9:-4]}_Comparison_prediction.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

