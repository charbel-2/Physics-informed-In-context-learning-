import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import json
import wandb
import sys
from torch.utils.data import  DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import savgol_filter
from PITransformer_interaction_model import EnhancedTransformer, PhysicsBasedLoss
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 

# Set the device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Path to your CSV file
csv_file_path_sponge1 = '../Test_Sponge1.csv'  
csv_file_path_sponge2 = '../Test_Sponge2.csv'  
csv_file_path_sponge3 = '../Test_sponge3.csv'   

# Load the CSV into a pandas DataFrame
df = pd.read_csv(csv_file_path_sponge1)
df2 = pd.read_csv(csv_file_path_sponge2)
df3 = pd.read_csv(csv_file_path_sponge3)


# Select relevant columns for modeling: positions, velocities, and forces
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
data2 = df2[relevant_columns].dropna()
data_outputs2 = df2[relevant_outputs].dropna()
data3 = df3[relevant_columns].dropna()
data_outputs3 = df3[relevant_outputs].dropna()


def create_decoder_inputs(data, ctx_length, output_len,q_length):  
    """
    Create the decoder inputs given:
     - data: the whole dataset
     - ctx_length: the length of the encoder input
     - output_len: the prediction horizon
     - q_length: the length of the decoder input
    """
    # Create sequences
    sequences = []
    for i in range(output_len+ctx_length-q_length, len(data) - q_length, ctx_length):
        sequences.append(data[i:i + q_length])
    return np.array(sequences)

# Create decoder inputs from ground truth outputs
def create_encoder_inputs(data, ctx_length): #better to not overlap the sequences, try to use continuous sequences
    """
    Create the decoder inputs given:
     - data: the whole dataset
     - ctx_length: the length of the encoder input
    """
    sequences = []
    for i in range(0,len(data) - ctx_length, ctx_length):
        sequences.append(data[i:i +ctx_length])
    return np.array(sequences)

# Downsample the data
downsampling_factor = 1
datasets = [data,data2, data3]
outputs = [data_outputs,data_outputs2, data_outputs3]

## Train and validation creation
downsampled_data_list = []
downsampled_output_list = []
downsampled_relevant_list = []
train_loaders = []
val_loaders = []
test_loaders = []

try:
    # Dataset creation, splitting training, validation for each dataset
    for i in range(3):
        downsampled_data = datasets[i].iloc[::downsampling_factor, :].reset_index(drop=True)
        downsampled_output = outputs[i].iloc[::downsampling_factor, :].reset_index(drop=True)
        downsampled_relevant = downsampled_data.copy()

        # Compute velocity and acceleration
        delta_t = 0.001 * downsampling_factor
        velocity_x = downsampled_data['velocity_x'].to_numpy()
        velocity_y = downsampled_data['velocity_y'].to_numpy()
        velocity_z = downsampled_data['velocity_z'].to_numpy()
        acceleration_x = np.zeros_like(velocity_x)
        acceleration_y = np.zeros_like(velocity_x)
        acceleration_z = np.zeros_like(velocity_x)

        acceleration_x[1:-1] = (velocity_x[2:] - velocity_x[:-2]) / (2 * delta_t)
        acceleration_x[0] = (velocity_x[1] - velocity_x[0]) / delta_t
        acceleration_x[-1] = (velocity_x[-1] - velocity_x[-2]) / delta_t

        acceleration_y[1:-1] = (velocity_y[2:] - velocity_y[:-2]) / (2 * delta_t)
        acceleration_y[0] = (velocity_y[1] - velocity_y[0]) / delta_t
        acceleration_y[-1] = (velocity_y[-1] - velocity_y[-2]) / delta_t

        acceleration_z[1:-1] = (velocity_z[2:] - velocity_z[:-2]) / (2 * delta_t)
        acceleration_z[0] = (velocity_z[1] - velocity_z[0]) / delta_t
        acceleration_z[-1] = (velocity_z[-1] - velocity_z[-2]) / delta_t
        

        downsampled_data = downsampled_data.iloc[:,3:6]

        position_x = downsampled_data['target_position_x'].to_numpy()
        position_y = downsampled_data['target_position_y'].to_numpy()
        position_z = downsampled_data['target_position_z'].to_numpy()
        target_velocity_x = np.zeros_like(position_x)
        target_velocity_y = np.zeros_like(position_x)
        target_velocity_z = np.zeros_like(position_x)

        target_velocity_x[1:-1] = (position_x[2:] - position_x[:-2]) / (2 * delta_t)
        target_velocity_x[0] = (position_x[1] - position_x[0]) / delta_t
        target_velocity_x[-1] = (position_x[-1] - position_x[-2]) / delta_t

        target_velocity_y[1:-1] = (position_y[2:] - position_y[:-2]) / (2 * delta_t)
        target_velocity_y[0] = (position_y[1] - position_y[0]) / delta_t
        target_velocity_y[-1] = (position_y[-1] - position_y[-2]) / delta_t

        target_velocity_z[1:-1] = (position_z[2:] - position_z[:-2]) / (2 * delta_t)
        target_velocity_z[0] = (position_z[1] - position_z[0]) / delta_t
        target_velocity_z[-1] = (position_z[-1] - position_z[-2]) / delta_t

        downsampled_data['target_velocity_x'] = savgol_filter(target_velocity_x, window_length=500, polyorder=3)
        downsampled_data['target_velocity_y'] = savgol_filter(target_velocity_y, window_length=500, polyorder=3)
        downsampled_data['target_velocity_z'] = savgol_filter(target_velocity_z, window_length=500, polyorder=3)

        downsampled_relevant['target_velocity_x'] = savgol_filter(target_velocity_x, window_length=500, polyorder=3)
        downsampled_relevant['target_velocity_y'] = savgol_filter(target_velocity_y, window_length=500, polyorder=3)
        downsampled_relevant['target_velocity_z'] = savgol_filter(target_velocity_z, window_length=500, polyorder=3)


        # Adding the accelerations and forces to relevant data
        downsampled_relevant['acceleration_x'] = acceleration_x
        downsampled_relevant['acceleration_y'] = acceleration_y
        downsampled_relevant['acceleration_z'] = acceleration_z
        downsampled_relevant['force_x'] = downsampled_output['force_x'].copy()
        downsampled_relevant['force_y'] = downsampled_output['force_y'].copy()
        downsampled_relevant['force_z'] = downsampled_output['force_z'].copy()
        target_acceleration_x = np.zeros_like(position_x)
        target_acceleration_y = np.zeros_like(position_x)
        target_acceleration_z = np.zeros_like(position_x)

        target_acceleration_x[1:-1] = (target_velocity_x[2:] - target_velocity_x[:-2]) / (2 * delta_t)
        target_acceleration_x[0] = (target_velocity_x[1] - target_velocity_x[0]) / delta_t
        target_acceleration_x[-1] = (target_velocity_x[-1] - target_velocity_x[-2]) / delta_t

        target_acceleration_y[1:-1] = (target_velocity_y[2:] - target_velocity_y[:-2]) / (2 * delta_t)
        target_acceleration_y[0] = (target_velocity_y[1] - target_velocity_y[0]) / delta_t
        target_acceleration_y[-1] = (target_velocity_y[-1] - target_velocity_y[-2]) / delta_t

        target_acceleration_z[1:-1] = (target_velocity_z[2:] - target_velocity_z[:-2]) / (2 * delta_t)
        target_acceleration_z[0] = (target_velocity_z[1] - target_velocity_z[0]) / delta_t
        target_acceleration_z[-1] = (target_velocity_z[-1] - target_velocity_z[-2]) / delta_t
        
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

        # Normalize the data (normalize sequences individually)
        mean = (downsampled_data.mean(axis=0)).to_numpy()
        std = (downsampled_data.std(axis=0)).to_numpy()
        joint_data = downsampled_data 

        mean_output = (downsampled_output.mean(axis=0)).to_numpy()
        std_output = (downsampled_output.std(axis=0)).to_numpy()
        joint_outputs = downsampled_output 

        relevant_mean = (downsampled_relevant.mean(axis=0)).to_numpy()
        relevant_std = (downsampled_relevant.std(axis=0)).to_numpy()
        joint_relevant = downsampled_relevant 

        # Save the mean and standard deviation of the data and outputs
        np.save('train_mean.npy', mean)
        np.save('train_std.npy', std)
        np.save('output_mean.npy', mean_output)
        np.save('output_std.npy', std_output)
        np.save('relevant_mean.npy', relevant_mean)
        np.save('relevant_std.npy', relevant_std)

        # Print the statistics of the data
        print(mean)
        print(std)
        print('initial std')
        print(std.mean(axis=0))

        # Create the inputs and outputs for training
        output_seq_len = 240
        seq_length = 352
        len_dec = 352
        X_decoder = create_decoder_inputs(joint_data, seq_length, output_seq_len,len_dec)
        X = create_encoder_inputs(joint_relevant, seq_length)
        Y = create_decoder_inputs(joint_outputs, seq_length, output_seq_len,len_dec)


        X = X[:Y.shape[0], :, :]
        X_decoder = X_decoder[:Y.shape[0], :, :]

        print('----------------')
        print('before split')
        print(X.shape)
        print(X_decoder.shape)
        print(Y.shape)
        print('----------------')

        # Split the data into training (85%) and validation sets(15%)
        X_train, X_val, Y_train, Y_val, X_decoder_train, X_decoder_val = train_test_split(
            X, Y, X_decoder, test_size=0.15, random_state=42, shuffle= False)
        print('----------------')
        print('after splitting')
        print(X_train.shape)
        print(Y_train.shape)
        print(X_decoder_train.shape)
        print('validation:')
        print(X_val.shape)
        print(Y_val.shape)
        print(X_decoder_val.shape)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(X_decoder_train),
            torch.FloatTensor(Y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(X_decoder_val),
            torch.FloatTensor(Y_val)
        )
        batch_size = 16
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
    embed_dim = 92
    num_heads = 4

    initial_params = {
        'inertia': [0.0887, 0.0857, 0.027],   # Same inertia for X, Y, Z
        'damping': [0.866, 0.02866, 0.266],  # Same damping for X, Y, Z
        'stiffness': [0.13, 0.131, 0.52],  # Same stiffness for X, Y, Z
        'damping2': [0.5, 0.5, 1.5]  # Same random offset for X, Y, Z
    }

    lower_bounds = {
        'J': [1e-6, 1e-6, 1e-6],  # Lower bounds for inertia across X, Y, Z
        'b': [1e-6, 1e-6, 1e-6],  # Lower bounds for damping across X, Y, Z
        'k': [1e-6, 1e-6, 1e-6],  # Lower bounds for stiffness across X, Y, Z
        'R': [1e-6, 1e-6, 1e-6]   # Lower bounds for random offset across X, Y, Z
    }

    criterion = PhysicsBasedLoss(lambda_phy=0.45, initial_params=initial_params, lower_bounds=lower_bounds).to(device) 
    criterion2 = nn.MSELoss()

    physics_params = {
        "J": criterion.J,
        "b": criterion.b,
        "k": criterion.k,
        "R": criterion.R
    }

    model = EnhancedTransformer(input_dim=18, n_heads=num_heads, n_layers=6, n_embd=embed_dim, forward_expansion=6, 
                            seq_len= seq_length, device = device) .to(device)

    wandb.init(project="meta-Franka-PhysicsInformed", name=f"Physics_Informed_train", reinit=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = 4000, eta_min=1e-6
        )

    @torch.no_grad()
    # Function for estimation of the validation loss
    def estimate_loss():
        model.eval()
        total_loss = 0.0
        loader_iters = [iter(loader) for loader in val_loaders]  
        num_batches = [len(loader) for loader in val_loaders]

        for selected_idx in range(len(val_loaders)):      
            loss = 0.0
            for _ in range(num_batches[selected_idx]):        
                X_batch, X_decoder_batch, Y_batch = next(loader_iters[selected_idx])
                X_batch = X_batch.to(device)
                X_decoder_batch = X_decoder_batch.to(device)
                Y_batch = Y_batch.to(device)

                # Normalize input sequences
                seq_mean = X_batch.mean(dim=1, keepdims=True)
                seq_std = X_batch.std(dim=1, keepdims=True) + 1e-8
                seq_mean_decoder = X_decoder_batch.mean(dim=1, keepdims=True)
                seq_std_decoder = X_decoder_batch.std(dim=1, keepdims=True) + 1e-8

                seq_std[:,:,0:3] = torch.maximum(X_batch[:,:,0:3].std(dim=1, keepdim= True), X_batch[:,:,3:6].std(dim=1, keepdim= True)) + 1e-6
                seq_std[:,:,3:6] = seq_std[:,:,0:3]

                seq_mean_decoder[:,:,0:3] = seq_mean[:,:,3:6]
                seq_mean_decoder[:,:,3:6] = seq_mean[:,:,9:12]
                seq_std_decoder[:, :, 0:3] = seq_std[:, :, 3:6]
                seq_std_decoder[:, :, 3:6] = seq_std[:, :, 9:12]
                
                X_batch_norm = (X_batch - seq_mean) / seq_std
                X_decoder_batch_norm = (X_decoder_batch - seq_mean_decoder) / seq_std_decoder
                
                # Prepare decoder inputs
                positions = X_batch_norm[:, :, 0:3]
                target_positions = X_batch_norm[:, :, 3:6]
                velocities = X_batch_norm[:, :, 6:9]
                target_velocities = X_batch_norm[:, :, 9:12]
                accelerations = X_batch_norm[:, :, 12:15]
                forces = X_batch_norm[:, :, 15:18]

                
                positions_next = X_decoder_batch_norm[:, :, 0:3].to(device)
                velocities_next = X_decoder_batch_norm[:, :, 3:6].to(device)
                accelerations_next = X_decoder_batch_norm[:, :, 6:9].to(device)

                # Forward pass
                output,J, b, R, k = model(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities,target_velocities, accelerations, forces,
                            positions_next, velocities_next, accelerations_next)
               # MSE loss
                mse_loss = criterion2(output[:,  (len_dec - output_seq_len)+20:, :], Y_batch_norm[:, (len_dec - output_seq_len)+20:, :])
                # mse_loss =  criterion2(output[:, len_dec-output_seq_len+30:, :].squeeze(), Y_batch_norm[:, len_dec-output_seq_len+30:, :].squeeze())
                physics_loss = criterion(output[:, :(len_dec - output_seq_len), :], Y_batch_norm[:, :(len_dec - output_seq_len), :], positions[:, -(len_dec - output_seq_len):, :], target_positions[:, -(len_dec - output_seq_len):, :],
                        velocities[:, -(len_dec - output_seq_len):, :], target_velocities[:, -(len_dec - output_seq_len):, :], accelerations[:, -(len_dec - output_seq_len):, :],
                        J, b, R, k).to(device)                
                
                loss_iter = physics_loss + mse_loss
                    
                loss += loss_iter.item()

            loss /= num_batches[selected_idx]
            loss = np.sqrt(loss)
            total_loss +=loss
        total_loss /= len(val_loaders)
        model.train()
        return total_loss
    
    # Training loop with batching
    epoch = 0
    epoch_num = 4000
    loss_threshold = 0.00015
    loss = 10.0
    stiffness = []
    inertia = []
    damping = []
    randomparam = []
    loss_val = np.nan
    best_val_loss = np.inf

    stiffness_loss = []

    print('-----------------------')
    print("Training started")
    while epoch < epoch_num and loss > loss_threshold:
        model.train()
        total_loss = 0.0
        
        # Create iterators for each train loader
        loader_iters = [iter(loader) for loader in train_loaders]  
        num_batches = sum(len(loader) for loader in train_loaders)  
        for _ in range(num_batches):  
            # Randomly pick a loader that still has data
            while True:
                selected_idx = random.randint(0, len(train_loaders) - 1)
                try:
                    X_batch, X_decoder_batch, Y_batch = next(loader_iters[selected_idx])
                    break  
                except StopIteration:
                    continue

            # Move data to the appropriate device
            X_batch = X_batch.to(device)
            X_decoder_batch = X_decoder_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Normalize input sequences
            seq_mean = X_batch.mean(dim=1, keepdims=True)
            seq_std = X_batch.std(dim=1, keepdims=True) + 1e-8
            seq_mean_decoder = X_decoder_batch.mean(dim=1, keepdims=True)
            seq_std_decoder = X_decoder_batch.std(dim=1, keepdims=True) + 1e-8

            seq_std[:,:,0:3] = torch.maximum(X_batch[:,:,0:3].std(dim=1, keepdim= True), X_batch[:,:,3:6].std(dim=1, keepdim= True)) + 1e-6
            seq_std[:,:,3:6] = seq_std[:,:,0:3]

            seq_mean_decoder[:,:,0:3] = seq_mean[:,:,3:6]
            seq_mean_decoder[:,:,3:6] = seq_mean[:,:,9:12]
            seq_std_decoder[:, :, 0:3] = seq_std[:, :, 3:6]
            seq_std_decoder[:, :, 3:6] = seq_std[:, :, 9:12]
            
            std_positions = seq_std[:,0,0:3].mean(dim = 0)
            std_target_positions = seq_std[:,0,3:6].mean(dim = 0)
            std_forces = seq_std[:,0,15:18].mean(dim = 0)
            
            X_batch_norm = (X_batch - seq_mean) / seq_std
            X_decoder_batch_norm = (X_decoder_batch - seq_mean_decoder) / seq_std_decoder
            Y_batch_norm = (Y_batch - seq_mean[:, :, -3:]) / seq_std[:, :, -3:]

            # Prepare decoder inputs
            positions = X_batch_norm[:, :, 0:3]
            target_positions = X_batch_norm[:, :, 3:6]
            velocities = X_batch_norm[:, :, 6:9]
            target_velocities = X_batch_norm[:, :, 9:12]
            accelerations = X_batch_norm[:, :, 12:15]
            forces = X_batch_norm[:, :, 15:18]

            
            positions_next = X_decoder_batch_norm[:, :, 0:3].to(device)
            velocities_next = X_decoder_batch_norm[:, :, 3:6].to(device)
            accelerations_next = X_decoder_batch_norm[:, :, 6:9].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output,J, b, R, k = model(X_batch_norm, X_decoder_batch_norm, positions, target_positions, velocities,target_velocities, accelerations, forces,
                        positions_next, velocities_next, accelerations_next)
            
            
            loss_overlap = criterion(output[:, :len_dec - output_seq_len, :], Y_batch_norm[:, :len_dec - output_seq_len, :], positions[:, -(len_dec - output_seq_len):, :], target_positions[:, -(len_dec - output_seq_len):, :],
                            velocities[:, -(len_dec - output_seq_len):, :], target_velocities[:, -(len_dec - output_seq_len):, :], accelerations[:, -(len_dec - output_seq_len):, :],
                            J, b, R, k).to(device)

            loss_new = criterion2(output[:, len_dec - output_seq_len:, :], Y_batch_norm[:,len_dec - output_seq_len:, :]).to(device)
                
                
                loss = loss_new + loss_overlap
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            

        # Compute average loss per epoch
        avg_loss = total_loss / num_batches
        scheduler.step()
        epoch += 1


        # Print progress
        if epoch % 50 == 0:
            wandb.log({ "train_loss": avg_loss, "validation_loss": loss_val})
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
        # Save model periodically
            if epoch % 200 == 0:
                loss_val = estimate_loss()
                print(f"\n{epoch=} {loss_val=:.4f}\n")

                if loss_val < best_val_loss:
                    wandb.log({"iter_save": epoch})
                    best_val_loss = loss_val
                    torch.save(model.state_dict(), 'Interaction_metamodel_physics.pth')
                    print("Model Saved!")
                learned_params = {
                            'inertia': J.detach().cpu().tolist(),  # Convert to list for JSON compatibility
                            'damping': b.detach().cpu().tolist(),
                            'stiffness': k.detach().cpu().tolist(),
                            'damping2': R.detach().cpu().tolist()
                            }
                # Save the learned parameters to a JSON file
                with open('learned_params_saved.json', 'w') as f:
                    json.dump(learned_params, f)
except KeyboardInterrupt:
    print("closing gracefully")
    sys.exit() 
