import os
import gc
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import tqdm
from utils import *      # Ensure these helper functions exist: enc_to_int_array, dna_to_int_array, pad_sequences, reset_weights, dna_similarity, bytes_to_dna
from models import *     # Ensure these models exist: RNN, FNN, GRN, LSTM, CustomHuberLoss
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


target_dim = 16

# Define a Dataset that directly reads from memmap arrays via indices.
class MemmapDataset(Dataset):
    def __init__(self, X_memmap, Y_memmap, indices):
        self.X = X_memmap
        self.Y = Y_memmap
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # Read the row from the memmap (avoid making a copy)
        x = self.X[i]
        y = self.Y[i]
        # Convert to torch tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a temporary directory for memmap files
temp_dir = tempfile.gettempdir()

# List CSV files in 'result_csv' (ignoring system files)
dirc = '/media/namng/NamBackup1/_research/PQC-CRISPR/result_csv'
file_paths = [os.path.join(dirc, x) for x in sorted(os.listdir(dirc)) if '.DS_Store']
#print(file_paths[90:])

for i in tqdm.tqdm(range(len(file_paths))):
    if '01.' not in file_paths[i]:
        file_path = file_paths[i]
        print("Processing file:", file_path)

        # Define paths for memmap storage
        x_map_path = os.path.join(temp_dir, f"X_memmap_{i}.dat")
        #y_map_path = os.path.join(temp_dir, f"Y_memmap_{i}.dat")
        
        # Read CSV in chunks (avoid loading entire file at once)
        df_chunks = pd.read_csv(file_path, chunksize=10000)
        df = pd.concat(df_chunks, ignore_index=True)
        
        # Convert DNA sequences to numerical arrays using helper functions
        df['estr_encoded'] = df['estr'].apply(enc_to_int_array)
        df['gstr_encoded'] = df['gstr'].apply(dna_to_int_array)
        
        # Determine maximum length for padding and pad sequences
        max_length = max(df['estr_encoded'].apply(len).max(),
                        df['gstr_encoded'].apply(len).max())
        df['estr_encoded'] = list(pad_sequences(df['estr_encoded'], max_length))
        #df['gstr_encoded'] = list(pad_sequences(df['gstr_encoded'], max_length))
        
        # ---------- Normalization into memmap arrays ----------
        # Create StandardScaler objects; with_mean=False to allow processing large arrays
        x_scaler = StandardScaler(with_mean=False)
        #y_scaler = StandardScaler(with_mean=False)
        
        norm_batch_size = 64
        num_samples = len(df)

        # Define the reduced dimensionality
        # Fit PCA to reduce dimensions
        pca_x = IncrementalPCA(n_components=target_dim)
        #pca_y = IncrementalPCA(n_components=target_dim)

        # ---------- Create train/validation/test splits ----------
        indices = np.arange(num_samples)
        train_idx, temp_idx = train_test_split(indices, test_size=0.5, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.6, random_state=42)

        # ---------- Fit PCA on the TRAIN set only (in chunks) ----------
        accumulated_X = []  # To accumulate small batches
        n_components = pca_x.n_components  # Assumed to be 16

        for start in range(0, len(train_idx), norm_batch_size):
            batch_idx = train_idx[start:start + norm_batch_size]
            # Extract the training samples for this batch
            X_batch = np.stack(df['estr_encoded'].iloc[batch_idx].values)
            accumulated_X.append(X_batch)
            # Concatenate accumulated batches along the first axis
            X_cat = np.concatenate(accumulated_X, axis=0)
            # Only update PCA if we have enough samples
            if X_cat.shape[0] >= n_components:
                pca_x.partial_fit(X_cat)
                accumulated_X = []  # Reset accumulator

        # If any samples remain in accumulator and they are enough, update PCA one more time
        if accumulated_X:
            X_cat = np.concatenate(accumulated_X, axis=0)
            if X_cat.shape[0] >= n_components:
                pca_x.partial_fit(X_cat)

        # ---------- Apply PCA transformation on the entire dataset ----------
        X_memmap = np.memmap(x_map_path, dtype='float32', mode='w+', shape=(num_samples, target_dim))
        for start in range(0, num_samples, norm_batch_size):
            end = min(start + norm_batch_size, num_samples)
            X_batch = np.stack(df['estr_encoded'].iloc[start:end].values)
            X_memmap[start:end] = x_scaler.fit_transform(pca_x.transform(X_batch))
        X_memmap.flush()

        # ---------- Create Datasets that read directly from the memmap arrays ----------
        loader_batch_size = 1024
        train_dataset = MemmapDataset(X_memmap, df['gstr_encoded'].values, train_idx)
        val_dataset   = MemmapDataset(X_memmap, df['gstr_encoded'].values, val_idx)
        test_dataset  = MemmapDataset(X_memmap, df['gstr_encoded'].values, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=0)

        # Remove the DataFrame now that data is in memmaps
        del df
        gc.collect()
        
       # ---------- Training and Evaluation (Transformer Only) ----------
        results = []
        num_epochs = 100
        num_eval = 5  # independent runs per model configuration

        # Get input/output sizes from a sample (from memmap, not converting whole array)
        sample_x = train_dataset[0][0]
        sample_y = train_dataset[0][1]
        input_size = sample_x.shape[0]
        output_size = sample_y.shape[0]
        print("Input size:", input_size, "Output size:", output_size)

        # Define a list of Transformer model configurations
        model_configs = [
            #('Transformer-32-4-32-2', 32, 4, 32, 2),
            #('Transformer-32-4-32-4', 32, 4, 32, 4),
            ('Transformer-32-4-32-8', 32, 4, 32, 8),
            ('Transformer-32-4-32-16', 32, 4, 32, 16),
            ('Transformer-32-4-64-2', 32, 4, 64, 2),
            ('Transformer-32-4-64-4', 32, 4, 64, 4),
            ('Transformer-32-4-64-8', 32, 4, 64, 8),
            ('Transformer-32-4-64-16', 32, 4, 64, 16),
            ('Transformer-32-8-64-2', 32, 8, 64, 2),
            ('Transformer-32-8-64-4', 32, 8, 64, 4),
            ('Transformer-32-8-64-8', 32, 8, 64, 8),
            ('Transformer-32-8-64-16', 32, 8, 64, 16),
        ]

        # Iterate over model configurations
        for model_name, embed_size, num_heads, hidden_size, num_layers in model_configs:
            print(f"Initializing {model_name} with Embed Size: {embed_size}, Heads: {num_heads}, Hidden: {hidden_size}, Layers: {num_layers}")
            model = TransformerModel(
                input_size=input_size,
                embed_size=embed_size,
                num_heads=num_heads,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size
            )


            model.to(device)
            
            optimizers = {
                #"SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
                "RMSprop": optim.RMSprop(model.parameters(), lr=0.005),
                #"AdamW": optim.AdamW(model.parameters(), lr=0.01)
            }
            loss_functions = {
                "MSELoss": nn.MSELoss(),
                #"L1Loss": nn.L1Loss(),
                #"SmoothL1Loss": nn.SmoothL1Loss(),
                #"HuberLoss_Default": CustomHuberLoss(delta=1.0)
            }
            
            for optimizer_name, optimizer in optimizers.items():
                for loss_name, criterion in loss_functions.items():
                    print(f"Evaluating with {optimizer_name} and {loss_name}")
                    for run in range(num_eval):
                        # Reset model weights between runs
                        model.apply(reset_weights)
                        model.to(device)
                        
                        train_loss = val_loss = test_loss = 0
                        train_lev = val_lev = test_lev = 0
                        train_hamming = val_hamming = test_hamming = 0
                        train_identity = val_identity = test_identity = 0
                        
                        best_val_loss = float('inf')
                        best_model_path = None
                        
                        for epoch in range(num_epochs):
                            model.train()
                            total_train_samples = 0
                            for inputs, targets in train_loader:
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                
                                optimizer.zero_grad()
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                loss.backward()
                                optimizer.step()
                                
                                # Compute metrics using helper functions
                                pred_bytes = torch.round(outputs).detach().cpu().numpy().astype(int)
                                true_bytes = targets.detach().cpu().numpy().astype(int)
                                #print([[bytes_to_dna(pred_bytes[i]), bytes_to_dna(true_bytes[i])] for i in range(len(true_bytes))])                  
                                batch_metrics = [dna_similarity(bytes_to_dna(pred_bytes[i]),
                                                                bytes_to_dna(true_bytes[i]))
                                                for i in range(len(true_bytes))]
                                batch_lev = np.mean([m[0] for m in batch_metrics])
                                batch_hamming = np.mean([m[1] for m in batch_metrics if m[1] is not None])
                                batch_identity = np.mean([m[2] for m in batch_metrics])
                                
                                train_loss += loss.item() * inputs.size(0)
                                train_lev += batch_lev * inputs.size(0)
                                train_hamming += batch_hamming * inputs.size(0)
                                train_identity += batch_identity * inputs.size(0)
                                total_train_samples += inputs.size(0)
                            train_loss /= total_train_samples
                            train_lev /= total_train_samples
                            train_hamming /= total_train_samples
                            train_identity /= total_train_samples
                            
                            # Validation Phase
                            model.eval()
                            total_val_samples = 0
                            with torch.no_grad():
                                for inputs, targets in val_loader:
                                    inputs = inputs.to(device)
                                    targets = targets.to(device)
                                    outputs = model(inputs)
                                    loss = criterion(outputs, targets)
                                    
                                    pred_bytes = torch.round(outputs).cpu().numpy().astype(int)
                                    true_bytes = targets.cpu().numpy().astype(int)
                                    batch_metrics = [dna_similarity(bytes_to_dna(pred_bytes[i]),
                                                                    bytes_to_dna(true_bytes[i]))
                                                    for i in range(len(true_bytes))]
                                    batch_lev = np.mean([m[0] for m in batch_metrics])
                                    batch_hamming = np.mean([m[1] for m in batch_metrics if m[1] is not None])
                                    batch_identity = np.mean([m[2] for m in batch_metrics])
                                    
                                    val_loss += loss.item() * inputs.size(0)
                                    val_lev += batch_lev * inputs.size(0)
                                    val_hamming += batch_hamming * inputs.size(0)
                                    val_identity += batch_identity * inputs.size(0)
                                    total_val_samples += inputs.size(0)
                            val_loss /= total_val_samples
                            val_lev /= total_val_samples
                            val_hamming /= total_val_samples
                            val_identity /= total_val_samples
                            #if epoch % 10 == 0:
                            print(f'Run {run+1} - Epoch [{epoch+1}/{num_epochs}] | '
                                f'Train Loss: {train_loss:.4f}, Lev: {train_lev:.2f}, '
                                f'Ham: {train_hamming:.2f}, ID: {train_identity:.2f}% | '
                                f'Val Loss: {val_loss:.4f}, Lev: {val_lev:.2f}, '
                                f'Ham: {val_hamming:.2f}, ID: {val_identity:.2f}%')
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model_path = f"/media/namng/Drive_1/pqc-crispr/results-pca-transformer/best-models/{os.path.basename(file_path)}_{model_name}-{optimizer_name}-{loss_name}-run{run+1}.pth"
                                # Optionally save the model state:
                                torch.save(model.state_dict(), best_model_path)
                            
                            # Test Phase (can also be done after training)
                            model.eval()
                            total_test_samples = 0
                            with torch.no_grad():
                                for inputs, targets in test_loader:
                                    inputs = inputs.to(device)
                                    targets = targets.to(device)
                                    outputs = model(inputs)
                                    loss = criterion(outputs, targets)
                                    
                                    pred_bytes = torch.round(outputs).cpu().numpy().astype(int)
                                    true_bytes = targets.cpu().numpy().astype(int)
                                    #print([[bytes_to_dna(pred_bytes[i]), bytes_to_dna(true_bytes[i])] for i in range(len(true_bytes))])                  

                                    batch_metrics = [dna_similarity(bytes_to_dna(pred_bytes[i]),
                                                                    bytes_to_dna(true_bytes[i]))
                                                    for i in range(len(true_bytes))]
                                    batch_lev = np.mean([m[0] for m in batch_metrics])
                                    batch_hamming = np.mean([m[1] for m in batch_metrics if m[1] is not None])
                                    batch_identity = np.mean([m[2] for m in batch_metrics])
                                    
                                    test_loss += loss.item() * inputs.size(0)
                                    test_lev += batch_lev * inputs.size(0)
                                    test_hamming += batch_hamming * inputs.size(0)
                                    test_identity += batch_identity * inputs.size(0)
                                    total_test_samples += inputs.size(0)
                            test_loss /= total_test_samples
                            test_lev /= total_test_samples
                            test_hamming /= total_test_samples
                            test_identity /= total_test_samples
                            
                            torch.cuda.empty_cache()
                        
                        results.append({
                            'Run': run + 1,
                            'Model': model_name,
                            'Optimizer': optimizer_name,
                            'Loss Function': loss_name,
                            'Train Loss': train_loss,
                            'Train Lev': train_lev,
                            'Train Ham': train_hamming,
                            'Train ID': train_identity,
                            'Val Loss': val_loss,
                            'Val Lev': val_lev,
                            'Val Ham': val_hamming,
                            'Val ID': val_identity,
                            'Test Loss': test_loss,
                            'Test Lev': test_lev,
                            'Test Ham': test_hamming,
                            'Test ID': test_identity,
                            'Best Model Path': best_model_path
                        })
        
        # Save results to CSV
        df_results = pd.DataFrame(results)
        results_file = f'/media/namng/Drive_1/pqc-crispr/results-pca-transformer/{os.path.basename(file_path)}-result-DNN.csv'
        df_results.to_csv(results_file, index=False)
        print(f"Evaluation complete. Results saved to '{results_file}'")
        
        # Clean up large objects and call garbage collection
        del X_memmap, df_results
        gc.collect()
        torch.cuda.empty_cache()
