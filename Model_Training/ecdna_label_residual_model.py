import numpy as np
import torch
import torch.nn as nn
from utils import compute_coef_slope


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
            
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class EcDNATileClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(EcDNATileClassifier, self).__init__()
        
        # Feature extractors for each tile
        self.res1 = ResidualBlock(input_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, hidden_dim)
        
        # Attention mechanism for tile aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = 1  # Single slide with multiple tiles for now
        num_tiles = x.shape[0] 
        
        # Process each tile through residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)  # Shape: [num_tiles, hidden_dim]
        
        # Apply attention to weight tiles
        attention_weights = self.attention(x)  # Shape: [num_tiles, 1]
        attention_weights = torch.softmax(attention_weights, dim=0)  # Normalize weights
        
        # Apply attention to get weighted average of tile features
        weighted_features = x * attention_weights  # Shape: [num_tiles, hidden_dim]
        aggregated_features = torch.sum(weighted_features, dim=0, keepdim=True)  # Shape: [1, hidden_dim]
        
        # Final prediction
        x = self.fc(aggregated_features)
        return self.sigmoid(x)
    

def training_epoch(model, optimizer, train_set, batch_size, l1_lambda=0.0, l2_lambda=0.0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.train()
    
    loss_fn = nn.BCELoss() 
    
    n_slides_train = len(train_set)
    
    # Shuffle training set
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)
    
    loss_list = []
    labels = []
    preds = []
    probs = []
    
    for i_batch in range(0, n_slides_train, batch_size):
        n_slides_batch = min(batch_size, n_slides_train - i_batch)
        
        # For each batch
        loss = 0
        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]
            
            x, y = train_set[idx]
  
            pred = model(x.to(device))
            y = y.view(1,1) # change y to be 1 x 1 vector

            # Calculate base loss
            base_loss = loss_fn(pred, y.float().to(device))

            # Convert to binary predictions
            binary_pred = (pred.detach().cpu().numpy() >= 0.5).astype(float)

            # Add regularization
            l1_reg = 0
            l2_reg = 0
            
            # Calculate L1 regularization (absolute value of weights)
            if l1_lambda > 0:
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)  # L1 norm
                
            # Calculate L2 regularization (squared value of weights)
            if l2_lambda > 0:
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2  # L2 norm squared
            
            # Combine losses: base_loss + L1 + L2
            total_loss = base_loss + (l1_lambda * l1_reg) + (l2_lambda * l2_reg)
            loss += total_loss
            
            labels.append(y.detach().cpu().numpy())
            probs.append(pred.detach().cpu().numpy())
            preds.append(binary_pred)

        
        # Average loss for the batch
        loss /= n_slides_batch
        loss_list += [loss.detach().cpu().numpy()]
        
        # Reset gradients to zero
        optimizer.zero_grad()
        
        # Compute gradients
        loss.backward()
        
        # Update parameters using gradients
        optimizer.step()
    
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    return loss_list, labels, preds, probs


