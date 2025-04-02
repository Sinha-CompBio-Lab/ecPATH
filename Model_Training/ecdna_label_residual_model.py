import numpy as np
import torch
import torch.nn as nn
from utils import compute_coef_slope
from sklearn.metrics import roc_auc_score

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
        # self.fc = nn.Linear(hidden_dim, 1)
        self.fc =  nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim,1)
        )
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

class EcDNATileClassifier_AucSelect(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, feature_selection_topk=20, dropout=0.2):
        super(EcDNATileClassifier_AucSelect, self).__init__()
        
        # Feature extractors for each tile
        self.res1 = ResidualBlock(input_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, hidden_dim)
        
        
        # Final prediction layer
        # self.fc = nn.Linear(hidden_dim, 1)
        # self.fc =  nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim,1)
        # )

        self.fc = nn.Linear(input_dim, 1)
        # self.fc =  nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim,1)
        # )
        self.sigmoid = nn.Sigmoid()

        # Feature selection parameters
        self.feature_selection_topk = feature_selection_topk
        self.update_counter = 0
        # self.feature_scores = torch.ones(hidden_dim)
        self.feature_scores = torch.ones(input_dim)
        
        # Initialize feature mask (all features enabled initially)
        _, top_indices = torch.topk(self.feature_scores, k=self.feature_selection_topk)
        # self.feature_mask = torch.zeros(hidden_dim, 1)
        self.feature_mask = torch.zeros(input_dim, 1)
        self.feature_mask[top_indices] = 1.0

    def set_feature_mask(self, device, train_set):
        batch_features = []
        batch_labels = []
        n_slides_train = len(train_set)

        for idx in range(n_slides_train):
            
            x, y = train_set[idx]
            x = x.to(device)
            y = y.view(1,1).float().to(device)  # change y to be 1 x 1 vector

            with torch.no_grad():
                # Get features before feature selection (for AUC calculation)
                # Forward pass through feature extraction blocks
                # features = self.res1(x)
                # features = self.res2(features)
                # features = self.res3(features)
                # aggregated_features = torch.mean(features, dim=0, keepdim=True)
                aggregated_features = torch.mean(x, dim=0, keepdim=True)
                
                # Store for AUC calculation
                batch_features.append(aggregated_features.detach())
                # batch_features.append(x.detach())
                batch_labels.append(y.detach())
                
        batch_features_tensor = torch.cat(batch_features, dim=0)
        batch_labels_tensor = torch.cat(batch_labels, dim=0)
        self.update_feature_mask(batch_features_tensor, batch_labels_tensor)


    def update_feature_mask(self, features_batch, labels_batch):
        """Update feature mask based on AUC scores"""
        # self.update_counter += 1
        
        # # Run this periodically, not every batch
        # if self.update_counter % 50 != 0:  # Update every 50 batches
        #     return
        
        # Compute AUC for each feature
        features_np = features_batch.detach().cpu().numpy()
        labels_np = labels_batch.detach().cpu().numpy()
        
        auc_scores = []
        for feature_idx in range(features_np.shape[1]):
            feature_values = features_np[:, feature_idx]
            try:
                auc = roc_auc_score(labels_np, feature_values)
                # Make sure AUC > 0.5, otherwise use (1-AUC)
                if auc < 0.5:
                    auc = 1 - auc
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)  # Default for features with no predictive power
        
        # # Print Auc score info
        # top_indices = np.argsort(auc_scores)[::-1][:20]
        # for idx in top_indices:
        #     print(f"Feature {idx}: AUC = {auc_scores[idx]:.4f}")
        # print(f"{np.mean(auc_scores)}\n")

        # Convert to tensor and store
        self.feature_scores = torch.tensor(auc_scores, device=features_batch.device)
        
        # Select top-k features
        # _, top_indices = torch.topk(self.feature_scores, k=min(self.feature_selection_topk, len(auc_scores)))
        threshold = 0.6
        top_indices = torch.where(self.feature_scores >= threshold)[0]

        print(f"Features with AUC score >= {threshold}:")
        print(len(top_indices))
        # for idx in top_indices:
        #     # Convert tensor index to integer for printing
        #     feature_idx = idx.item()
        #     score = self.feature_scores[feature_idx].item()
        #     print(f"Feature {feature_idx}: AUC = {score:.4f}")

        # Create new mask (all zeros)
        new_mask = torch.zeros(len(auc_scores), 1, device=features_batch.device)
        
        # Set 1's for the top features
        new_mask[top_indices] = 1.0
        
        # Update the feature mask (gradual update to stabilize training)
        alpha = 0.9  # Exponential moving average factor
        self.feature_mask = alpha * self.feature_mask + (1 - alpha) * new_mask
        self.feature_mask = new_mask
        
    # Update the feature mask
    def select_features_by_auc(self, features):
        """Apply feature mask to get selected features"""
        # Make sure feature_mask is on the same device as features
        if self.feature_mask.device != features.device:
            self.feature_mask = self.feature_mask.to(features.device)
            
        return features * self.feature_mask.t()
    
    def forward(self, x):
        batch_size = 1  # Single slide with multiple tiles for now
        num_tiles = x.shape[0] 
        
        # # # Process each tile through residual blocks
        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.res3(x)  # Shape: [num_tiles, hidden_dim]

         # Average pooling across tiles
        aggregated_features = torch.mean(x, dim=0,keepdim=True) # Shape: [1, hidden_dim]

         # Apply feature selection
        selected_features = self.select_features_by_auc(aggregated_features)

        #Titan
        # selected_features = self.select_features_by_auc(x)
        
        # Final prediction using the linear layer
        output = self.fc(selected_features)
        return self.sigmoid(output)

def training_epoch_with_auc_select(model, optimizer, train_set, batch_size, l1_lambda=0.0, l2_lambda=0.0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.train()
    
    loss_fn = nn.BCELoss() 
    
    n_slides_train = len(train_set)
    
    # Shuffle training set
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)
    
    loss_list = []
    labels = []
    binary_preds = []
    probs = []

    for i_batch in range(0, n_slides_train, batch_size):
        n_slides_batch = min(batch_size, n_slides_train - i_batch)
        
        # For each batch
        loss = 0
        batch_features = []
        batch_labels = []
        
        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]
            
            x, y = train_set[idx]
            x = x.to(device)
            y = y.view(1,1).float().to(device)  # change y to be 1 x 1 vector

            with torch.no_grad():
                # Get features before feature selection (for AUC calculation)
                # Forward pass through feature extraction blocks
                # features = model.res1(x)
                # features = model.res2(features)
                # features = model.res3(features)
                # aggregated_features = torch.mean(features, dim=0, keepdim=True)
                # aggregated_features = torch.mean(x, dim=0, keepdim=True)
                
                # Store for AUC calculation
                # batch_features.append(aggregated_features.detach())
                batch_features.append(x.detach())
                batch_labels.append(y.detach())
            
            # Complete the forward pass with feature selection
            pred = model(x)
            
            # Calculate base loss
            base_loss = loss_fn(pred, y)


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
            binary_preds.append((pred.detach().cpu().numpy() >= 0.5).astype(float))

        # Average loss for the batch
        loss /= n_slides_batch
        loss_list += [loss.detach().cpu().numpy()]
        
        # Reset gradients to zero
        optimizer.zero_grad()
        
        # Compute gradients
        loss.backward()
        
        # Update parameters using gradients
        optimizer.step()
        
        # # Update feature mask after processing the batch
        if batch_features:
            batch_features_tensor = torch.cat(batch_features, dim=0)
            batch_labels_tensor = torch.cat(batch_labels, dim=0)
            model.update_feature_mask(batch_features_tensor, batch_labels_tensor)
    
    labels = np.array(labels)
    binary_preds = np.array(binary_preds)
    probs = np.array(probs)

    return loss_list, labels, binary_preds, probs


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


