import numpy as np
import torch
import torch.nn as nn
from utils import compute_coef_slope
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

class EcDNATileClassifier_AucSelect(nn.Module):
    def __init__(self, input_dim, feature_type, hidden_dim=256, use_attention=False, 
                 use_feature_selection=True, feature_selection={'percent': 0.1}, dropout=0.2):
        super(EcDNATileClassifier_AucSelect, self).__init__()
        
        self.feature_type = feature_type
        self.use_attention = use_attention
        self.use_feature_selection = use_feature_selection
        # Feature selection parameters
        self.feature_selection = feature_selection

        # Tested and no longer used. 
        self.res1 = ResidualBlock(input_dim, hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Linear(input_dim, 1)
        # self.fc =  nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim,1)
        # )

        self.sigmoid = nn.Sigmoid()
        
        # Initialize feature mask (all features enabled initially)        
        self.feature_mask = torch.ones(input_dim, 1)

    def set_feature_mask(self, device, train_set):
        self.feature_mask = self.feature_mask.to(device)
        batch_features = []
        batch_labels = []
        n_slides_train = len(train_set)

        for idx in range(n_slides_train):
            
            x, y = train_set[idx]
            x = x.to(device)
            y = y.view(1,1).float().to(device)  # change y to be 1 x 1 vector

            with torch.no_grad():
                # Get features before feature selection (for AUC calculation)
                aggregated_features = torch.mean(x, dim=0, keepdim=True)
                # Store for AUC calculation
                batch_features.append(aggregated_features.detach())
                # Titan
                # batch_features.append(x.detach())
                batch_labels.append(y.detach())
                
        batch_features_tensor = torch.cat(batch_features, dim=0)
        batch_labels_tensor = torch.cat(batch_labels, dim=0)
        self.update_feature_mask(batch_features_tensor, batch_labels_tensor,mvAvg=False)

    def feature_mask_select(self, auc_scores, selector):
        if 'threshold' in selector:
            # Select features with AUC scores >= threshold
            threshold = selector['threshold']
            selected_indices = torch.where(auc_scores >= threshold)[0]
            
        elif 'topK' in selector:
            # Select top K features by AUC score
            k = min(selector['topK'], len(auc_scores))
            _, selected_indices = torch.topk(auc_scores, k=k)
            
        elif 'percent' in selector:
            # Select top N% of features by AUC score
            percent = selector['percent']
            k = max(1, int(len(auc_scores) * percent))  # Ensure at least 1 feature
            _, selected_indices = torch.topk(auc_scores, k=k)
            
        else:
            # Default: return all features
            selected_indices = torch.arange(len(auc_scores), device=auc_scores.device)
        
        return selected_indices

    def update_feature_mask(self, features_batch, labels_batch, mvAvg=True):
        """Update feature mask based MvAvg of AUC scores unless MvAvg set to false"""
        
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

        # Convert to tensor and store
        feature_scores = torch.tensor(auc_scores, device=features_batch.device)

        top_indices = self.feature_mask_select(feature_scores, self.feature_selection)

        print(f"\n{len(top_indices)}")
        meanscore = []
        for idx in top_indices:
            # Convert tensor index to integer for printing
            feature_idx = idx.item()
            score = feature_scores[feature_idx].item()
            meanscore.append(score)
            # print(f"Feature {feature_idx}: AUC = {score:.4f}")
        meanscore = sum(meanscore) / len(meanscore)
        print(f"Mean AUC score is: {meanscore}\n")

        # Create new mask (all zeros)
        new_mask = torch.zeros(len(auc_scores), 1, device=features_batch.device)
        
        # Set 1's for the top features
        new_mask[top_indices] = 1.0
        
        if mvAvg:
            # Update the feature mask (gradual update to stabilize training)
            alpha = 0.5 #.9  # Exponential moving average factor
            self.feature_mask = alpha * self.feature_mask + (1 - alpha) * new_mask
        else:
            self.feature_mask = new_mask
        
    # Update the feature mask
    def select_features_by_auc(self, features):
        """Apply feature mask to get selected features"""
        # Make sure feature_mask is on the same device as features
        if self.feature_mask.device != features.device:
            self.feature_mask = self.feature_mask.to(features.device)
            
        return features * self.feature_mask.t()
    
    def forward(self, x):
        if self.feature_type == "titan":
            aggregated_features = x
        else:
            if self.use_attention:
                attention_weights = self.attention(x) # Shape: [num_tiles, 1]
                attention_weights = torch.softmax(attention_weights, dim=0) # Normalize weights
                aggregated_features = torch.sum(x * attention_weights, dim=0, keepdim=True)  # Shape: [num_tiles, hidden_dim]
            else:
                aggregated_features = torch.mean(x, dim=0, keepdim=True) 

        if self.use_feature_selection:
            selected_features = self.select_features_by_auc(aggregated_features) 
        else:
            selected_features = aggregated_features

        output = self.fc(selected_features)
        return self.sigmoid(output)
    

def training_epoch_with_auc_select(model, optimizer, train_set, model_feature_select,feature_type, 
                                   batch_size, l1_lambda=0.0, l2_lambda=0.0):
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
            
            # Store for AUC calculation
            if model_feature_select == 'mvavg':
                with torch.no_grad():
                    if feature_type == 'titan':
                        batch_features.append(x.detach()) #Titan
                    else:
                        # Get features before feature selection (for AUC calculation)
                        aggregated_features = torch.mean(x, dim=0, keepdim=True)
                        batch_features.append(aggregated_features.detach()) #Uni

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
        

        # Update feature mask after processing the batch
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

class RandomForestModel_Class():
    def __init__(self, input_dim,n_estimators=100, random_state=42,feature_selection_topk=10):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_selection_topk= feature_selection_topk
        self.feature_mask = torch.zeros(input_dim, 1)

    def update_feature_mask(self,features,labels):
        auc_scores = []
        
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

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
       
 
        feature_scores = torch.tensor(auc_scores, device=features.device)
        # Select top K features
        # top_indices = np.argsort(auc_scores)[::-1][:10]
        # for idx in top_indices:
        #     print(f"Feature {idx}: AUC = {auc_scores[idx]:.4f}")
        # print(f"{np.mean(auc_scores)}\n")
        # _, top_indices = torch.topk(feature_scores, k=min(self.feature_selection_topk, len(auc_scores)))

        # Select features based on threshold
        threshold = 0.6
        top_indices = torch.where(feature_scores >= threshold)[0]
        # print(f"Features with AUC score >= {threshold}:")
        # print(len(top_indices))

        # Create feature mask
        new_mask = torch.zeros(len(auc_scores), 1, device=features.device)
        new_mask[top_indices] = 1.0

        # Update the feature mask (gradual update to stabilize training)
        # alpha = 0.9  # Exponential moving average factor
        # self.feature_mask = alpha * self.feature_mask + (1 - alpha) * new_mask
        self.feature_mask = new_mask
    
    def select_features(self, features):
        if self.feature_mask.device != features.device:
            self.feature_mask = self.feature_mask.to(features.device)
        return features * self.feature_mask.t()
                
    def train_rf_model(self,train_set,selc_fet=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Extract all features and targets from the Dataset
        # We need to convert from PyTorch Dataset format to numpy arrays
        all_features = []
        all_targets = []
        for i in range(len(train_set)):
            features, target = train_set[i]
            features = features.to(device)
            target = target.to(device)  # change y to be 1 x 1 vector
            # Uni 
            aggregated_features = torch.mean(features, dim=0, keepdim=True)
            all_features.append(aggregated_features.detach())
            # Convert from PyTorch tensors to numpy
            # all_features.append(features.detach())
            all_targets.append(target.detach())

        X_train = torch.cat(all_features, dim=0)
        y_train = torch.cat(all_targets, dim=0)

        if selc_fet:
           self.update_feature_mask(X_train,y_train)
           X_train = self.select_features(X_train)

        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()
        # Fit the model
        self.model.fit(X_train_np, y_train_np)
        
        return self.model
    

    def evaluate_rf_model(self,test_set,selc_fet=False):
        # Extract all features and targets from the test Dataset
        all_features = []
        all_targets = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(len(test_set)):
            features, target = test_set[i]
            features = features.to(device)
            target = target.to(device)  # change y to be 1 x 1 vector
            aggregated_features = torch.mean(features, dim=0, keepdim=True)
            all_features.append(aggregated_features.detach())
            # Convert from PyTorch tensors to numpy
            # all_features.append(features.detach())
            all_targets.append(target.detach())

        # Stack all features and targets into single arrays
        X_test = torch.cat(all_features, dim=0)
        y_test = torch.cat(all_targets, dim=0)
        if selc_fet:
            X_test = self.select_features(X_test)

        X_test_np = X_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        # If you want probability scores instead of class predictions
        y_pred_proba = self.model.predict_proba(X_test_np)

        # Check if binary classification for ROC AUC
        if len(np.unique(y_test_np)) == 2:
            # Use probability of positive class
            roc_auc = roc_auc_score(y_test_np, y_pred_proba[:, 1])
            print(f"N estimator:{self.n_estimators}")
            print(f"ROC AUC Score: {roc_auc:.4f}")
            
        else:
            print(f"Warning: Only one class present in evaluation set ({np.unique(y_test_np)[0]}). Returning 0.5 AUC.")
            print(f"ROC AUC score: 0.5")
            roc_auc = 0.5
        return roc_auc

     