import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAccuracy, MultilabelAveragePrecision

# Đặt biến môi trường để chỉ định GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Sử dụng GPU 0 và 5

# Đọc dữ liệu
data = pd.read_csv("text_only_classification_dataset.csv")

# Xử lý nhãn
data['label'] = data['label'].apply(lambda x: x.split(','))  # Tách chuỗi thành danh sách
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(data['label'])  # Chuyển thành dạng one-hot encoding
label_classes = mlb.classes_

# Custom Loss Functions
class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss for multilabel classification to address class imbalance
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(MultilabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        # Calculate pt (probability of being correct)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            focal_weight = focal_weight * torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class LabelSmoothingBCELoss(nn.Module):
    """
    BCE Loss with label smoothing for more robust training
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Smooth the labels
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)

# Custom Dataset Implementations
class MultilabelDataset(Dataset):
    """Basic multilabel dataset"""
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)  # Multilabel yêu cầu FloatTensor
        }

class OversampledMultilabelDataset(Dataset):
    """Dataset with oversampling for rare classes"""
    def __init__(self, texts, labels, tokenizer, max_len=512, oversample_factor=2.0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Calculate class frequencies
        class_counts = np.sum(labels, axis=0)
        max_count = np.max(class_counts)
        
        # Find samples with rare classes for oversampling
        additional_indices = []
        for i in range(labels.shape[1]):
            if class_counts[i] < max_count / oversample_factor:
                # Find samples with this rare class
                rare_indices = np.where(labels[:, i] == 1)[0]
                # Calculate how many times to repeat these samples
                repeat_factor = int(max_count / class_counts[i] / oversample_factor)
                for _ in range(repeat_factor - 1):  # -1 because we already have one copy
                    additional_indices.extend(rare_indices)
        
        # Add oversampled indices
        self.indices = list(range(len(texts))) + additional_indices
        print(f"Original dataset size: {len(texts)}, After oversampling: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        text = str(self.texts[original_idx])
        label = self.labels[original_idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# Custom Trainer for Imbalanced Data
class ImbalancedMultilabelTrainer(Trainer):
    """
    Custom Trainer for imbalanced multilabel classification
    """
    def __init__(self, pos_weights=None, gamma=2.0, alpha=0.25, use_focal_loss=False, 
                 label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weights = pos_weights
        self.gamma = gamma
        self.alpha = alpha
        self.use_focal_loss = use_focal_loss
        self.label_smoothing = label_smoothing
        
        # Calculate label frequencies if pos_weights not provided
        if self.pos_weights is None and self.train_dataset is not None:
            self.pos_weights = self._calculate_pos_weights()
    
    def _calculate_pos_weights(self):
        """Calculate positive weights based on label frequencies in training data"""
        all_labels = []
        for i in range(len(self.train_dataset)):
            all_labels.append(self.train_dataset[i]['labels'].numpy())
        
        all_labels = np.array(all_labels)
        pos_counts = np.sum(all_labels, axis=0)
        neg_counts = len(all_labels) - pos_counts
        
        # Avoid division by zero
        pos_counts = np.maximum(pos_counts, 1)
        
        # Calculate weights: higher weight for less frequent classes
        weights = neg_counts / pos_counts
        
        # Normalize weights to avoid extremely large values
        weights = np.clip(weights, 0.1, 10.0)
        
        return torch.FloatTensor(weights).to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        """
        Custom loss computation with weighted BCE or focal loss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Choose loss function based on configuration
        if self.use_focal_loss:
            loss_fct = MultilabelFocalLoss(gamma=self.gamma, alpha=self.alpha)
            loss = loss_fct(logits, labels)
        elif self.label_smoothing > 0:
            loss_fct = LabelSmoothingBCELoss(smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)
        else:
            # Weighted BCE loss
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
            loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics_with_threshold_tuning(self, eval_pred):
        """
        Compute metrics with threshold tuning for each class
        """
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Find optimal thresholds for each class
        thresholds = self._find_optimal_thresholds(logits, labels)
        
        # Apply class-specific thresholds
        preds = np.zeros_like(logits)
        for i in range(logits.shape[1]):
            preds[:, i] = (logits[:, i] > thresholds[i]).astype(int)
        
        # Create a new eval_pred with thresholded predictions
        from collections import namedtuple
        EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])
        new_eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
        
        # Use the compute_metrics function with thresholded predictions
        metrics = self.compute_metrics(new_eval_pred)
        
        # Add threshold information
        metrics['thresholds'] = thresholds.tolist()
        
        return metrics
    
    def _find_optimal_thresholds(self, logits, labels, steps=100):
        """
        Find optimal threshold for each class using F1 score
        """
        thresholds = np.zeros(logits.shape[1])
        
        for i in range(logits.shape[1]):
            best_f1 = 0
            best_threshold = 0.5  # Default threshold
            
            # Try different thresholds
            for threshold in np.linspace(0.1, 0.9, steps):
                preds = (logits[:, i] > threshold).astype(int)
                
                # Calculate F1 score components
                tp = np.sum((preds == 1) & (labels[:, i] == 1))
                fp = np.sum((preds == 1) & (labels[:, i] == 0))
                fn = np.sum((preds == 0) & (labels[:, i] == 1))
                
                # Calculate F1 score
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Update best threshold
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            thresholds[i] = best_threshold
        
        return thresholds
    
    def create_balanced_sampler(self, dataset):
        """
        Create a weighted sampler to balance class frequencies
        """
        all_labels = []
        for i in range(len(dataset)):
            all_labels.append(dataset[i]['labels'].numpy())
        
        all_labels = np.array(all_labels)
        
        # Calculate sample weights based on label rarity
        weights = np.zeros(len(dataset))
        
        for i in range(len(dataset)):
            # Get the rarest label in this sample
            sample_labels = all_labels[i]
            if np.sum(sample_labels) > 0:
                # Use the rarest positive label's weight
                pos_indices = np.where(sample_labels == 1)[0]
                rarest_label_idx = pos_indices[np.argmax(self.pos_weights.cpu().numpy()[pos_indices])]
                weights[i] = self.pos_weights.cpu().numpy()[rarest_label_idx]
            else:
                # For samples with no positive labels
                weights[i] = 1.0
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
        return sampler

    def get_train_dataloader(self):
        """
        Override to use balanced sampler
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Create balanced sampler
        sampler = self.create_balanced_sampler(train_dataset)
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# Enhanced metrics computation
def compute_imbalanced_metrics(eval_pred):
    """
    Compute comprehensive metrics for imbalanced multilabel classification
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Apply default threshold
    preds = (logits > 0).astype(int)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert numpy arrays to tensors with proper dtypes
    logits_tensor = torch.tensor(logits, dtype=torch.float32).to(device)
    preds_tensor = torch.tensor(preds, dtype=torch.int32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.int32).to(device)
    
    # Initialize metrics
    acc = MultilabelAccuracy(num_labels=labels.shape[1], average='macro').to(device)
    precision_metric = MultilabelPrecision(num_labels=labels.shape[1], average='macro').to(device)
    recall_metric = MultilabelRecall(num_labels=labels.shape[1], average='macro').to(device)
    f1_metric = MultilabelF1Score(num_labels=labels.shape[1], average='macro').to(device)
    mAP = MultilabelAveragePrecision(num_labels=labels.shape[1]).to(device)
    
    # Per-class metrics
    per_class_precision = MultilabelPrecision(num_labels=labels.shape[1], average=None).to(device)
    per_class_recall = MultilabelRecall(num_labels=labels.shape[1], average=None).to(device)
    per_class_f1 = MultilabelF1Score(num_labels=labels.shape[1], average=None).to(device)
    
    # Calculate overall metrics
    acc_value = acc(preds_tensor, labels_tensor).item()
    precision_value = precision_metric(preds_tensor, labels_tensor).item()
    recall_value = recall_metric(preds_tensor, labels_tensor).item()
    f1_value = f1_metric(preds_tensor, labels_tensor).item()
    map_value = mAP(logits_tensor, labels_tensor).item()
    
    # Calculate per-class metrics
    class_precision = per_class_precision(preds_tensor, labels_tensor).cpu().numpy()
    class_recall = per_class_recall(preds_tensor, labels_tensor).cpu().numpy()
    class_f1 = per_class_f1(preds_tensor, labels_tensor).cpu().numpy()
    
    # Calculate class support (number of instances per class)
    class_support = np.sum(labels, axis=0)
    
    # Prepare metrics dictionary
    metrics = {
        'accuracy': acc_value,
        'precision': precision_value,
        'recall': recall_value,
        'f1': f1_value,
        'mAP': map_value,
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(label_classes):
        metrics[f"precision_{class_name}"] = float(class_precision[i])
        metrics[f"recall_{class_name}"] = float(class_recall[i])
        metrics[f"f1_{class_name}"] = float(class_f1[i])
        metrics[f"support_{class_name}"] = int(class_support[i])
    
    return metrics

# Main execution code
def main():
    # Split data into train and validation sets
    texts = data['rewrite'].values
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_classes),
        problem_type='multi_label_classification'
    )
    
    # Calculate positive weights based on label frequencies
    label_counts = np.sum(train_labels, axis=0)
    total_samples = len(train_labels)
    neg_counts = total_samples - label_counts
    pos_weights = neg_counts / np.maximum(label_counts, 1)  # Avoid division by zero
    pos_weights = np.clip(pos_weights, 0.1, 10.0)  # Clip to avoid extreme values
    pos_weights = torch.FloatTensor(pos_weights)
    
    # Print class distribution
    print("Label distribution in training set:")
    for i, class_name in enumerate(label_classes):
        print(f"{class_name}: {label_counts[i]} samples ({label_counts[i]/total_samples*100:.2f}%)")
    
    # Create datasets
    # Option 1: Regular datasets
    train_dataset = MultilabelDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultilabelDataset(val_texts, val_labels, tokenizer)
    
    # Option 2: Oversampled dataset for training
    train_dataset_oversampled = OversampledMultilabelDataset(
        train_texts, train_labels, tokenizer, oversample_factor=3.0
    )
    
    # Choose which dataset to use
    use_oversampling = True  # Set to True to use oversampled dataset
    final_train_dataset = train_dataset_oversampled if use_oversampling else train_dataset
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs= 10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        weight_decay=0.005,
        learning_rate=5e-5,
        logging_dir='./logs',
        logging_steps= 100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        greater_is_better=False,  # Loss thấp hơn là tốt hơn
        # local_rank=-1,  # Cho phép distributed training
        fp16=True,      # Sử dụng mixed precision để tăng tốc
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        metric_for_best_model='eval_loss',
        save_total_limit=2,  # Keep only the 2 best checkpoints
    )
    
    # Initialize the custom trainer
    trainer = ImbalancedMultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=final_train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_imbalanced_metrics,
        pos_weights=pos_weights,
        gamma=2.0,  # Focal loss gamma parameter
        alpha=0.25,  # Focal loss alpha parameter
        use_focal_loss=True,  # Use focal loss instead of weighted BCE
        label_smoothing=0.1,  # Apply label smoothing
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate with threshold tuning
    print("\nEvaluating with optimal thresholds for each class...")
    tuned_results = trainer.compute_metrics_with_threshold_tuning(
        trainer.predict(val_dataset)
    )
    
    # Print evaluation results
    print("\nEvaluation results with threshold tuning:")
    print(f"Overall Accuracy: {tuned_results['accuracy']:.4f}")
    print(f"Overall Precision: {tuned_results['precision']:.4f}")
    print(f"Overall Recall: {tuned_results['recall']:.4f}")
    print(f"Overall F1: {tuned_results['f1']:.4f}")
    print(f"Mean Average Precision: {tuned_results['mAP']:.4f}")
    
    print("\nOptimal thresholds for each class:")
    for i, class_name in enumerate(label_classes):
        threshold = tuned_results['thresholds'][i]
        print(f"{class_name}: {threshold:.4f}")
    
    print("\nPer-class metrics:")
    for class_name in label_classes:
        precision = tuned_results[f"precision_{class_name}"]
        recall = tuned_results[f"recall_{class_name}"]
        f1 = tuned_results[f"f1_{class_name}"]
        support = tuned_results[f"support_{class_name}"]
        print(f"{class_name} (support: {support}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
    
    # Save the best model
    trainer.save_model("./best_model")
    tokenizer.save_pretrained("./best_model")
    
    # Save optimal thresholds for inference
    import json
    with open("./best_model/thresholds.json", "w") as f:
        json.dump({
            "class_names": label_classes.tolist(),
            "thresholds": tuned_results['thresholds']
        }, f)
    
    print("\nTraining complete! Model and thresholds saved to ./best_model")

# Inference code for using the trained model
def inference_with_optimal_thresholds(text, model_path="./best_model"):
    # Load model, tokenizer and thresholds
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    with open(f"{model_path}/thresholds.json", "r") as f:
        threshold_data = json.load(f)
        class_names = threshold_data["class_names"]
        thresholds = threshold_data["thresholds"]
    
    # Tokenize input
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
    
    # Apply class-specific thresholds
    predictions = []
    for i, class_name in enumerate(class_names):
        if logits[i] > thresholds[i]:
            predictions.append(class_name)
    
    # Return predictions and confidence scores
    confidence_scores = {
        class_name: float(logits[i])
        for i, class_name in enumerate(class_names)
    }
    
    return {
        "predictions": predictions,
        "confidence_scores": confidence_scores
    }

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
