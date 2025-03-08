import os

os.environ["HF_HOME"] = "../.cache"
os.environ["TORCH_HOME"] = "../.cache"

import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from tqdm import tqdm

torch.hub.set_dir("../.cache")


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT Embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width] --> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        x = x.squeeze(1)
        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),  # 7 emotions
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # 3 sentiments
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features
        combined_features = torch.cat(
            [text_features, video_features, audio_features], dim=1
        )  # [batch_size, 128 * 3]

        fused_features = self.fusion_layer(combined_features)

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {"emotions": emotion_output, "sentiments": sentiment_output}


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset size
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print(f"\nDataset sizes:")
        print(f"Training samples: {train_size:,}")  # 10000 -> 10,000
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        # Logging to tensorboard
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")  # Dec17_14-22-35
        base_dir = (
            "opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        )
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # Very high: 1, high: 0.1 - 0.01, medium: 1e-3, low: 1e-4, very low: 1e-5
        self.optimizer = torch.optim.Adam(
            [
                {"params": model.text_encoder.parameters(), "lr": 8e-6},
                {"params": model.video_encoder.parameters(), "lr": 8e-5},
                {"params": model.audio_encoder.parameters(), "lr": 8e-5},
                {"params": model.fusion_layer.parameters(), "lr": 5e-4},
                {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
                {"params": model.sentiment_classifier.parameters(), "lr": 5e-4},
            ],
            weight_decay=1e-5,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=2
        )

        self.current_train_losses = None

        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else:  # Validation
            self.writer.add_scalar(
                "loss/total/train", self.current_train_losses["total"], self.global_step
            )
            self.writer.add_scalar("loss/total/val", losses["total"], self.global_step)
            self.writer.add_scalar(
                "loss/emotion/train",
                self.current_train_losses["emotion"],
                self.global_step,
            )
            self.writer.add_scalar(
                "loss/emotion/val", losses["emotion"], self.global_step
            )
            self.writer.add_scalar(
                "loss/sentiment/train",
                self.current_train_losses["sentiment"],
                self.global_step,
            )
            self.writer.add_scalar(
                "loss/sentiment/val", losses["sentiment"], self.global_step
            )

        if metrics:
            self.writer.add_scalar(
                f"{phase}/emotion_precision",
                metrics["emotion_precision"],
                self.global_step,
            )
            self.writer.add_scalar(
                f"{phase}/emotion_accuracy",
                metrics["emotion_accuracy"],
                self.global_step,
            )
            self.writer.add_scalar(
                f"{phase}/sentiment_precision",
                metrics["sentiment_precision"],
                self.global_step,
            )
            self.writer.add_scalar(
                f"{phase}/sentiment_accuracy",
                metrics["sentiment_accuracy"],
                self.global_step,
            )

    def train_epoch(self):
        self.model.train()
        running_loss = {"total": 0, "emotion": 0, "sentiment": 0}

        for batch in self.train_loader:
            batch_num = 0
            print(f"Batch: {batch_num}")
            batch_num += 1
            device = next(self.model.parameters()).device
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device),
            }
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Calculate losses using raw logits
            emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiments"], sentiment_labels
            )
            total_loss = emotion_loss + sentiment_loss

            # Backward pass. Calculate gradients
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track losses
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()

            self.log_metrics(
                {
                    "total": total_loss.item(),
                    "emotion": emotion_loss.item(),
                    "sentiment": sentiment_loss.item(),
                }
            )

            self.global_step += 1

        return {k: v / len(self.train_loader) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {"total": 0, "emotion": 0, "sentiment": 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    "input_ids": batch["text_input"]["input_ids"].to(device),
                    "attention_mask": batch["text_input"]["attention_mask"].to(device),
                }
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)
                emotion_labels = batch["emotion_label"].to(device)
                sentiment_labels = batch["sentiment_label"].to(device)

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Calculate losses using raw logits
                emotion_loss = self.emotion_criterion(
                    outputs["emotions"], emotion_labels
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiments"], sentiment_labels
                )
                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(
                    outputs["emotions"].argmax(dim=1).cpu().numpy()
                )
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs["sentiments"].argmax(dim=1).cpu().numpy()
                )
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track losses
                losses["total"] += total_loss.item()
                losses["emotion"] += emotion_loss.item()
                losses["sentiment"] += sentiment_loss.item()

        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}

        # Compute precision and accuracy
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average="weighted"
        )
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average="weighted"
        )
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)

        self.log_metrics(
            avg_loss,
            metrics={
                "emotion_precision": emotion_precision,
                "emotion_accuracy": emotion_accuracy,
                "sentiment_precision": sentiment_precision,
                "sentiment_accuracy": sentiment_accuracy,
            },
            phase=phase,
        )

        if phase == "val":
            self.scheduler.step(avg_loss["total"])

        return avg_loss, {
            "emotion_precision": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy,
        }
