import torch
import torch.nn as nn
from transformers import BertModel


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for sequence-based inputs.
    Default backbone: BERT.
    """
    def __init__(self, input_dim, num_classes, pretrained_model="bert-base-uncased", use_pretrained=True):
        """
        Args:
            input_dim (int): Input dimension of the sequence.
            num_classes (int): Number of classes for classification.
            pretrained_model (str): Name of the pretrained transformer model.
            use_pretrained (bool): Whether to use a pretrained model.
        """
        super().__init__()
        if use_pretrained:
            self.transformer = BertModel.from_pretrained(pretrained_model)
            self.hidden_dim = self.transformer.config.hidden_size
        else:
            self.transformer = BertModel(config=BertModel.config_class())
            self.hidden_dim = input_dim

        self.classifier_head = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for classification.
        Args:
            input_ids (torch.Tensor): Input token IDs (B, T).
            attention_mask (torch.Tensor, optional): Attention mask (B, T).
        Returns:
            torch.Tensor: Class logits (B, num_classes).
        """
        # Extract the sequence output or pooled output
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = transformer_outputs.pooler_output  # Use the CLS token embedding
        logits = self.classifier_head(pooled_output)
        return logits


if __name__ == "__main__":
    # Example usage
    num_classes = 10
    model = TransformerClassifier(input_dim=768, num_classes=num_classes, use_pretrained=True)

    # Dummy input (batch of 4 sequences with 20 tokens each)
    dummy_input_ids = torch.randint(0, 30522, (4, 20))  # Random token IDs
    dummy_attention_mask = torch.ones_like(dummy_input_ids)  # Full attention

    outputs = model(dummy_input_ids, dummy_attention_mask)
    print("Logits shape:", outputs.shape)  # Expected shape: (4, 10)
