import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast, PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class DynamicDocumentEmbeddingConfig(PretrainedConfig):
    model_type = "dynamic_document_embedding"

    def __init__(self, model_name='dmis-lab/biobert-v1.1', **kwargs): # training starts from the biobert-checkpoint
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_size = kwargs.get('hidden_size', 768)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        with torch.no_grad():
            self.value_proj.weight.fill_(1.)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_embedding, key_embeddings, key_padding_mask=None, training=True):
        batch_size = query_embedding.size(0)
        seq_len = key_embeddings.size(1)

        # Project query, keys
        Q = self.query_proj(query_embedding)          # [batch_size, hidden_size]
        K = self.key_proj(key_embeddings)             # [batch_size, seq_len, hidden_size]

        # Compute attention scores between each query and all documents' tokens
        # Resulting in [batch_size, batch_size, seq_len]
        attention_scores = torch.einsum("qh,dlh->qdl", Q, K)  # [batch_size, batch_size, seq_len]
        attention_scores = attention_scores / (K.size(-1) ** 0.5)  # Scale scores

        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, batch_size, seq_len]
            if training:
                attention_scores = attention_scores.masked_fill(key_padding_mask_expanded, float(0))
            else:
                attention_scores = attention_scores.masked_fill(key_padding_mask_expanded, float('-inf'))

        return attention_scores  # [batch_size, batch_size, seq_len]


class DynamicDocumentEmbeddingModel(PreTrainedModel):
    config_class = DynamicDocumentEmbeddingConfig

    def __init__(self, config):
        super().__init__(config)
        self.query_encoder = BertModel.from_pretrained(config.model_name)
        self.doc_encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.query_encoder.config.hidden_size

        # Use the custom attention layer
        self.attention_layer = AttentionLayer(self.hidden_size)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask, training=True, use_dual_loss=True):
        batch_size = query_input_ids.size(0)

        # Encode the query into CLS embedding
        query_outputs = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_cls_embedding = query_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Encode the document into token embeddings
        doc_outputs = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        doc_token_embeddings = doc_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Exclude CLS token from document token embeddings
        doc_token_embeddings = doc_token_embeddings[:, 1:, :]  # [batch_size, seq_len - 1, hidden_size]
        doc_attention_mask = doc_attention_mask[:, 1:]         # [batch_size, seq_len - 1]

        # Prepare key padding mask
        key_padding_mask = ~doc_attention_mask.bool()  # [batch_size, seq_len - 1]

        # Compute raw attention scores
        attention_scores = self.attention_layer(
            query_cls_embedding,        # [batch_size, hidden_size]
            doc_token_embeddings,       # [batch_size, seq_len - 1, hidden_size]
            key_padding_mask,           # [batch_size, seq_len - 1]
            training=training
        )  # [batch_size, batch_size, seq_len - 1]

        if training:
            
            if use_dual_loss:
                # Compute attention weights
                attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, batch_size, seq_len - 1]

                # Compute entropy loss
                attention_weights_correct = attention_weights[torch.arange(batch_size), torch.arange(batch_size), :]  # [batch_size, seq_len - 1]
                entropy_loss = - (attention_weights_correct * torch.log(attention_weights_correct + 1e-12)).sum(dim=1).mean()

            else:
                entropy_loss = 0.0

            # Sum raw attention scores over tokens and normalize by number of tokens
            num_tokens = doc_attention_mask.sum(dim=1).float()      # [batch_size]
            num_tokens = num_tokens.unsqueeze(0).expand(batch_size, -1)  # [batch_size, batch_size]
            relevance_scores = attention_scores.sum(dim=2) / num_tokens  # [batch_size, batch_size]

            # Labels for contrastive loss (diagonal elements)
            labels = torch.arange(batch_size).to(relevance_scores.device)

            if use_dual_loss:
                return relevance_scores, labels, entropy_loss
            else:
                return relevance_scores, labels
        else:
            # Apply softmax to compute attention weights
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, batch_size, seq_len - 1]

            # Compute document embeddings
            V = self.attention_layer.value_proj(doc_token_embeddings)  # [batch_size, seq_len - 1, hidden_size]
            V_expanded = V.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, batch_size, seq_len -1, hidden_size]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, batch_size, seq_len -1, 1]
            doc_embeddings = (attention_weights * V_expanded).sum(dim=2)  # [batch_size, batch_size, hidden_size]

            return doc_embeddings

    def get_document_embedding(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        with torch.no_grad():
            batch_size = query_input_ids.size(0)
            # Encode the query into CLS embedding
            query_outputs = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
            query_cls_embedding = query_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

            # Encode the document into token embeddings
            doc_outputs = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
            doc_token_embeddings = doc_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

            # Exclude CLS token from document token embeddings
            doc_token_embeddings = doc_token_embeddings[:, 1:, :]  # [batch_size, seq_len - 1, hidden_size]
            doc_attention_mask = doc_attention_mask[:, 1:]         # [batch_size, seq_len - 1]

            # Prepare key padding mask
            key_padding_mask = ~doc_attention_mask.bool()  # [batch_size, seq_len - 1]

            # Compute attention scores
            attention_scores = self.attention_layer(
                query_cls_embedding,        # [batch_size, hidden_size]
                doc_token_embeddings,       # [batch_size, seq_len - 1, hidden_size]
                key_padding_mask,           # [batch_size, seq_len - 1]
                training=False
            )  

            attention_weights = self.attention_layer.softmax(attention_scores, dim=-1) 
            # Compute document embeddings
            V = self.attention_layer.value_proj(doc_token_embeddings)  # [batch_size, seq_len - 1, hidden_size]
            V_expanded = V.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, batch_size, seq_len -1, hidden_size]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, batch_size, seq_len -1, 1]
            doc_embeddings = (attention_weights * V_expanded).sum(dim=2)  # [batch_size, batch_size, hidden_size]

            return doc_embedding, attention_weights  # [batch_size, hidden_size], [batch_size, seq_len - 1]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = DynamicDocumentEmbeddingConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location=device)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")