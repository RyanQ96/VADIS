
import torch
import wandb
from transformers import BertTokenizerFast, HfArgumentParser
from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset

from model import DynamicDocumentEmbeddingModel, DynamicDocumentEmbeddingConfig
from dataloader import TrainingDataset
from model.utils import contrastive_loss
from arguments import TrainingArguments


def train_model(model, data_loader, args: TrainingArguments):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    wandb.init(project=args.project_name, name=args.report_name) 

    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            # Move inputs to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            doc_input_ids = batch['doc_input_ids'].to(device)
            doc_attention_mask = batch['doc_attention_mask'].to(device)

            
            outputs = model(
                query_input_ids,
                query_attention_mask,
                doc_input_ids,
                doc_attention_mask,
                training=True,
                use_dual_loss=args.use_dual_loss
            )
            
            if args.use_dual_loss:
                relevance_scores, labels, entropy_loss = outputs
            else:
                relevance_scores, labels = outputs
                entropy_loss = 0.0

            # Compute contrastive loss
            contrastive_loss_value = contrastive_loss(relevance_scores)

            # Total loss
            loss = contrastive_loss_value + args.entropy_weight * entropy_loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            # Log metrics to W&B
            wandb.log({'batch_loss': loss.item()})

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch +1}/{args.num_epochs}, Loss: {avg_loss}")
        wandb.log({'epoch_loss': avg_loss})
        model.save_pretrained(f"{args.model_save_path}/{args.report_name}/epoch_{epoch + 1}")

    wandb.finish()
    

if __name__ == '__main__':
    # Load the training arguments
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model_path)

    datasets = []
    for dataset_name in args.datasets:
        if dataset_name.lower() == 'squad':
            dataset = load_dataset('squad', split='train')
        elif dataset_name.lower() == 'emrqa':
            dataset = load_dataset('Eladio/emrqa-msquad', split='train')
        elif dataset_name.lower() == 'triviaqa':
            dataset = load_dataset('trivia_qa', 'rc', split='train')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Create custom dataset
        custom_dataset = TrainingDataset(dataset, tokenizer, max_length=args.max_length)
        datasets.append(custom_dataset)

    if len(datasets) > 1:
        combined_dataset = ConcatDataset(datasets)
    else:
        combined_dataset = datasets[0]
    
    data_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    
    config = DynamicDocumentEmbeddingConfig(model_name=args.pretrained_model_path)
    model = DynamicDocumentEmbeddingModel(config)

    train_model(model, data_loader, args)