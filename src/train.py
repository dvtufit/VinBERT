import os
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from vin_bert_modeling import BERTInternVLChatModel
from utils import freeze_model_org, unfreeze_model_lora
from vin_bert_dataset import UITDataset, PromptTemplate, ImageProcessor
from trainer import Evaluation, TrainingConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

def main(args):

    train_image_dir = os.path.join(args.train_dir, "train-images")
    train_text_path = os.path.join(args.train_dir, "vimmsd-train.json")

    config = TrainingConfig({
        "epochs": args.epochs,
        "train_batch": args.train_batch,
        "train_size": args.train_size,
        "val_batch": args.val_batch,
        "test_batch": args.test_batch,
        "log": True,
        "lr": args.learning_rate
    })

    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True, use_fast=False)
    template = PromptTemplate()
    image_processor = ImageProcessor()
    dataset = UITDataset(train_image_dir, train_text_path, tokenizer, template, image_processor)

    config_model = AutoConfig.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True)
    model = AutoModel.from_pretrained("5CD-AI/Vintern-1B-v2", low_cpu_mem_usage=True, trust_remote_code=True)
    
    vin_bert = BERTInternVLChatModel(
        config=config_model,
        model_mlp=model.mlp1,
        vision_model=model.vision_model,
        language_model=model.language_model
    )

    vin_bert.img_context_token_id = 151648

    freeze_model_org(vin_bert)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "patch_embedding",
            "language_model.embed_tokens",
            "self.query",
            "self.key",
            "self.value",
            "output.dense"
        ]
    )

    vin_bert_with_lora = get_peft_model(vin_bert, lora_config)
    unfreeze_model_lora(vin_bert_with_lora)


    evaluator = Evaluation(vin_bert_with_lora, dataset)

    train_size = int(0.7 * len(dataset)) 
    test_size = len(dataset) - train_size 

    val_size = int(0.5 * test_size) 
    test_size = test_size - val_size 

    train_dataset,  test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch, shuffle=False) 

    optimizer = AdamW(vin_bert_with_lora.parameters(), lr=config.lr, weight_decay=0.01)

    for epoch in range(config.epochs):
        vin_bert_with_lora.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            inputs = dataset.create_inputs(batch)

            optimizer.zero_grad()

            outputs = vin_bert_with_lora(inputs)
            logits = outputs.logits
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/train'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_batch', type=int, default=8)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_batch', type=int, default=8)
    parser.add_argument('--test_batch', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    args = parser.parse_args()
    main(args)
