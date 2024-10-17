import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from vin_bert_modeling import BERTInternVLChatModel
from peft import LoraConfig, get_peft_model
from utils import freeze_model_org, unfreeze_model_lora
from vin_bert_dataset import UITDataset, PromptTemplate, ImageProcessor
from torch.utils.data import DataLoader


if __name__ == '__main__':
    train_image_dir = "data/train-images"
    train_text_path = "data/vimmsd-train.json"

    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True, use_fast=False)
    template = PromptTemplate()
    image_processor = ImageProcessor()
    dataset = UITDataset(train_image_dir, train_text_path, tokenizer, template, image_processor)

    train_size = int(0.7 * len(dataset)) 
    test_size = len(dataset) - train_size 

    val_size = int(0.5 * test_size) 
    test_size = test_size - val_size 

    train_dataset,  test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False) 


    config = AutoConfig.from_pretrained(
        "5CD-AI/Vintern-1B-v2" ,
        trust_remote_code=True, 
    )
    model = AutoModel.from_pretrained(
        "5CD-AI/Vintern-1B-v2", 
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()

    vin_bert = BERTInternVLChatModel(
        config = config,
        model_mlp=model.mlp1, 
        vision_model=model.vision_model, 
        language_model=model.language_model
    )
    vin_bert.img_context_token_id = 151648

    vin_bert = vin_bert.cuda()
    freeze_model_org(vin_bert)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "patch_embedding",
            "language_model.embed_tokens" 
            "self.query",                         
            "self.key",
            "self.value",
            "output.dense"                        
        ]
    )

    vin_bert_with_lora = get_peft_model(vin_bert, lora_config)
    unfreeze_model_lora(vin_bert_with_lora)

