from configuration_internvl_chat import InternVLChatConfig
from transformers import AutoModel, AutoConfig
from vin_bert_modeling import BERTInternVLChatModel
from peft import LoraConfig, get_peft_model

def freeze_model_org(vin_bert):
    for param in vin_bert.parameters():
        param.requires_grad = False

    total_model_params_org = sum(p.numel() for p in vin_bert.parameters())
    
    print(f'Total parameter in vin_bert: {total_model_params_org//1000000} M')

    total_trainable_params_org = sum(p.numel() for p in vin_bert.parameters() if p.requires_grad)

    print(f'Total trainable parameter in vin_bert: {total_trainable_params_org/1000000} M')



def unfreeze_model_lora(vin_bert_with_lora):  
    for param in vin_bert_with_lora.mlp1.parameters():  
        param.requires_grad = True

    for param in vin_bert_with_lora.phobert.pooler.parameters():  
        param.requires_grad = True
    
    for param in vin_bert_with_lora.phobert.mlp.parameters():  
        param.requires_grad = True

    for param in vin_bert_with_lora.phobert.classifier.parameters():  
        param.requires_grad = True
             
    
    lora_params = []

    for name, param in vin_bert_with_lora.named_parameters():
        if "lora" in name:  
            lora_params.append(param)

    total_lora_params = sum(p.numel() for p in lora_params)

    print(f'Total trainable of LoRA: {total_lora_params/1000000}')
    
    total_trainable_params_org = sum(p.numel() for p in vin_bert_with_lora.parameters() if p.requires_grad)
    
    print(f'Total trainable parameter in vin_bert_with_lora: {total_trainable_params_org/1000000} M')


if __name__ == '__main__':
    config = AutoConfig.from_pretrained(
        "5CD-AI/Vintern-1B-v2" ,
        trust_remote_code=True, 
    )
    model = AutoModel.from_pretrained(
        "5CD-AI/Vintern-1B-v2", 
    #     torch_dtype=torch.float16,
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
            "q_proj", "k_proj", "v_proj", "o_proj",# Attention modules trong llm
            "patch_embedding",
            "language_model.embed_tokens" 
            "self.query",                         # Phobert Attention
            "self.key",
            "self.value",
            "output.dense"                        # Phobert MLP
        ]
    )

    vin_bert_with_lora = get_peft_model(vin_bert, lora_config)

    unfreeze_model_lora(vin_bert_with_lora)

