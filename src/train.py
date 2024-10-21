import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoConfig, AutoTokenizer
from vin_bert_modeling import BERTInternVLChatModel
from utils import freeze_model_org, unfreeze_model_lora
from vin_bert_dataset import UITDataset, PromptTemplate, ImageProcessor
from trainer import Evaluation, Trainer, TrainingConfig
from peft import LoraConfig, get_peft_model

def main(args):
    # Khởi tạo quá trình phân tán
    if dist.is_available() and not dist.is_initialized():
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            dist.init_process_group(backend='nccl')

    # Xử lý thiết bị GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Lấy local_rank từ biến môi trường nếu có

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Thiết lập các đường dẫn dữ liệu
    train_image_dir = os.path.join(args.train_dir, "train-images")
    train_text_path = os.path.join(args.train_dir, "vimmsd-train.json")

    # Cấu hình huấn luyện
    config = TrainingConfig({
        "epochs": args.epochs,
        "train_batch": args.train_batch,
        "train_size": args.train_size,
        "val_batch": args.val_batch,
        "test_batch": args.test_batch,
        "log": True,
        "lr": args.learning_rate
    })

    # Tokenizer, ImageProcessor và Dataset
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True, use_fast=False)
    template = PromptTemplate()
    image_processor = ImageProcessor()
    dataset = UITDataset(train_image_dir, train_text_path, tokenizer, template, image_processor)

    # Cấu hình mô hình
    config = AutoConfig.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True)
    model = AutoModel.from_pretrained("5CD-AI/Vintern-1B-v2", low_cpu_mem_usage=True, trust_remote_code=True).to(device)

    vin_bert = BERTInternVLChatModel(
        config=config,
        model_mlp=model.mlp1,
        vision_model=model.vision_model,
        language_model=model.language_model
    )

    vin_bert.img_context_token_id = 151648
    vin_bert = vin_bert.to(device)

    # Đóng băng mô hình gốc
    freeze_model_org(vin_bert)

    # Định nghĩa cấu hình LoRA
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

    # Nếu có nhiều tiến trình thì sử dụng DDP
    if dist.is_initialized():
        vin_bert_with_lora = DDP(vin_bert_with_lora, device_ids=[local_rank], output_device=local_rank)

    # Khởi tạo Trainer và Evaluation
    evaluator = Evaluation(vin_bert_with_lora, dataset)
    trainer = Trainer(config, vin_bert_with_lora, dataset, evaluator)

    # Bắt đầu huấn luyện
    trainer.train(log=True)

    # Lưu mô hình đã huấn luyện
    model_dir = os.path.join(args.model_dir, "model.pth")
    trainer.save_model(model_dir)

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
