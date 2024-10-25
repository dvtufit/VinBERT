import sagemaker
# from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch 
from datetime import datetime

# Khởi tạo session SageMaker
sagemaker_session = sagemaker.Session()
region = 'ap-southeast-1'

# Định nghĩa IAM role
role = "arn:aws:iam::308764189237:role/sagemaker-local"  

# Định nghĩa image URI từ ECR
image_uri = "308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_training"

# Khởi tạo PyTorch Estimator với cấu hình cho SMP
estimator = PyTorch(
    region_name=region,
    entry_point='train.py',  # Đường dẫn đến file huấn luyện
    source_dir='test',  # Đường dẫn đến thư mục chứa source code
    role=role,
    instance_count=1,  # Số lượng instance
    instance_type='ml.trn1.*',  # Loại instance phù hợp
    image_uri=image_uri, 
    # use_spot_instances=True, 
    # max_wait=3600, 
    max_run=3600,  
    checkpoint_s3_uri='s3://kaggle-save/checkpoints',  
    checkpoint_local_path='/opt/ml/checkpoints',  
    hyperparameters={
        'epochs': 10,  # Thay đổi số epochs
        'train_batch': 8,  # Thay đổi batch size cho huấn luyện
        'train_size': 0.8,  # Kích thước tập huấn luyện
        'learning_rate': 1e-5,  # Tỷ lệ học
        'test_batch': 8,  # Batch size cho kiểm tra
        'val_batch': 8,  # Batch size cho xác thực
        'train_dir': '/opt/ml/input/data/train',  # Đường dẫn đến dữ liệu huấn luyện
        'model_dir': '/opt/ml/model',  # Đường dẫn để lưu mô hình,
        # --nproc_per_node=2
        # 'nproc_per_node': 4  # Số lượng process trên mỗi node
    },
    output_path='s3://kaggle-save/model-output',  
    distribution={ "torch_distributed": { "enabled": True } }  # torchrun, activates SMDDP AllGather
)

# Bắt đầu huấn luyện
estimator.fit( job_name=f'vin-bert-training-job-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
