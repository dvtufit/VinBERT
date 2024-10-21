import sagemaker
from sagemaker.estimator import Estimator
from datetime import datetime

# Khởi tạo session SageMaker
sagemaker_session = sagemaker.Session()

# Định nghĩa IAM role
role = "arn:aws:iam::308764189237:role/sagemaker-local"  

# Định nghĩa image URI từ ECR
image_uri = "308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_1"

# Khởi tạo PyTorch Estimator
estimator = Estimator(
    role=role,
    instance_count=1,  
    instance_type='ml.p3.2xlarge',  
    image_uri=image_uri, 
    use_spot_instances=True, 
    max_wait=3600, 
    max_run=3600,  
    checkpoint_s3_uri='s3://kaggle-save/checkpoints',  
    checkpoint_local_path='/opt/ml/checkpoints',  
    hyperparameters={
        'epochs': 1,
        'train_batch': 4,
        'train_size': 0.2,
        'learning_rate': 1e-5,
        'test_batch': 2,
        'val_batch': 2,
        'train_dir': '/opt/ml/input/data/train',
        'model_dir': '/opt/ml/model'
    },
    output_path='s3://kaggle-save/model-output'  ,
    distribution={
    'mpi': {
        'enabled': True,
        'processes_per_host': 1,  # Chỉ 1 tiến trình trên mỗi node
        'processes_per_gpu': 1,   # Một tiến trình trên mỗi GPU
        'custom_mpi_options': '-verbose'
    }
}

)

estimator.fit({'train' : 's3://vin-bert/'} ,job_name=f'vin-bert-training-job-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')  
