from sagemaker.pytorch import PyTorch
import sagemaker
import boto3

sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name='us-west-2'))
region = 'us-west-2'

role = "arn:aws:iam::308764189237:role/sagemaker-local"  

image_url = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.20.0-ubuntu20.04"


pytorch_estimator = PyTorch(
    region_name=region,
    sagemaker_session=sagemaker_session,
    entry_point='train.py',
    source_dir='src',
    role=role,
    instance_count=1,
    instance_type='ml.trn1.2xlarge',
    image_uri=image_url,
    model_data='s3://vinbert-training/model.tar.gz',
    distribution={
        "torch_distributed": {
            "enabled": True
        }
    },
    hyperparameters={
        'epochs': 3,
        'train_batch': 2,
        'val_batch': 2,
        'test_batch': 2,
        'learning_rate': 1e-5,
        'sample': 100,
        'model_dir': "/opt/ml/model",
        'output_dir': "/opt/ml/output",
    },

    use_spot_instances=True, 
    max_run=3600,             
    max_wait=7200    
)

pytorch_estimator.fit("s3://vinbert-training/data.tar.gz")
