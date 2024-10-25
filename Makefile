login:
	aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com
build:
	docker build -t vin_bert .
tag:
	docker tag vin_bert 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_1
push:
	docker push 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_1

run: build tag push

build-base:
	docker build -t vin_bert_base -f Dockerfile.flash-base .

tag-base:
	docker tag vin_bert_base 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_base

push-base:
	docker push 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_base

run-base: build-base tag-base push-base

build-training:
	docker build -t vin_bert_training -f Dockerfile.training .

tag-training:
	docker tag vin_bert_training 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_training

push-training:
	docker push 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_training

run-training: build-training tag-training push-training
