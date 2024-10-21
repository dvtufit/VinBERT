login:
	aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com
build:
	docker build -t vin_bert .
tag:
	docker tag vin_bert 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_1
push:
	docker push 308764189237.dkr.ecr.ap-southeast-1.amazonaws.com/vin_bert:vin_bert_1

run: build tag push