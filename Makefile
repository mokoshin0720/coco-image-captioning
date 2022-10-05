# dockerの起動
init:
	docker-compose build
	docker-compose up -d

# pythonの実行
run:
	docker-compose exec coco-image-captioning python src/sample.py

# dockerの削除
down:
	docker-compose down

install:
	docker-compose exec coco-image-captioning pip install -r requirements.txt