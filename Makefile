current_dir = $(shell pwd)

.PHONY: check
check:
	flake8 --config .flake8-code . --count
	flake8 --config .flake8-doc . --count
	pylint lookout

.PHONY: docker-build
docker-build:
	docker build -t srcd/style-analyzer .

.PHONY: docker-test
docker-test:
	docker ps | grep bblfshd  # bblfsh server should be run. Try `make bblfsh-start` command.
	docker run --rm -it --network host \
		-v $(current_dir)/.git:/style-analyzer/.git \
		-v $(current_dir)/lookout/core/server:/style-analyzer/lookout/core/server \
		--entrypoint python3 -w /style-analyzer \
			srcd/style-analyzer -m unittest discover

.PHONY: bblfsh-start
bblfsh-start:
	! docker ps | grep bblfshd # bblfsh server has been run already.
	docker run -d --name style_analyzer_bblfshd --privileged -p 9432\:9432 bblfsh/bblfshd\:v2.5.0
	docker exec style_analyzer_bblfshd bblfshctl driver install \
		javascript docker://bblfsh/javascript-driver\:v1.2.0
