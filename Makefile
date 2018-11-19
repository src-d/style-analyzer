current_dir = $(shell pwd)

PROJECT = style-analyzer

DOCKERFILES = Dockerfile:$(PROJECT)
DOCKER_ORG = "srcd"

# Including ci Makefile
CI_REPOSITORY ?= https://github.com/src-d/ci.git
CI_BRANCH ?= v1
CI_PATH ?= .ci
MAKEFILE := $(CI_PATH)/Makefile.main
$(MAKEFILE):
	git clone --quiet --depth 1 -b $(CI_BRANCH) $(CI_REPOSITORY) $(CI_PATH);
-include $(MAKEFILE)


.PHONY: check
check:
	! grep -R /tmp lookout/style/*/tests
	flake8 --count
	pylint lookout

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
	docker run -d --rm --name style_analyzer_bblfshd --privileged -p 9432\:9432 \
		bblfsh/bblfshd\:v2.5.0
	docker exec style_analyzer_bblfshd bblfshctl driver install \
		javascript docker://bblfsh/javascript-driver\:v1.2.0
