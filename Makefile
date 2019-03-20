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

.ONESHELL:
.POSIX:
check:
	! grep -R /tmp lookout/style/*/tests
	flake8 --count
	pylint lookout

docker-check:
	version=$$(grep lookout-sdk-ml requirements.txt|cut -d"=" -f3)
	grep "FROM srcd/lookout-sdk-ml:$$version" Dockerfile >/dev/null
	docker pull srcd/lookout-sdk-ml:$$version

docker-test:
	docker ps | grep bblfshd  # bblfsh server should be run. Try `make bblfsh-start` command.
	docker run --rm -it --network host \
		-v $(current_dir)/.git:/style-analyzer/.git \
		-v $(current_dir)/lookout/core/server:/style-analyzer/lookout/core/server \
		--entrypoint python3 -w /style-analyzer \
			srcd/style-analyzer:test -m unittest discover

bblfsh-start:
	! docker ps | grep bblfshd # bblfsh server has been run already.
	docker run -d --rm --name style_analyzer_bblfshd --privileged -p 9432\:9432 \
		bblfsh/bblfshd\:v2.11.0 --log-level DEBUG
	docker exec style_analyzer_bblfshd bblfshctl driver install \
		javascript docker://bblfsh/javascript-driver\:v2.7.1

REPORTS_DIR ?= $(current_dir)/lookout/style/format/benchmarks/reports
REPORT_VERSION ?= untagged
REPORT_DIR ?= $(REPORTS_DIR)/$(REPORT_VERSION)
SMOKE_REPORT_DIR ?= $(REPORT_DIR)/js_smoke
NOISY_REPORT_DIR ?= $(REPORT_DIR)/noisy
QUALITY_REPORT_DIR ?= $(REPORT_DIR)/quality
SMOKE_INIT ?= ./lookout/style/format/benchmarks/data/js_smoke_init.tar.xz
QUALITY_REPORT_REPOS ?= ./lookout/style/format/benchmarks/data/quality_report_repos.csv
QUALITY_REPORT_REPOS_WITH_VNODE ?= ./lookout/style/format/benchmarks/data/quality_report_repos_with_vnodes_number.csv
BASE_REPORT_VERSION ?= 0.1.0

$(SMOKE_REPORT_DIR) $(NOISY_REPORT_DIR) $(QUALITY_REPORT_DIR):
	mkdir -p $@
report-smoke: $(SMOKE_REPORT_DIR)
	python3 -m lookout.style.format gen-smoke-dataset $(SMOKE_INIT) $(SMOKE_REPORT_DIR)/dataset \
		2>&1 | tee $(SMOKE_REPORT_DIR)/dataset_logs.txt
	python3 -m lookout.style.format eval-smoke-dataset $(SMOKE_REPORT_DIR)/dataset \
		$(SMOKE_REPORT_DIR)/report 2>&1 | tee $(SMOKE_REPORT_DIR)/report_logs.txt
	xz -k $(NOISY_REPORT_DIR)/logs.txt
report-noisy: $(NOISY_REPORT_DIR)
	python3 -m lookout.style.format quality-report-noisy --retrain -o $(NOISY_REPORT_DIR) \
		2>&1 | tee $(NOISY_REPORT_DIR)/logs.txt
	xz -k $(NOISY_REPORT_DIR)/logs.txt
report-quality: $(QUALITY_REPORT_DIR)
	python3 -m lookout.style.format --log-level DEBUG quality-report -o $(QUALITY_REPORT_DIR) \
		-i $(QUALITY_REPORT_REPOS) 2>&1 | tee $(QUALITY_REPORT_DIR)/logs.txt
	xz -k $(QUALITY_REPORT_DIR)/logs.txt
report-compare:
	python3 -m lookout.style.format compare-quality \
		--base $(REPORTS_DIR)/$(BASE_REPORT_VERSION)/quality/summary-train_report.md \
		--new $(QUALITY_REPORT_DIR)/summary-train_report.md -o -
	python3 -m lookout.style.format compare-quality \
		--base $(REPORTS_DIR)/$(BASE_REPORT_VERSION)/quality/summary-test_report.md \
		--new $(QUALITY_REPORT_DIR)/summary-test_report.md -o -

report-release: report-smoke report-noisy report-quality

expected-vnodes-number:
	python3 -m lookout.style.format calc-expected-vnodes-number -i $(QUALITY_REPORT_REPOS) \
		-o $(QUALITY_REPORT_REPOS_WITH_VNODE)

.PHONY: check docker-test bblfsh-start report-smoke report-noisy report-quality report-release expected-vnodes-number
