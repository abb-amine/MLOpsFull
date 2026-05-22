.PHONY: install lint format test docker-build docker-run clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

lint:
	flake8 madewithml/ tests/

format:
	black madewithml/ tests/
	isort madewithml/ tests/

test:
	pytest tests/ -v -m "not requires_scibert"

test-all:
	pytest tests/ -v

docker-build:
	docker build -t madewithml .

docker-run:
	docker run --rm -it \
		-e GITHUB_USERNAME=$(GITHUB_USERNAME) \
		-v $(PWD)/efs:/app/efs \
		-v $(PWD)/logs:/app/logs \
		madewithml $(CMD)

docker-shell:
	docker run --rm -it \
		-e GITHUB_USERNAME=$(GITHUB_USERNAME) \
		-v $(PWD)/efs:/app/efs \
		-v $(PWD)/logs:/app/logs \
		--entrypoint /bin/bash \
		madewithml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/ outputs/ efs/ *.egg-info/ dist/ build/
