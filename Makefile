.PHONY: install lint format test docker-build docker-run clean jenkins-build jenkins-run jenkins-logs jenkins-password jenkins-stop

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

# Jenkins
jenkins-build:
	docker build -t madewithml-jenkins -f Dockerfile.jenkins .

jenkins-run:
	-docker stop jenkins && docker rm jenkins
	mkdir -p efs logs
	docker run -d \
		--name jenkins \
		--restart unless-stopped \
		-p 8081:8080 \
		-p 50000:50000 \
		-v jenkins_home:/var/jenkins_home \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v $(PWD):/workspace \
		--group-add "$(shell stat -c '%g' /var/run/docker.sock)" \
		-e DOCKER_HOST=unix:///var/run/docker.sock \
		madewithml-jenkins

jenkins-logs:
	docker logs -f jenkins

jenkins-password:
	docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword

jenkins-stop:
	docker stop jenkins && docker rm jenkins

jenkins-shell:
	docker exec -it -u root jenkins /bin/bash

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/ outputs/ efs/ *.egg-info/ dist/ build/
