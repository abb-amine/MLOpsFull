pipeline {
    agent any

    environment {
        PYTHONPATH = "${WORKSPACE}"
        PATH = "${HOME}/.local/bin:${env.PATH}"
    }

    stages {
        stage('Setup') {
            steps {
                checkout scm
                sh 'pip install --upgrade pip --break-system-packages'
                sh 'pip install --retries 5 --timeout 180 -r requirements.txt --break-system-packages'
                sh 'pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --break-system-packages'
                sh 'pip install -e . pytest-cov flake8 black isort --break-system-packages'
            }
        }

        stage('Lint') {
            steps {
                sh 'flake8 madewithml/ tests/'
                sh 'black --check madewithml/ tests/'
                sh 'isort --check-only madewithml/ tests/'
            }
        }

        stage('Test') {
            steps {
                sh 'mkdir -p test-results && pytest tests/ -v -m "not requires_scibert" --cov=madewithml --cov-report=xml --cov-report=term --junitxml=test-results/results.xml'
            }
        }

        stage('Build') {
            agent {
                docker {
                    image 'docker:latest'
                    args '-v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                sh 'docker build -t madewithml .'
            }
        }
    }

    post {
        always {
            junit 'test-results/**/*.xml'
        }
        failure {
            echo 'Pipeline failed! Check the logs.'
        }
        success {
            echo 'Pipeline succeeded! Image madewithml is built.'
        }
    }
}
