pipeline {
    agent any

    environment {
        PYTHONPATH = "${WORKSPACE}"
    }

    stages {
        stage('Setup') {
            steps {
                dir("${WORKSPACE}") {
                    sh 'pip install --upgrade pip --break-system-packages'
                    sh 'pip install -r requirements.txt --break-system-packages'
                    sh 'pip install -e . pytest-cov flake8 black isort --break-system-packages'
                }
            }
        }

        stage('Lint') {
            steps {
                dir("${WORKSPACE}") {
                    sh 'flake8 madewithml/ tests/'
                    sh 'black --check madewithml/ tests/'
                    sh 'isort --check-only madewithml/ tests/'
                }
            }
        }

        stage('Test') {
            steps {
                dir("${WORKSPACE}") {
                    sh 'pytest tests/ -v -m "not requires_scibert" --cov=madewithml --cov-report=xml --cov-report=term'
                }
            }
        }

        stage('Build') {
            steps {
                dir("${WORKSPACE}") {
                    sh 'docker build -t madewithml .'
                }
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
