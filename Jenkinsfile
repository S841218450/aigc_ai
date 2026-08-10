pipeline {
    agent any

    environment {
        APP_IMAGE = 'aigc_ai'
        APP_TAG = "${env.BUILD_NUMBER}"
    }

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')   // 本项目依赖 langchain/chromadb，构建耗时长
        disableConcurrentBuilds()
    }

    stages {
        stage('拉取代码 Checkout') {
            steps {
                checkout scm
                sh """
                    echo "=== WebHook分支：${env.GIT_BRANCH}"
                """
            }
        }

        stage('构建Docker镜像') {
            steps {
                sh """
                    docker build -t ${APP_IMAGE}:${APP_TAG} .
                    docker tag ${APP_IMAGE}:${APP_TAG} ${APP_IMAGE}:latest
                    echo "✅ 镜像构建完成 ${APP_IMAGE}:${APP_TAG}"
                """
            }
        }

        stage('本地直接部署') {
            steps {
                script {
                    String realBranch = env.GIT_BRANCH.replace("origin/", "")
                    println("处理后分支：${realBranch}")

                    if (realBranch == 'main') {
                        println("✅ main分支，执行本地部署")
                        sh """
                            echo "停止旧容器"
                            docker stop aigc-ai || true
                            docker rm aigc-ai || true

                            echo "启动容器"
                            CONTAINER_ID=\$(docker run -d --name aigc-ai \
                                -p 8000:8000 \
                                -v /home/www/aigc_ai/logs:/app/logs \
                                -v /home/www/aigc_ai/chroma_db:/app/chroma_db \
                                -v /home/www/aigc_ai/data:/app/data \
                                -e DATABASE_URL=sqlite:////app/data/aigc_platform.db \
                                --env-file /home/www/aigc_ai/.env \
                                --restart unless-stopped \
                                ${APP_IMAGE}:${APP_TAG})
                            echo "容器ID: \${CONTAINER_ID}"

                            sleep 4
                            echo "==== 所有容器 ===="
                            docker ps -a | grep aigc-ai

                            if ! docker ps --filter "name=aigc-ai" | grep aigc-ai ; then
                                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                                echo "容器后台退出，打印应用日志"
                                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                                docker logs aigc-ai
                                exit 1
                            fi

                            echo "等待健康检查通过"
                            for i in \$(seq 1 120); do
                                if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
                                    echo "✅ 健康检查通过"
                                    break
                                fi
                                if [ \$i -eq 120 ]; then
                                    echo "❌ 健康检查超时，打印应用日志"
                                    docker logs aigc-ai
                                    exit 1
                                fi
                                sleep 1
                            done

                            docker image prune -f
                            echo "✅ 服务正常运行"
                        """
                    } else {
                        println("❌ 非main分支，跳过部署")
                    }
                }
            }
        }
    }

    post {
        success {
            echo "✅ 流水线执行成功！镜像版本：${APP_TAG}"
        }
        failure {
            echo "❌ 流水线执行失败，请查看上方应用崩溃日志"
        }
        always {
            cleanWs()
        }
    }
}
