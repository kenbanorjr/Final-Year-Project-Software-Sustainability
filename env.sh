# Usage: source env.sh
# Sets environment variables for the pipeline in the current shell session.

# SonarQube
export SONAR_SCANNER="/mnt/c/Users/kenba/sonar-scanner/sonar-scanner/bin/sonar-scanner"
export SONAR_HOST_URL="http://localhost:9000"
export SONAR_TOKEN="your_sonar_token"

# LLM (Gemini)
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your_gemini_key"
export GEMINI_MODEL="gemini-pro"

# Uncomment to use OpenAI instead of Gemini:
# export LLM_PROVIDER="openai"
# export OPENAI_API_KEY="your_openai_key"
# export OPENAI_MODEL="gpt-4o-mini"

echo "Environment variables set for this shell session."
