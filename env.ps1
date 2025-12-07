# Usage: .\env.ps1
# Sets environment variables for the pipeline in the current PowerShell session.

# SonarQube (replace placeholders with your values)
$Env:SONAR_SCANNER = "C:\path\to\sonar-scanner.bat"
$Env:SONAR_HOST_URL = "http://localhost:9000"
$Env:SONAR_TOKEN = "your_sonar_token"

# LLM (Gemini; replace with your key/model)
$Env:LLM_PROVIDER = "gemini"
$Env:GEMINI_API_KEY = "your_gemini_key"
$Env:GEMINI_MODEL = "gemini-pro"

# Uncomment to use OpenAI instead of Gemini:
# $Env:LLM_PROVIDER = "openai"
# $Env:OPENAI_API_KEY = "your_openai_key"
# $Env:OPENAI_MODEL = "gpt-4o-mini"

Write-Host "Environment variables set for this session."
