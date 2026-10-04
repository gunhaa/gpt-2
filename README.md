
# 0. 가상환경 활성화
source .venv/bin/activate

# 1. 패키지 설치
pip install tiktoken

# 2. 설치된 패키지 목록을 파일로 저장
pip freeze > requirements.txt

# 3. 저장된 명세서로 동일하게 패키지 일괄 설치
pip install -r requirements.txt