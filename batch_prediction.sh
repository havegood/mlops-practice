# Codespaces에서만 발생하는 버그 같습니다 :/
# 오류 발생 원인은 sudo에서 pipenv를 인식을 못해서 가상환경을 생성하지 못해서 발생한 문제입니다
# 해결방법은 pipenv 가상환경의 Python 경로를 직접 입력해주면 됩니다. :)
# pipenv shell 상태에서 which python 한 경로를 PIPENV_PYTHON 경로에 넣어주면 됩니다

PIPENV_PYTHON=/home/codespace/.local/share/virtualenvs/mlops-practice-2_cZx3tu/bin/python
ROOT_PATH=/workspaces//mlops-practice

$PIPENV_PYTHON $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1
