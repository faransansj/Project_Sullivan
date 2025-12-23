# ============================================
# Project Sullivan - One-Click Colab Setup
# ============================================
# 이 셀을 클릭하고 실행(Control+Enter)하면 모든 설정과 학습이 시작됩니다.

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone Repository
import os
if not os.path.exists('/content/Project_Sullivan'):
    !git clone https://github.com/faransansj/Project_Sullivan.git
    %cd /content/Project_Sullivan
else:
    %cd /content/Project_Sullivan
    !git pull

# 3. Install Dependencies
!pip install -r requirements.txt

# 4. Extract Dataset
# Google Drive의 Dataset.zip을 /content/sullivan_data에 압축 해제합니다.
# 이미 추출되어 있다면 빠르게 스킵됩니다.
!python scripts/extract_gdrive_dataset.py

# 5. Start Training
# --streaming 플래그를 사용하여 대용량 데이터를 안전하게 학습합니다.
!python scripts/train_transformer.py \
    --config configs/colab_gdrive_config.yaml \
    --streaming \
    --gpus 1
