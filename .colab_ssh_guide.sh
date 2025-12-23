#!/bin/bash
# Colab SSH Setup - Quick Guide
# 이 가이드를 따라 Colab에서 SSH를 설정하세요

echo "============================================"
echo "  Colab SSH 설정 가이드"
echo "============================================"
echo ""
echo "1️⃣ 브라우저에서 Colab이 열렸으면:"
echo "   - 'File' → 'New notebook' 클릭"
echo ""
echo "2️⃣ 첫 번째 셀에 다음 코드 복사/붙여넣기:"
echo ""
cat << 'EOF'
# SSH 설정
!pip install colab-ssh --quiet
from colab_ssh import launch_ssh_cloudflared

print("🔌 SSH 터널 설정 중...")
launch_ssh_cloudflared(password="sullivan2025")

print("\n✅ SSH 준비 완료!")
print("위에 출력된 Hostname을 복사하세요")
EOF
echo ""
echo "3️⃣ 셀 실행 (Shift + Enter)"
echo ""
echo "4️⃣ 출력된 Hostname을 확인한 후,"
echo "   이 터미널에서 다음 명령 실행:"
echo ""
echo "   ./scripts/colab_connect.sh save"
echo ""
echo "5️⃣ Hostname 입력 후 SSH 접속:"
echo ""
echo "   ./scripts/colab_connect.sh connect"
echo ""
echo "============================================"
