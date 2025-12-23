#!/bin/bash
# ============================================
# Google Drive Sync Script using rclone
# ============================================
# Usage:
#   ./scripts/gdrive_sync.sh setup     - 초기 설정
#   ./scripts/gdrive_sync.sh push      - 로컬 → Drive 업로드
#   ./scripts/gdrive_sync.sh pull      - Drive → 로컬 다운로드
#   ./scripts/gdrive_sync.sh list      - Drive 파일 목록
#   ./scripts/gdrive_sync.sh status    - 연결 상태 확인

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_NAME="gdrive"
REMOTE_PATH="Sullivan_Dataset"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Google Drive Sync (rclone)${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# Setup rclone with Google Drive
cmd_setup() {
    echo -e "${YELLOW}🔧 Setting up rclone with Google Drive...${NC}"
    echo ""
    echo "다음 단계를 따라 설정하세요:"
    echo "1. 'n' 입력 (새 remote 생성)"
    echo "2. 이름: gdrive"
    echo "3. Storage type: Google Drive (숫자 선택)"
    echo "4. client_id, client_secret: 엔터 (기본값 사용)"
    echo "5. scope: 1 (Full access)"
    echo "6. root_folder_id: 엔터 (기본값)"
    echo "7. service_account_file: 엔터"
    echo "8. Auto config: y (브라우저에서 인증)"
    echo "9. Team Drive: n"
    echo "10. 확인 후 'q' 로 종료"
    echo ""
    read -p "설정을 시작하시겠습니까? (y/n): " confirm
    if [[ "$confirm" == "y" ]]; then
        rclone config
    fi
}

# Check connection status
cmd_status() {
    echo -e "${YELLOW}📊 Checking Google Drive connection...${NC}"
    
    if rclone listremotes | grep -q "^${REMOTE_NAME}:"; then
        echo -e "${GREEN}✅ Remote '${REMOTE_NAME}' is configured${NC}"
        
        # Check if Sullivan_Dataset exists
        if rclone lsd "${REMOTE_NAME}:" 2>/dev/null | grep -q "${REMOTE_PATH}"; then
            echo -e "${GREEN}✅ Found ${REMOTE_PATH} folder${NC}"
            echo ""
            echo "폴더 내용:"
            rclone lsd "${REMOTE_NAME}:${REMOTE_PATH}" 2>/dev/null || true
        else
            echo -e "${YELLOW}⚠️ ${REMOTE_PATH} folder not found. Please create it.${NC}"
        fi
    else
        echo -e "${RED}❌ Remote '${REMOTE_NAME}' not configured${NC}"
        echo "Run: $0 setup"
    fi
}

# List files in Drive
cmd_list() {
    local path="${1:-$REMOTE_PATH}"
    echo -e "${YELLOW}📁 Listing: ${REMOTE_NAME}:${path}${NC}"
    rclone ls "${REMOTE_NAME}:${path}" --max-depth 2 | head -50
}

# Push local files to Drive
cmd_push() {
    local local_path="${1:-$PROJECT_ROOT/data/processed}"
    local remote_path="${2:-$REMOTE_PATH}"
    
    echo -e "${YELLOW}📤 Uploading to Google Drive...${NC}"
    echo "   From: $local_path"
    echo "   To:   ${REMOTE_NAME}:${remote_path}"
    echo ""
    
    rclone sync "$local_path" "${REMOTE_NAME}:${remote_path}" \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --exclude "*.tmp" \
        --exclude ".DS_Store"
    
    echo -e "${GREEN}✅ Upload complete!${NC}"
}

# Pull files from Drive to local
cmd_pull() {
    local remote_path="${1:-$REMOTE_PATH}"
    local local_path="${2:-$PROJECT_ROOT/data/gdrive_sync}"
    
    echo -e "${YELLOW}📥 Downloading from Google Drive...${NC}"
    echo "   From: ${REMOTE_NAME}:${remote_path}"
    echo "   To:   $local_path"
    echo ""
    
    mkdir -p "$local_path"
    rclone sync "${REMOTE_NAME}:${remote_path}" "$local_path" \
        --progress \
        --transfers 4 \
        --checkers 8
    
    echo -e "${GREEN}✅ Download complete!${NC}"
}

# Download checkpoints only
cmd_checkpoints() {
    local remote_path="Sullivan_Checkpoints"
    local local_path="$PROJECT_ROOT/models/colab_checkpoints"
    
    echo -e "${YELLOW}💾 Downloading checkpoints from Google Drive...${NC}"
    
    mkdir -p "$local_path"
    rclone sync "${REMOTE_NAME}:${remote_path}" "$local_path" \
        --progress \
        --include "*.ckpt"
    
    echo -e "${GREEN}✅ Checkpoints downloaded to: $local_path${NC}"
    ls -lh "$local_path"
}

# Show help
cmd_help() {
    print_header
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup              - rclone 초기 설정 (Google Drive 연결)"
    echo "  status             - 연결 상태 확인"
    echo "  list [path]        - Drive 파일 목록"
    echo "  push [local] [remote] - 로컬 → Drive 업로드"
    echo "  pull [remote] [local] - Drive → 로컬 다운로드"
    echo "  checkpoints        - 체크포인트만 다운로드"
    echo "  help               - 도움말"
    echo ""
    echo "Examples:"
    echo "  $0 setup                           # 초기 설정"
    echo "  $0 status                          # 상태 확인"
    echo "  $0 list                            # Sullivan_Dataset 목록"
    echo "  $0 push ./data/processed           # 데이터 업로드"
    echo "  $0 checkpoints                     # 체크포인트 다운로드"
}

# Main
print_header

case "${1:-help}" in
    setup)       cmd_setup ;;
    status)      cmd_status ;;
    list)        cmd_list "${@:2}" ;;
    push)        cmd_push "${@:2}" ;;
    pull)        cmd_pull "${@:2}" ;;
    checkpoints) cmd_checkpoints ;;
    help)        cmd_help ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
