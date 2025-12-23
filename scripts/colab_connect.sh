#!/bin/bash
# ============================================
# Colab SSH Connection Script
# ============================================
# Usage:
#   ./scripts/colab_connect.sh          - SSH 연결 (기본)
#   ./scripts/colab_connect.sh info     - 연결 정보 표시
#   ./scripts/colab_connect.sh tunnel   - 터널 직접 설정

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Config file for storing connection info
CONFIG_FILE="$PROJECT_ROOT/.colab_ssh_config"

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Colab SSH Connection${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# Show connection info/instructions
cmd_info() {
    echo -e "${CYAN}📋 Colab SSH 연결 방법${NC}"
    echo ""
    echo "1️⃣  Colab 노트북에서 다음 셀을 실행하세요:"
    echo ""
    echo -e "${GREEN}# Colab SSH Setup Cell${NC}"
    echo "!pip install colab-ssh --quiet"
    echo "from colab_ssh import launch_ssh_cloudflared"
    echo 'launch_ssh_cloudflared(password="sullivan2025")'
    echo ""
    echo "2️⃣  출력된 연결 정보를 아래에 입력하세요:"
    echo ""
    
    if [[ -f "$CONFIG_FILE" ]]; then
        echo -e "${YELLOW}저장된 연결 정보:${NC}"
        cat "$CONFIG_FILE"
        echo ""
    fi
}

# Save connection info
cmd_save() {
    echo -e "${YELLOW}연결 정보를 입력하세요 (Colab 출력에서 복사):${NC}"
    echo ""
    read -p "Hostname (예: abc-xyz.trycloudflare.com): " hostname
    read -p "Port (기본값: 22): " port
    port=${port:-22}
    
    echo "HOSTNAME=$hostname" > "$CONFIG_FILE"
    echo "PORT=$port" >> "$CONFIG_FILE"
    echo "USER=root" >> "$CONFIG_FILE"
    echo "PASSWORD=sullivan2025" >> "$CONFIG_FILE"
    
    echo -e "${GREEN}✅ 연결 정보가 저장되었습니다.${NC}"
}

# Connect via SSH
cmd_connect() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}❌ 연결 정보가 없습니다.${NC}"
        echo "먼저 Colab에서 SSH를 설정하고 연결 정보를 저장하세요:"
        echo "  $0 save"
        exit 1
    fi
    
    source "$CONFIG_FILE"
    
    echo -e "${YELLOW}🔗 Connecting to Colab...${NC}"
    echo "   Host: $HOSTNAME"
    echo "   User: $USER"
    echo ""
    echo -e "${CYAN}Password: $PASSWORD${NC}"
    echo ""
    
    # SSH via cloudflared
    ssh -o ProxyCommand="cloudflared access ssh --hostname %h" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${USER}@${HOSTNAME}"
}

# Run a command on Colab
cmd_run() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}❌ 연결 정보가 없습니다.${NC}"
        exit 1
    fi
    
    source "$CONFIG_FILE"
    local cmd="$*"
    
    echo -e "${YELLOW}🚀 Running on Colab: $cmd${NC}"
    
    ssh -o ProxyCommand="cloudflared access ssh --hostname %h" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${USER}@${HOSTNAME}" "$cmd"
}

# Check training status on Colab
cmd_status() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${YELLOW}⚠️ 연결 정보 없음. Colab에서 SSH 설정 필요.${NC}"
        cmd_info
        return
    fi
    
    source "$CONFIG_FILE"
    
    echo -e "${YELLOW}📊 Checking training status on Colab...${NC}"
    
    # Try to get GPU status and training logs
    ssh -o ProxyCommand="cloudflared access ssh --hostname %h" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=10 \
        "${USER}@${HOSTNAME}" "nvidia-smi && echo '---' && tail -20 /content/Project_Sullivan/logs/training/*/metrics.csv 2>/dev/null || echo 'No training logs found'" 2>/dev/null || {
        echo -e "${RED}❌ Cannot connect to Colab. Session may have ended.${NC}"
    }
}

# Show help
cmd_help() {
    print_header
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  info              - 연결 방법 안내"
    echo "  save              - 연결 정보 저장"
    echo "  connect           - SSH 연결"
    echo "  run <command>     - Colab에서 명령 실행"
    echo "  status            - 학습 상태 확인"
    echo "  help              - 도움말"
    echo ""
    echo "Examples:"
    echo "  $0 info                            # 설정 방법 보기"
    echo "  $0 save                            # 연결 정보 저장"
    echo "  $0 connect                         # SSH 접속"
    echo "  $0 run 'nvidia-smi'                # GPU 상태 확인"
    echo "  $0 status                          # 학습 상태 확인"
}

# Main
print_header

case "${1:-connect}" in
    info)    cmd_info ;;
    save)    cmd_save ;;
    connect) cmd_connect ;;
    run)     cmd_run "${@:2}" ;;
    status)  cmd_status ;;
    help)    cmd_help ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
