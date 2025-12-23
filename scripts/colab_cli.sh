#!/bin/bash
# ============================================
# Project Sullivan - Colab CLI Management Tool
# ============================================
# 로컬에서 Colab 학습을 관리하기 위한 CLI 스크립트
# Usage:
#   ./scripts/colab_cli.sh push     - 코드 푸시
#   ./scripts/colab_cli.sh status   - 학습 상태 확인
#   ./scripts/colab_cli.sh logs     - 최근 로그 확인

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GITHUB_REPO="faransansj/Project_Sullivan"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Project Sullivan - Colab CLI${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# Push code to GitHub for Colab to pull
cmd_push() {
    echo -e "${YELLOW}📤 Pushing code to GitHub...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Check for uncommitted changes
    if [[ -n $(git status -s) ]]; then
        echo -e "${YELLOW}Uncommitted changes detected. Staging all changes...${NC}"
        git add -A
        
        # Get commit message
        if [[ -n "$1" ]]; then
            COMMIT_MSG="$1"
        else
            COMMIT_MSG="[Colab Update] $(date '+%Y-%m-%d %H:%M:%S')"
        fi
        
        git commit -m "$COMMIT_MSG"
    fi
    
    git push origin main
    
    echo -e "${GREEN}✅ Code pushed successfully!${NC}"
    echo -e "${BLUE}Colab에서 다음을 실행하세요:${NC}"
    echo -e "  !git pull origin main"
}

# Check training status (placeholder - requires Google Drive API)
cmd_status() {
    echo -e "${YELLOW}📊 Checking training status...${NC}"
    
    # Check local logs if synced
    LOCAL_LOGS="$PROJECT_ROOT/results/colab_runs"
    
    if [[ -d "$LOCAL_LOGS" ]]; then
        echo -e "${GREEN}Found local sync of Colab logs:${NC}"
        ls -lt "$LOCAL_LOGS" | head -5
    else
        echo -e "${YELLOW}No local logs found.${NC}"
        echo -e "Google Drive에서 체크포인트를 확인하세요:"
        echo -e "  MyDrive/Sullivan_Checkpoints/"
    fi
    
    echo ""
    echo -e "${BLUE}💡 Tips:${NC}"
    echo -e "  - Colab에서 TensorBoard로 실시간 모니터링"
    echo -e "  - Google Drive에서 체크포인트 확인"
}

# View recent logs
cmd_logs() {
    echo -e "${YELLOW}📋 Viewing recent training logs...${NC}"
    
    LOCAL_LOGS="$PROJECT_ROOT/logs/training"
    
    if [[ -d "$LOCAL_LOGS" ]]; then
        LATEST=$(ls -td "$LOCAL_LOGS"/*/ 2>/dev/null | head -1)
        if [[ -n "$LATEST" && -f "$LATEST/metrics.csv" ]]; then
            echo -e "${GREEN}Latest metrics from: $LATEST${NC}"
            tail -20 "$LATEST/metrics.csv"
        else
            echo -e "${YELLOW}No metrics.csv found in latest log directory.${NC}"
        fi
    else
        echo -e "${YELLOW}No local training logs found.${NC}"
    fi
}

# Create data shard for incremental training
cmd_shard() {
    echo -e "${YELLOW}📦 Creating data shards...${NC}"
    
    if [[ -z "$1" ]]; then
        echo -e "${RED}Usage: $0 shard <source_dir> <num_shards>${NC}"
        exit 1
    fi
    
    python "$PROJECT_ROOT/scripts/shard_dataset.py" "$@"
}

# Show help
cmd_help() {
    print_header
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  push [message]   - Push code to GitHub"
    echo "  status           - Check training status"
    echo "  logs             - View recent training logs"
    echo "  shard <dir> <n>  - Create n data shards from directory"
    echo "  help             - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 push \"Fix learning rate\""
    echo "  $0 status"
    echo "  $0 shard /path/to/data 10"
}

# Main
print_header

case "${1:-help}" in
    push)   cmd_push "${@:2}" ;;
    status) cmd_status ;;
    logs)   cmd_logs ;;
    shard)  cmd_shard "${@:2}" ;;
    help)   cmd_help ;;
    *)      
        echo -e "${RED}Unknown command: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
