#!/bin/bash
# ============================================================
#  安全检测脚本 - 检查是否存在类似 HUsBp32V 的入侵痕迹
#  使用方法: sudo bash check_security.sh
# ============================================================

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

WARN=0

echo -e "${CYAN}"
echo "============================================================"
echo "         安全检测脚本 v1.0"
echo "         $(date)"
echo "         主机: $(hostname)"
echo "============================================================"
echo -e "${NC}"

# ─────────────────────────────────────────────
# 1. 检查 cron.d 中的可疑条目
# ─────────────────────────────────────────────
echo -e "${CYAN}[1/8] 检查 /etc/cron.d 可疑条目...${NC}"
SUSPICIOUS_CRON=()
for f in /etc/cron.d/*; do
    # 跳过已知正常文件
    if [[ "$f" =~ (anacron|e2scrub|sysstat|popularity|php|apt) ]]; then
        continue
    fi
    # 随机命名特征：文件名包含大小写混合或随机字符
    basename_f=$(basename "$f")
    if echo "$basename_f" | grep -qE '^[a-zA-Z0-9]{6,12}$'; then
        SUSPICIOUS_CRON+=("$f")
        echo -e "  ${RED}[!] 可疑 cron 文件: $f${NC}"
        cat "$f"
        WARN=$((WARN+1))
    fi
done
if [ ${#SUSPICIOUS_CRON[@]} -eq 0 ]; then
    echo -e "  ${GREEN}[OK] 未发现可疑 cron.d 条目${NC}"
fi

# ─────────────────────────────────────────────
# 2. 检查 root crontab
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[2/8] 检查 root crontab...${NC}"
ROOT_CRON=$(crontab -u root -l 2>/dev/null)
if [ -n "$ROOT_CRON" ]; then
    echo -e "  ${RED}[!] root 存在 crontab:${NC}"
    echo "$ROOT_CRON"
    WARN=$((WARN+1))
else
    echo -e "  ${GREEN}[OK] root 无 crontab${NC}"
fi

# ─────────────────────────────────────────────
# 3. 检查 ld.so.preload
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[3/8] 检查 /etc/ld.so.preload...${NC}"
if [ -s /etc/ld.so.preload ]; then
    echo -e "  ${RED}[!] ld.so.preload 非空，存在进程劫持风险:${NC}"
    cat /etc/ld.so.preload
    WARN=$((WARN+1))
else
    echo -e "  ${GREEN}[OK] ld.so.preload 为空或不存在${NC}"
fi

# ─────────────────────────────────────────────
# 4. 检查可疑 systemd 服务
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[4/8] 检查可疑 systemd 服务...${NC}"
SUSPICIOUS_SVC=()
for f in /etc/systemd/system/*.service; do
    basename_f=$(basename "$f" .service)
    # 随机命名特征
    if echo "$basename_f" | grep -qE '^[a-zA-Z0-9]{6,12}$'; then
        SUSPICIOUS_SVC+=("$f")
        echo -e "  ${RED}[!] 可疑服务: $f${NC}"
        cat "$f"
        WARN=$((WARN+1))
    fi
done
if [ ${#SUSPICIOUS_SVC[@]} -eq 0 ]; then
    echo -e "  ${GREEN}[OK] 未发现可疑 systemd 服务${NC}"
fi

# ─────────────────────────────────────────────
# 5. 检查 /bin /usr/bin 中的可疑文件
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[5/8] 检查 /bin /usr/bin 中的可疑文件...${NC}"
SUSPICIOUS_BIN=()
# 查找随机命名的可执行文件（大小写混合，6-12位）
while IFS= read -r f; do
    fname=$(basename "$f")
    if echo "$fname" | grep -qE '^[a-zA-Z]{6,12}$' && \
       echo "$fname" | grep -qE '[A-Z]' && \
       echo "$fname" | grep -qE '[a-z]'; then
        SUSPICIOUS_BIN+=("$f")
        echo -e "  ${RED}[!] 可疑文件: $f$(ls -la $f)${NC}"
        WARN=$((WARN+1))
    fi
done < <(find /bin /usr/bin -maxdepth 1 -type f -executable 2>/dev/null)

# 检查隐藏文件
while IFS= read -r f; do
    echo -e "  ${RED}[!] 发现隐藏可执行文件: $f${NC}"
    ls -la "$f"
    WARN=$((WARN+1))
done < <(find /bin /usr/bin /usr/local/bin -maxdepth 1 -name ".*" -type f 2>/dev/null)

if [ ${#SUSPICIOUS_BIN[@]} -eq 0 ] && [ $WARN -eq 0 ]; then
    echo -e "  ${GREEN}[OK] 未发现可疑二进制文件${NC}"
fi

# ─────────────────────────────────────────────
# 6. 检查 SSH authorized_keys
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[6/8] 检查 SSH authorized_keys...${NC}"
for home_dir in /root /home/*; do
    auth_file="$home_dir/.ssh/authorized_keys"
    if [ -f "$auth_file" ]; then
        count=$(wc -l < "$auth_file")
        echo -e "  ${YELLOW}[?] $auth_file ($count 个密钥):${NC}"
        cat "$auth_file"
        echo ""
    fi
done

# ─────────────────────────────────────────────
# 7. 检查可疑网络连接
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[7/8] 检查可疑外连进程...${NC}"
echo "当前 ESTABLISHED 连接:"
ss -antp | grep ESTAB | grep -v -E "(127\.0\.0|::1)" | while read line; do
    echo -e "  ${YELLOW}$line${NC}"
done

# 检查是否有挖矿相关进程
MINING=$(ps aux | grep -iE "xmr|miner|stratum|monero|crypto" | grep -v grep)
if [ -n "$MINING" ]; then
    echo -e "  ${RED}[!] 发现疑似挖矿进程:${NC}"
    echo "$MINING"
    WARN=$((WARN+1))
else
    echo -e "  ${GREEN}[OK] 未发现疑似挖矿进程${NC}"
fi

# ─────────────────────────────────────────────
# 8. 检查 OOM / killed 日志
# ─────────────────────────────────────────────
echo -e "\n${CYAN}[8/8] 检查进程异常 killed 日志...${NC}"
KILLED=$(dmesg 2>/dev/null | grep -i "killed process" | tail -5)
if [ -n "$KILLED" ]; then
    echo -e "  ${YELLOW}[?] 发现进程被杀记录:${NC}"
    echo "$KILLED"
else
    echo -e "  ${GREEN}[OK] 无异常 killed 记录${NC}"
fi

# ─────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────
echo ""
echo "============================================================"
if [ $WARN -gt 0 ]; then
    echo -e "${RED}  ⚠️  检测完成：发现 $WARN 处可疑项，请立即处理！${NC}"
else
    echo -e "${GREEN}  ✅  检测完成：未发现明显入侵痕迹${NC}"
fi
echo "============================================================"