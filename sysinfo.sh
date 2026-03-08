#!/bin/bash
# ============================================================
#  系统配置信息采集脚本 - 用于论文实验环境描述
#  使用方法: sudo bash get_sysinfo.sh
# ============================================================

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "============================================================"
echo "         系统配置信息采集脚本"
echo "         $(date)"
echo "         主机: $(hostname)"
echo "============================================================"
echo -e "${NC}"

# ─────────────────────────────────────────────
# 1. 操作系统信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[1] 操作系统${NC}"
echo "  OS         : $(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "  内核版本   : $(uname -r)"
echo "  架构       : $(uname -m)"
echo ""

# ─────────────────────────────────────────────
# 2. CPU 信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[2] CPU${NC}"
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)
CPU_PHYSICAL=$(grep "physical id" /proc/cpuinfo | sort -u | wc -l)
CPU_CORES=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
CPU_THREADS=$(grep -c "processor" /proc/cpuinfo)
CPU_FREQ=$(grep -m1 "cpu MHz" /proc/cpuinfo | cut -d: -f2 | xargs)
CPU_CACHE=$(grep -m1 "cache size" /proc/cpuinfo | cut -d: -f2 | xargs)

echo "  型号       : $CPU_MODEL"
echo "  物理CPU数  : $CPU_PHYSICAL"
echo "  每CPU核心数: $CPU_CORES"
echo "  总线程数   : $CPU_THREADS"
echo "  主频       : ${CPU_FREQ} MHz"
echo "  缓存       : $CPU_CACHE"
echo ""

# ─────────────────────────────────────────────
# 3. 内存信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[3] 内存${NC}"
MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
MEM_FREE=$(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
SWAP_TOTAL=$(grep SwapTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')

echo "  总内存     : $MEM_TOTAL"
echo "  可用内存   : $MEM_FREE"
echo "  Swap       : $SWAP_TOTAL"

# 尝试获取内存条详细信息
if command -v dmidecode &>/dev/null; then
    echo "  内存条详情:"
    sudo dmidecode -t memory 2>/dev/null | grep -E "Size:|Type:|Speed:|Manufacturer:|Part Number:" | grep -v "No Module" | grep -v "Unknown" | sed 's/^/    /'
fi
echo ""

# ─────────────────────────────────────────────
# 4. GPU 信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[4] GPU${NC}"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r name driver mem_total mem_free cap; do
        echo "  型号       :$name"
        echo "  驱动版本   :$driver"
        echo "  显存总量   :$mem_total"
        echo "  显存可用   :$mem_free"
        echo "  计算能力   :$cap"
    done
    echo ""
    echo "  完整 GPU 列表:"
    nvidia-smi -L 2>/dev/null | sed 's/^/    /'
else
    echo "  未检测到 NVIDIA GPU 或 nvidia-smi 不可用"
    # 尝试 lspci
    if command -v lspci &>/dev/null; then
        echo "  lspci 检测到的显卡:"
        lspci | grep -iE "vga|display|3d|gpu" | sed 's/^/    /'
    fi
fi
echo ""

# ─────────────────────────────────────────────
# 5. 存储信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[5] 存储${NC}"
echo "  磁盘挂载情况:"
df -h --output=source,size,used,avail,target 2>/dev/null | grep -v tmpfs | grep -v udev | sed 's/^/    /'
echo ""
if command -v lsblk &>/dev/null; then
    echo "  块设备列表:"
    lsblk -d -o NAME,SIZE,TYPE,MODEL,ROTA 2>/dev/null | sed 's/^/    /'
fi
echo ""

# ─────────────────────────────────────────────
# 6. TEE / 安全硬件信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[6] TEE / 安全硬件${NC}"

# Intel SGX
SGX_FOUND=0
echo "  --- Intel SGX ---"
if ls /dev/sgx* 2>/dev/null | grep -q sgx; then
    echo "  SGX 设备    : $(ls /dev/sgx* 2>/dev/null)"
    SGX_FOUND=1
fi
if ls /dev/isgx 2>/dev/null &>/dev/null; then
    echo "  SGX 设备    : /dev/isgx (旧版驱动)"
    SGX_FOUND=1
fi
SGX_CPU=$(grep -m1 "flags" /proc/cpuinfo | grep -o "sgx[^ ]*" | tr '\n' ' ')
if [ -n "$SGX_CPU" ]; then
    echo "  CPU SGX标志 : $SGX_CPU"
    SGX_FOUND=1
fi
if command -v sgx-detect &>/dev/null; then
    echo "  sgx-detect 输出:"
    sgx-detect 2>/dev/null | sed 's/^/    /'
elif command -v sgxdetect &>/dev/null; then
    sgxdetect 2>/dev/null | sed 's/^/    /'
fi
# 通过 cpuid 检测 SGX
if command -v cpuid &>/dev/null; then
    SGX_CPUID=$(cpuid -1 2>/dev/null | grep -i sgx | head -5)
    [ -n "$SGX_CPUID" ] && echo "  cpuid SGX信息: $SGX_CPUID"
fi
[ $SGX_FOUND -eq 0 ] && echo "  未检测到 SGX 设备"

# Intel TDX
echo ""
echo "  --- Intel TDX ---"
if ls /dev/tdx* 2>/dev/null | grep -q tdx; then
    echo "  TDX 设备    : $(ls /dev/tdx* 2>/dev/null)"
else
    echo "  未检测到 TDX 设备"
fi

# AMD SEV
echo ""
echo "  --- AMD SEV ---"
if ls /dev/sev* 2>/dev/null | grep -q sev; then
    echo "  SEV 设备    : $(ls /dev/sev* 2>/dev/null)"
    if command -v sevctl &>/dev/null; then
        sevctl show 2>/dev/null | sed 's/^/    /'
    fi
else
    echo "  未检测到 AMD SEV 设备"
fi

# TPM
echo ""
echo "  --- TPM ---"
if ls /dev/tpm* 2>/dev/null | grep -q tpm; then
    echo "  TPM 设备    : $(ls /dev/tpm* 2>/dev/null)"
    if command -v tpm2_getcap &>/dev/null; then
        echo "  TPM 版本    : $(tpm2_getcap properties-fixed 2>/dev/null | grep -i "tpmspecification\|version" | head -3 | sed 's/^/    /')"
    fi
else
    echo "  未检测到 TPM 设备"
fi

# BIOS/UEFI 中的安全特性
echo ""
echo "  --- BIOS/UEFI 安全特性 ---"
if command -v dmidecode &>/dev/null; then
    sudo dmidecode -t 0 2>/dev/null | grep -E "Vendor|Version|Release Date|Characteristics" | sed 's/^/  /'
fi
echo ""

# ─────────────────────────────────────────────
# 7. 网络信息
# ─────────────────────────────────────────────
echo -e "${CYAN}[7] 网络${NC}"
echo "  网络接口:"
ip -br addr show 2>/dev/null | grep -v "^lo" | sed 's/^/    /'
echo ""

# ─────────────────────────────────────────────
# 8. Python / 深度学习环境
# ─────────────────────────────────────────────
echo -e "${CYAN}[8] Python / 深度学习环境${NC}"
echo "  Python 版本: $(python3 --version 2>/dev/null || echo '未安装')"
echo "  pip 版本   : $(pip3 --version 2>/dev/null | cut -d' ' -f1-2 || echo '未安装')"
echo ""

# 检查常用深度学习库版本
echo "  主要库版本:"
for pkg in torch torchvision tensorflow numpy scipy scikit-learn cryptography; do
    ver=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
    if [ -n "$ver" ]; then
        printf "    %-20s: %s\n" "$pkg" "$ver"
    fi
done

# PyTorch CUDA 支持
CUDA_AVAIL=$(python3 -c "import torch; print('可用, 版本:', torch.version.cuda, '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无CUDA')" 2>/dev/null)
[ -n "$CUDA_AVAIL" ] && echo "  CUDA(PyTorch): $CUDA_AVAIL"
echo ""

# ─────────────────────────────────────────────
# 9. 论文实验环境摘要（英文）
# ─────────────────────────────────────────────
echo -e "${CYAN}[9] 论文实验环境描述摘要 (English)${NC}"
echo "  ----------------------------------------"
OS_STR=$(lsb_release -d 2>/dev/null | cut -f2)
KERNEL_STR=$(uname -r)
echo "  OS      : $OS_STR (Kernel $KERNEL_STR)"
echo "  CPU     : $CPU_MODEL"
echo "            $CPU_PHYSICAL socket(s) x $CPU_CORES cores, $CPU_THREADS threads total"
echo "  RAM     : $MEM_TOTAL"
if command -v nvidia-smi &>/dev/null; then
    GPU_STR=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU     : $GPU_STR"
fi
echo "  ----------------------------------------"
echo ""

echo -e "${GREEN}============================================================"
echo "  采集完成！"
echo "============================================================${NC}"
