#!/bin/bash

# System Report Generator
# Detects and displays GPU, CPU, Memory, and other system information
# Usage: ./system_report.sh [--tight]

# Parse command line arguments
TIGHT_MODE=false
if [ "$1" = "--tight" ]; then
    TIGHT_MODE=true
fi

# Color codes for pretty printing
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Box drawing characters
HORIZONTAL="━"
VERTICAL="┃"
TOP_LEFT="┏"
TOP_RIGHT="┓"
BOTTOM_LEFT="┗"
BOTTOM_RIGHT="┛"
T_RIGHT="┣"
T_LEFT="┫"
CROSS="╋"

# Function to print a horizontal line
print_line() {
    local width=${1:-80}
    printf "${CYAN}${TOP_LEFT}"
    printf "${HORIZONTAL}%.0s" $(seq 1 $((width-2)))
    printf "${TOP_RIGHT}${NC}\n"
}

print_separator() {
    local width=${1:-80}
    printf "${CYAN}${T_RIGHT}"
    printf "${HORIZONTAL}%.0s" $(seq 1 $((width-2)))
    printf "${T_LEFT}${NC}\n"
}

print_bottom() {
    local width=${1:-80}
    printf "${CYAN}${BOTTOM_LEFT}"
    printf "${HORIZONTAL}%.0s" $(seq 1 $((width-2)))
    printf "${BOTTOM_RIGHT}${NC}\n"
}

# Function to print a section header
print_header() {
    local text="$1"
    local width=80
    printf "${CYAN}${VERTICAL}${NC} ${BOLD}${WHITE}%-$((width-4))s${NC} ${CYAN}${VERTICAL}${NC}\n" "$text"
}

# Function to print a key-value pair
print_info() {
    local key="$1"
    local value="$2"
    local width=80
    local key_width=25
    printf "${CYAN}${VERTICAL}${NC} ${GREEN}%-${key_width}s${NC} ${YELLOW}%-$((width-key_width-5))s${NC} ${CYAN}${VERTICAL}${NC}\n" "$key:" "$value"
}

# Function to print plain text
print_text() {
    local text="$1"
    local width=80
    printf "${CYAN}${VERTICAL}${NC} %-$((width-4))s ${CYAN}${VERTICAL}${NC}\n" "$text"
}

# Clear screen and print title
clear
echo ""
print_line 80
print_header "SYSTEM REPORT"
print_header "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
print_separator 80

# ============================================================================
# OPERATING SYSTEM
# ============================================================================
print_header "OPERATING SYSTEM"
print_separator 80

if [ -f /etc/os-release ]; then
    . /etc/os-release
    print_info "Distribution" "$NAME"
    if [ "$TIGHT_MODE" = false ]; then
        print_info "Version" "$VERSION"
        print_info "Kernel" "$(uname -r)"
    else
        print_info "Version" "$VERSION"
    fi
else
    print_info "OS" "$(uname -s)"
    if [ "$TIGHT_MODE" = false ]; then
        print_info "Kernel" "$(uname -r)"
    fi
fi

if [ "$TIGHT_MODE" = false ]; then
    print_info "Hostname" "$(hostname)"
    print_info "Uptime" "$(uptime -p 2>/dev/null || uptime | awk -F'up ' '{print $2}' | awk -F',' '{print $1}')"
fi

print_separator 80

# ============================================================================
# CPU INFORMATION
# ============================================================================
print_header "CPU INFORMATION"
print_separator 80

if [ -f /proc/cpuinfo ]; then
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)
    CPU_THREADS=$(nproc)
    
    print_info "Model" "$CPU_MODEL"
    print_info "Cores" "$CPU_CORES"
    
    if [ "$TIGHT_MODE" = false ]; then
        print_info "Threads" "$CPU_THREADS"
        
        # CPU frequency
        if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]; then
            FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
            FREQ_GHZ=$(awk "BEGIN {printf \"%.2f\", $FREQ/1000000}")
            print_info "Current Frequency" "${FREQ_GHZ} GHz"
        fi
        
        # CPU usage
        if command -v mpstat &> /dev/null; then
            CPU_USAGE=$(mpstat 1 1 | awk '/Average/ {print 100-$NF"%"}')
            print_info "CPU Usage" "$CPU_USAGE"
        fi
    fi
fi

print_separator 80

# ============================================================================
# MEMORY INFORMATION
# ============================================================================
print_header "MEMORY INFORMATION"
print_separator 80

if [ -f /proc/meminfo ]; then
    MEM_TOTAL=$(grep "MemTotal" /proc/meminfo | awk '{printf "%.2f GB", $2/1024/1024}')
    
    print_info "Total Memory" "$MEM_TOTAL"
    
    if [ "$TIGHT_MODE" = false ]; then
        MEM_AVAILABLE=$(grep "MemAvailable" /proc/meminfo | awk '{printf "%.2f GB", $2/1024/1024}')
        MEM_FREE=$(grep "MemFree" /proc/meminfo | awk '{printf "%.2f GB", $2/1024/1024}')
        MEM_USED=$(awk '/MemTotal/ {total=$2} /MemAvailable/ {avail=$2} END {printf "%.2f GB", (total-avail)/1024/1024}' /proc/meminfo)
        MEM_PERCENT=$(awk '/MemTotal/ {total=$2} /MemAvailable/ {avail=$2} END {printf "%.1f%%", (total-avail)/total*100}' /proc/meminfo)
        
        print_info "Used Memory" "$MEM_USED ($MEM_PERCENT)"
        print_info "Available Memory" "$MEM_AVAILABLE"
        print_info "Free Memory" "$MEM_FREE"
        
        # Swap information
        SWAP_TOTAL=$(grep "SwapTotal" /proc/meminfo | awk '{printf "%.2f GB", $2/1024/1024}')
        SWAP_FREE=$(grep "SwapFree" /proc/meminfo | awk '{printf "%.2f GB", $2/1024/1024}')
        print_info "Swap Total" "$SWAP_TOTAL"
        print_info "Swap Free" "$SWAP_FREE"
    fi
fi

print_separator 80

# ============================================================================
# GPU INFORMATION
# ============================================================================
print_header "GPU INFORMATION"
print_separator 80

# Check for NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    
    if [ "$TIGHT_MODE" = true ]; then
        # Tight mode: only show name, driver, and memory for first GPU
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
        GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)
        GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0)
        
        print_info "NVIDIA GPU" "$GPU_NAME"
        print_info "Driver" "$GPU_DRIVER"
        print_info "Memory" "$GPU_MEM_TOTAL"
    else
        # Full mode
        print_info "NVIDIA GPUs Detected" "$GPU_COUNT"
        print_text ""
        
        # Get detailed info for each GPU
        for i in $(seq 0 $((GPU_COUNT-1))); do
            print_text "  GPU $i:"
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $i)
            GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i $i)
            GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i $i)
            GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $i)
            GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $i)
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i $i)
            GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i $i)
            GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader -i $i)
            
            print_text "    Name: $GPU_NAME"
            print_text "    Driver: $GPU_DRIVER"
            print_text "    Memory: $GPU_MEM_USED / $GPU_MEM_TOTAL (Free: $GPU_MEM_FREE)"
            print_text "    Utilization: $GPU_UTIL"
            print_text "    Temperature: ${GPU_TEMP}°C"
            print_text "    Power Draw: $GPU_POWER"
            
            if [ $i -lt $((GPU_COUNT-1)) ]; then
                print_text ""
            fi
        done
        
        # CUDA version
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
            print_text ""
            print_info "CUDA Version" "$CUDA_VERSION"
        fi
    fi
else
    print_text "No NVIDIA GPUs detected or nvidia-smi not available"
fi

if [ "$TIGHT_MODE" = false ]; then
    # Check for AMD GPUs
    if command -v rocm-smi &> /dev/null; then
        print_text ""
        print_info "AMD ROCm GPUs" "Detected"
        rocm-smi --showproductname 2>/dev/null | grep -v "=" | while read line; do
            [ ! -z "$line" ] && print_text "  $line"
        done
    elif lspci 2>/dev/null | grep -i "amd.*vga\|amd.*display" &> /dev/null; then
        print_text ""
        print_text "AMD GPU detected (rocm-smi not available for details)"
        lspci | grep -i "amd.*vga\|amd.*display" | while read line; do
            print_text "  $line"
        done
    fi

    # Check for Intel GPUs
    if lspci 2>/dev/null | grep -i "intel.*vga\|intel.*display" &> /dev/null; then
        print_text ""
        print_text "Intel GPU detected:"
        lspci | grep -i "intel.*vga\|intel.*display" | while read line; do
            print_text "  $line"
        done
    fi
fi

print_separator 80

# ============================================================================
# PYTHON ENVIRONMENT
# ============================================================================
print_header "PYTHON ENVIRONMENT"
print_separator 80

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python Version" "$PYTHON_VERSION"
    
    if [ "$TIGHT_MODE" = false ]; then
        if command -v pip3 &> /dev/null; then
            PIP_VERSION=$(pip3 --version | awk '{print $2}')
            print_info "Pip Version" "$PIP_VERSION"
        fi
    fi
    
    # Check for PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null)
        
        print_info "PyTorch Version" "$TORCH_VERSION"
        
        if [ "$TIGHT_MODE" = true ]; then
            # Tight mode: only show CUDA version
            print_info "CUDA Version" "$TORCH_CUDA_VERSION"
        else
            # Full mode
            TORCH_CUDA=$(python3 -c "import torch; print('Available' if torch.cuda.is_available() else 'Not Available')" 2>/dev/null)
            print_info "PyTorch CUDA" "$TORCH_CUDA"
            if [ "$TORCH_CUDA" = "Available" ]; then
                print_info "PyTorch CUDA Version" "$TORCH_CUDA_VERSION"
                TORCH_GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
                print_info "PyTorch GPU Count" "$TORCH_GPU_COUNT"
            fi
        fi
    fi
else
    print_text "Python not found"
fi

if [ "$TIGHT_MODE" = false ]; then
    print_separator 80

    # ============================================================================
    # DISK INFORMATION
    # ============================================================================
    print_header "DISK INFORMATION"
    print_separator 80

    if command -v df &> /dev/null; then
        print_text "Filesystem Usage:"
        print_text ""
        df -h | grep -E "^/dev/" | while read line; do
            DEVICE=$(echo $line | awk '{print $1}')
            SIZE=$(echo $line | awk '{print $2}')
            USED=$(echo $line | awk '{print $3}')
            AVAIL=$(echo $line | awk '{print $4}')
            USE_PERCENT=$(echo $line | awk '{print $5}')
            MOUNT=$(echo $line | awk '{print $6}')
            
            print_text "  $MOUNT"
            print_text "    Device: $DEVICE"
            print_text "    Size: $SIZE | Used: $USED | Available: $AVAIL | Usage: $USE_PERCENT"
            print_text ""
        done
    fi

    print_separator 80

    # ============================================================================
    # NETWORK INFORMATION
    # ============================================================================
    print_header "NETWORK INFORMATION"
    print_separator 80

    if command -v ip &> /dev/null; then
        print_text "Network Interfaces:"
        print_text ""
        ip -br addr show | grep -v "lo" | while read line; do
            IFACE=$(echo $line | awk '{print $1}')
            STATE=$(echo $line | awk '{print $2}')
            ADDR=$(echo $line | awk '{print $3}')
            
            print_text "  $IFACE ($STATE)"
            print_text "    Address: $ADDR"
        done
    elif command -v ifconfig &> /dev/null; then
        print_text "Network Interfaces:"
        print_text ""
        ifconfig | grep -E "^[a-z]" | awk '{print $1}' | grep -v "lo" | while read iface; do
            ADDR=$(ifconfig $iface | grep "inet " | awk '{print $2}')
            print_text "  $iface"
            print_text "    Address: $ADDR"
        done
    fi
fi

print_bottom 80
echo ""
