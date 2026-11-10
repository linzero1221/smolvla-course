# 课程项目的 WSL2 安装与卸载

| 项目 | 位置或名称 |
|---|---|
| Linux 发行版 | `Ubuntu-24.04`，使用 WSL2 |
| Ubuntu 专用安装目录 | `D:\WSL\SmolVLA\Ubuntu-24.04`，包含 Linux 文件系统的虚拟磁盘 |
| Windows 项目目录 | `D:\AiCoding\VLA`，在 Ubuntu 中对应 `/mnt/d/AiCoding/VLA` |
| 要长期保留的实验产物 | `D:\AiCoding\VLA\outputs` 和 `D:\AiCoding\VLA\deliverables` |

2026 年 11 月 10 日，本机核验了 Ubuntu 24.04.5 LTS 的 WSL2 注册和 D 盘虚拟磁盘 `D:\WSL\SmolVLA\Ubuntu-24.04\ext4.vhdx`。Ubuntu 内的 `nvidia-smi` 识别到 RTX 5060（8151 MiB，驱动 610.88），CUDA 驱动测试成功分配显存并完成主机与显卡之间的数据往返。PyTorch 张量检查和 LIBERO 单回合评估需要在依赖安装完成后验收。

## 安装

1. 在管理员 PowerShell 中安装 Microsoft WSL 基础组件。首次启用虚拟机平台时，按 Windows 提示保存工作并手动重启。 [微软安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)

   ```powershell
   wsl.exe --install --no-distribution --web-download
   ```

2. 重启后运行 `wsl.exe --status`，确认 WSL2 可以启动。在当前 Windows 用户的 PowerShell 中把 Ubuntu 安装到专用目录。 [WSL 安装参数](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#install)

   ```powershell
   New-Item -ItemType Directory -Force -Path 'D:\WSL\SmolVLA' | Out-Null
   wsl.exe --install -d Ubuntu-24.04 --web-download --location 'D:\WSL\SmolVLA\Ubuntu-24.04' --no-launch
   wsl.exe --list --verbose
   Get-ChildItem -LiteralPath 'D:\WSL\SmolVLA\Ubuntu-24.04' -Filter '*.vhdx'
   ```

   安装验收时确认 `Ubuntu-24.04` 的 `VERSION` 为 `2`，并确认 D 盘专用目录里有虚拟磁盘文件。

   本机在线安装曾报 `WININET_E_INVALID_SERVER_RESPONSE`。以下命令使用经 SHA-256 核验的 Ubuntu 24.04.5 AMD64 官方镜像完成安装。若更换镜像版本，请同时核对对应的哈希。[Ubuntu 24.04 发布目录](https://releases.ubuntu.com/24.04/) · [微软 WSL 发行版清单](https://github.com/microsoft/WSL/blob/master/distributions/DistributionInfo.json)

   ```powershell
   New-Item -ItemType Directory -Force -Path 'D:\WSL\SmolVLA' | Out-Null
   $ubuntuImage = 'D:\WSL\SmolVLA\ubuntu-24.04.5-wsl-amd64.wsl'
   curl.exe --fail --location --retry 3 --output $ubuntuImage 'https://releases.ubuntu.com/24.04.5/ubuntu-24.04.5-wsl-amd64.wsl'
   if ($LASTEXITCODE -ne 0) { throw 'Ubuntu 镜像下载失败' }
   $expectedSha256 = 'bb415d824822c4b878125729af451a5d18fb13d1cf5cbed9a7393ad64ac6039e'
   if ((Get-FileHash -LiteralPath $ubuntuImage -Algorithm SHA256).Hash -ne $expectedSha256) {
       throw 'Ubuntu 镜像 SHA-256 不匹配'
   }
   wsl.exe --install --from-file $ubuntuImage --name Ubuntu-24.04 --location 'D:\WSL\SmolVLA\Ubuntu-24.04' --no-launch
   ```

3. 在 PowerShell 执行 `wsl.exe -d Ubuntu-24.04`。首次启动按提示创建 Ubuntu 用户，输入密码时终端不会显示字符。进入 Ubuntu 后运行 `nvidia-smi` 或 `/usr/lib/wsl/lib/nvidia-smi` 确认显卡可用。WSL 使用 Windows 上的 NVIDIA 驱动。[Ubuntu 首次启动说明](https://ubuntu.com/wsl/docs/latest/howto/install-ubuntu-wsl2/#starting-an-ubuntu-instance) · [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 在 Ubuntu 中配置项目

用 Ubuntu 的 Python 3.12 建立项目虚拟环境。CMake 和 C++ 编译工具用于构建 LIBERO 的 EGL 探测库，Mesa 提供仿真渲染。以下命令在 Ubuntu 终端执行。

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv build-essential cmake libgl1 libegl1 libopengl0 libosmesa6 ffmpeg
python3 -m venv "$HOME/.venvs/smolvla-course"
source "$HOME/.venvs/smolvla-course/bin/activate"
python -m pip install uv
uv pip install --torch-backend=cu128 torch==2.11.0 torchvision==0.26.0 \
    'lerobot[training,evaluation,smolvla,libero,peft]==0.6.1'
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
cd /mnt/d/AiCoding/VLA
mkdir -p outputs/environment
python -m pip freeze > outputs/environment/pip-freeze.txt
bash scripts/smoke_eval.sh
```

若 LIBERO 首次询问是否自定义数据集目录，输入 `n` 使用默认位置。若 CUDA 张量正常而 EGL 初始化失败，运行 `MUJOCO_GL=osmesa bash scripts/smoke_eval.sh` 试用软件渲染，并记录实际采用的后端。

虚拟环境和默认的模型缓存位于 Ubuntu 文件系统，实际占用 D 盘虚拟磁盘空间。单回合完成后检查 `outputs/smoke/` 中该次运行的日志、`eval_info.json` 和 MP4。将这组产物用于环境验收，正式评测按实验配置另行运行。[LeRobot 安装文档](https://huggingface.co/docs/lerobot/v0.6.1/en/installation) · [PyTorch CUDA 12.8 安装说明](https://pytorch.org/get-started/previous-versions/#v2110)

## 日常停用

等待下载结束，或等训练与评估保存完产物，再在 PowerShell 中停止这个发行版。下次通过 `wsl.exe -d Ubuntu-24.04` 重新进入。 [WSL 停止命令](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#terminate)

```powershell
wsl.exe --terminate Ubuntu-24.04
```

## 作业完成后卸载

1. 将放在 Ubuntu 文件系统中的模型、日志、回放及未提交的代码复制到 Windows 项目目录，并在 Windows 中检查文件能打开。已写入 `/mnt/d/AiCoding/VLA` 的文件本身就在 D 盘项目目录。需要保留完整 Linux 环境时，可以先用 `wsl.exe --export` 导出到另一个目录。 [WSL 导出命令](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#export-a-distribution)
2. 备份完成后，核对 `wsl.exe --list --verbose` 显示的名称。以下命令会永久删除 `Ubuntu-24.04` 内的文件和设置。注销后再次列出发行版，并检查专用目录。如果只剩空目录，可以在资源管理器里删除它。 [微软发行版卸载说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#unregister-or-uninstall-a-linux-distribution)

   ```powershell
   wsl.exe --unregister Ubuntu-24.04
   wsl.exe --list --verbose
   ```

3. 如果以后也不使用其他 WSL 发行版或依赖 WSL 的工具，可在 **设置 → 应用 → 已安装的应用** 中卸载 **Windows Subsystem for Linux**。如需关闭本次启用的「虚拟机平台」，先确认没有其他程序依赖它，再通过「启用或关闭 Windows 功能」关闭，并按系统提示重启。 [微软 WSL 安装与组件说明](https://learn.microsoft.com/en-us/windows/wsl/install)
