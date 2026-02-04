import os
import sys
import shutil
import zipfile
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Scrcpy 配置
SCRCPY_VERSION = "v2.4"
SCRCPY_FILENAME = f"scrcpy-win64-{SCRCPY_VERSION}.zip"
SCRCPY_DOWNLOAD_URL = f"https://github.com/Genymobile/scrcpy/releases/download/{SCRCPY_VERSION}/{SCRCPY_FILENAME}"

# ADB (Platform Tools) 配置
PLATFORM_TOOLS_URL = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"

def download_file(url: str, dest_path: Path):
    """下载文件并显示进度"""
    logger.info(f"Downloading from {url} to {dest_path}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        # 简单的进度打印，避免刷屏
                        if total_size > 0 and bytes_downloaded % (1024 * 1024) == 0: # 每 1MB 打印一次
                            percent = (bytes_downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}% ({bytes_downloaded // 1024} KB)", end="", flush=True)
            print() # 换行
        logger.info("Download completed.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # 下载失败删除不完整文件
        if dest_path.exists():
            dest_path.unlink()
        raise

def unzip_file(zip_path: Path, extract_to: Path):
    """解压 ZIP 文件"""
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Extraction completed.")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

def ensure_scrcpy_installed(tools_dir: str = "tools") -> str:
    """
    确保 Scrcpy 已安装。
    如果未安装，将自动下载并解压到 tools_dir。
    
    Returns:
        scrcpy.exe 的绝对路径
    """
    root_dir = Path(__file__).parent.parent.parent # d:\Project\auto-wzry
    tools_path = root_dir / tools_dir
    scrcpy_dir = tools_path / f"scrcpy-win64-{SCRCPY_VERSION}"
    scrcpy_exe = scrcpy_dir / "scrcpy.exe"

    # 1. 检查是否存在
    if scrcpy_exe.exists():
        logger.info(f"Scrcpy found at: {scrcpy_exe}")
        return str(scrcpy_exe)

    # 2. 如果不存在，准备下载
    tools_path.mkdir(parents=True, exist_ok=True)
    zip_path = tools_path / SCRCPY_FILENAME

    # 检查是否已经下载了压缩包但没解压
    if not zip_path.exists():
        logger.info(f"Scrcpy not found. Downloading {SCRCPY_VERSION}...")
        download_file(SCRCPY_DOWNLOAD_URL, zip_path)

    # 3. 解压
    unzip_file(zip_path, tools_path)

    # 4. 验证
    if not scrcpy_exe.exists():
        # 有时候解压后的目录结构可能不同，做个简单的容错查找
        found = list(tools_path.glob("**/scrcpy.exe"))
        if found:
            return str(found[0])
        raise FileNotFoundError(f"Failed to install Scrcpy. {scrcpy_exe} not found after extraction.")

    return str(scrcpy_exe)

def ensure_adb_installed(tools_dir: str = "tools") -> str:
    """
    确保 ADB 已安装。
    """
    root_dir = Path(__file__).parent.parent.parent
    tools_path = root_dir / tools_dir
    adb_exe = tools_path / "platform-tools" / "adb.exe"

    if adb_exe.exists():
        return str(adb_exe.parent) # 返回包含 adb 的目录路径，以便添加到 PATH

    # 下载 ADB
    logger.info("ADB not found. Downloading platform-tools...")
    tools_path.mkdir(parents=True, exist_ok=True)
    zip_path = tools_path / "platform-tools.zip"
    
    download_file(PLATFORM_TOOLS_URL, zip_path)
    unzip_file(zip_path, tools_path)
    
    if not adb_exe.exists():
         raise FileNotFoundError("Failed to install ADB.")
         
    return str(adb_exe.parent)

if __name__ == "__main__":
    # 测试下载
    logging.basicConfig(level=logging.INFO)
    try:
        path = ensure_scrcpy_installed()
        print(f"Scrcpy path: {path}")
    except Exception as e:
        print(f"Error: {e}")
