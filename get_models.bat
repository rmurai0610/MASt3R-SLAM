@echo off
setlocal enabledelayedexpansion

echo MASt3R-SLAM Model Download Script
echo ===================================
echo.

if not exist "mast3r_slam" (
    echo Error: Run this script from the MASt3R-SLAM root directory
    exit /b 1
)

echo Creating checkpoints directory...
if not exist "checkpoints" mkdir checkpoints

echo.
echo Downloading MASt3R checkpoints...
echo Note: These files are large (1-2 GB each). This may take several minutes.
echo.

set "checkpoint_url=https://download.europe.naverlabs.com/ComputerVision/MASt3R"

echo Downloading MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth...
powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('!checkpoint_url!/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth') }" || (echo Warning: Failed to download metric checkpoint using PowerShell method)

echo.
echo Downloading MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth...
powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('!checkpoint_url!/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth', 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth') }" || (echo Warning: Failed to download retrieval trainingfree checkpoint using PowerShell method)

echo.
echo Downloading MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl...
powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('!checkpoint_url!/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl', 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl') }" || (echo Warning: Failed to download codebook checkpoint using PowerShell method)

echo.
echo Verifying checkpoints...
if exist "checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" (
    echo [OK] MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
) else (
    echo [MISSING] MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
)

if exist "checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" (
    echo [OK] MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
) else (
    echo [MISSING] MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
)

if exist "checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" (
    echo [OK] MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
) else (
    echo [MISSING] MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
)

echo.
echo Model download completed.
echo Checkpoints location: %CD%\checkpoints
echo.

if errorlevel 1 (
    echo Note: If downloads failed, you can manually download from:
    echo %checkpoint_url%/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    echo %checkpoint_url%/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
    echo %checkpoint_url%/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
)

echo.
