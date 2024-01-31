@echo off
CALL C:\Users\analyst\Anaconda3\Scripts\activate.bat tf
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
python "C:\Users\analyst\Documents\Python Scripts\BactUnet\bactunet_gui.py"
pause
