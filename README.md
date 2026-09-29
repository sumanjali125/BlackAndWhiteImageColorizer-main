Converts Black and White images into colorized images.
Developed a Black and White Image Colorizer using Python and a machine learning model to add color to black and white images. It helps bring old or grayscale photos to life by guessing what the original colors might have been.
Run commands:


cd "C:\Users\user\Downloads\BlackAndWhiteImageColorizer-main (1)\BlackAndWhiteImageColorizer-main"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pyautogui
pip uninstall opencv-python -y
pip install opencv-python==4.10.0.84
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2.dnn,'readNetFromCaffe'))"
python colorize.py -i "images\images\einstein.jpg"
