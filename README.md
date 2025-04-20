setup
```
conda create -n pje python=3.12
conda activate pje

pip install djitellopy
pip install opencv-python
pip install keyboard # for manual flying in record.py
```

brew install portaudio

to run `drone_tracker` run `uv run python -m drone_tracker.drone_controller` from the main directory