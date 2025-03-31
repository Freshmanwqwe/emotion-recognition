# emotion-recognition
## Requirements

```shell
pip install -r requirements.txt
```

### Models

Download models from https://huggingface.co/wind-strider/emotion-detection/tree/main

"model_epoch_5.pth" and "yolo_face_detection.pt" have to be in the same directory as  testOnCamera.py and  testOnMp4.py

### Run

Run testOnCamera.py to recognize the faces and the emotions in front of your PC camera.

```shell
python testOnCamera.py
```



There is a line of code in the 47th line of testOnMp4.py.

```python
video_path = "class.mp4"
```

Change the video_path to a video path whichever you like and run testOnMp4.py to detect the faces and the emotions in the video.

```shell
python testOnMp4.py
```

