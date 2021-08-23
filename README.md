<!-- Instructions to run the program:

-> Store front and side images in the 'in' folder in the repo directory.
-> Change directory to hmr2/src
-> Run download_model.py if running for the first time.
-> Run final.py and enter height. Enter height as 0 to stop the program.
-> Output is stored in the 'out' folder in the repo directory.
-> accuracy.py in the repo directory can be used to check accuracy of the calculation if actual measurements 
    are stored in the actual_measure.json file in the 'in' folder. -->

# 2D Auto Measurement
Extraction of Human Body Measurements from 2D images for Clothing options

|![Images of Sample pose](/sample_data/Pose_3d_model/front_view.png "Sample Pose - Front") | ![Images of Sample pose](/sample_data/Pose_3d_model/side_view.png "Sample Pose - Side")|

## Instructions

Install requirements
```sh
pip install requirements.txt
```
Store front and side images in the 'in' folder in the repo directory.
Change directory to hmr2/src.
```sh
cd hmr2/src
```
Run download_model.py if running for the first time.
```sh
python download_model.py
```
Run final.py and enter height. Enter height as 0 to stop the program.
```sh
python final.py
```
Run accuracy.py to check accuracy of the calculation after storing actual measurements in the actual_measure.json file in the 'in' folder.
```sh
cd .. 
cd ..
python accuracy.py
```

### Timeline PPT
[Canva][ppt_url]


[ppt_url]: https://www.canva.com/design/DAEdhnt0qx8/2dCrGLU2EL8e3qGovTzmnA/view?utm_content=DAEdhnt0qx8&utm_campaign=designshare&utm_medium=link&utm_source=sharebuttong

