## Avataar Assignment Readme

Stable Diffusion v1.5 based Image Generation Pipeline
It supports generating images from following

1) text Prompt
2) text Prompt + Depth Map
3) text Prompt + Normal Map
4) text Prompt + Normal Map + Depth Map


test.py : main file to run with following arguments
```
python test.py -i <depth/image/path(png, npy)> -p <prompt> -m <int>
```
mode description:
* 0 for txt2img
* 1 for txt-depth2img
* 2 for txt-normal2img
* 3 for txt-depth-normal2img

stable_diffusion.py :  definition of stable diffusion methods

depth2normal.py : analytical method for getting normal map from depth map