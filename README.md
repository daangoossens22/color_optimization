# Research project CSE3000 (triangle coloring methods to approximate 2D images)

## Introduction
Low-poly images are getting more popular, where an image is triangulated and colored mostly with constant colors. Here different coloring methods will be explored instead of only constant color. As this project is more focussed on coloring methods, it will only work with a fixed grid and square input images (otherwise the result will be squashed into a square output image). The program uses the following basic pipeline to produce the images.

![Program Pipelin](README_images/pipeline.png)


## Building and Running
This program has only been tested on (arch) linux.

### Prerequisites
Some programs are needed, namely:
* glfw ( ```pacman -S glfw-x11``` or ```pacman -S glfw-wayland``` )
* opencv ( ```pacman -S opencv opencv-samples``` )
* gsl ( ```pacman -S gsl``` )
* glm ( ```pacman -S glm``` )
* freeimage ( ```pacman -S freeimage``` )

### Building
To build the executable ```make``` can be executed. To clean all buildfiles ```make clean``` can be executed.

### Running
To run the executable the following command can be run:
```./coloring_methods {image_path}```

For example: ```./coloring_methods input_images/apple.jpg```

Caution when opening the edge map or the saliency map, as they can only be closed by pressing any key on the keyboad. Closing it by pressing the x button crashes the program.

### Calculating the MSE
There is a python script that can calculate the mean squared error and produces a image which is the absolute difference between the 2 provided images. To run it the following command can be executed:
```python3 mse_calc.py {target_image} {produced_image}```

## Provided coloring methods
Here the provided coloring methods are shown on one of the input images.
|||||||
----|----|----|----|----|----
target image|average color no saliency|average color with saliency|center color|linear split constant color|quadratic split constant color
<img src="input_images/apple.jpg" width="100" height="100"> | <img src="README_images/avg_no_saliency.png" width="100" height="100"> | <img src="README_images/avg_saliency.png" width="100" height="100"> | <img src="README_images/center_color.png" width="100" height="100"> | <img src="README_images/linear_split_constant_color.png" width="100" height="100"> | <img src="README_images/quadratic_split_constant_color.png" width="100" height="100"> | 
|bilinear interpolation (no opt)|bilinear interpolation (opt)|biquadratic interpolation (opt)|bicubic interpolation (opt)|bicubic interpolation (opt)
<img src="README_images/bilinear_interpolation_no_opt.png" width="100" height="100"> | <img src="README_images/bilinear_interpolation_opt.png" width="100" height="100"> | <img src="README_images/biquadratic_interpolation_opt.png" width="100" height="100"> | <img src="README_images/bicubic_interpolation_opt.png" width="100" height="100"> | <img src="README_images/biquartic_interpolation_opt.png" width="100" height="100"> | 


## Intermediate steps (needed for some coloring methods)
The saliency map is used by the coloring method; average color no saliency. The edge map is used by the coloring methods; linear split constant color and quadratic split constant color.
||||
----|----|----
target image|saliency map|edge map
<img src="input_images/apple.jpg" width="100" height="100"> | <img src="README_images/saliency_map.png" width="100" height="100"> | <img src="README_images/edge_map.png" width="100" height="100">

## (potential) future work/ideas
* Add more coloring methods
* improve the edge and saliency map
* How does better triangulisation affect the output and how can it be optimised for each coloring method?
* Combine different coloring methods (based on different factors such as the saliency/edge map)
  * for example combining linear split with nonlinear interpolation to get sharp edges on the main subject
* How good nonlinear interpolation functions as compression?
  * How does optimizing the triangulatisation affect the MSE (and how much more data does it take to store)?
  * How does one large bezier triangle compare against multiple smaller bezier triangles?
  * Does optimizing each triangle with a different order bezier triangle depending on the minimum error to the target image?
  * Does making the control points on the edges and vertices of each triangle shared affect the final result and how does it affect the time it takes to get a best fit?

## Reasearch paper
TODO provide link here

