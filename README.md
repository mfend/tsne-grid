# tsne-grid
This is a python script for [t-SNE](https://lvdmaaten.github.io/tsne/) visualization of multiple images in a square grid.
<p align="center">
<img src="./tsne_grid_smaller.jpg" width="290" height="290" />
</p>

### Setup
Dependencies:
* [tensorflow](https://www.tensorflow.org/install/)
* [keras](https://keras.io/)

### Usage
Basic usage:
```bash
python tsne_grid.py --dir ./examples/inputs/ --size 4
```
#### Options (required)
* `--dir`: Path to directory containing image collection.
* `--size`: Number of small images in a row/column in output image.

#### Options (optional)
* `--res`: Width/height if small images (in pixels). Default is 224.
* `--name`: Output filename. Default is tsne_grid.jpg
* `--path`: Output image path. Default is current directory.
* `--per`: Perplexity for t-SNE algorithm. Default is 50.
* `--iter`: Number of iterations for t-SNE algorithm. Default is 5000.

### Implementation details
VGG16 (without fc layers on top) is used to generate high dimensional feature representations of images. 2D representaions of these features are formed using scikit-learn's t-SNE implementation. These 2D representations are converted into a square grid using [Jonker-Volgenant](https://blog.sourced.tech/post/lapjv/) algorithm.

### Support
The script was tested with tensorflow (1.4.0) and keras (2.1.1) on Nvidia GeForce GT 740M

### References
* L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008. [PDF](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf) [[Supplemental material]](https://lvdmaaten.github.io/publications/misc/Supplement_JMLR_2008.pdf) [[Talk]](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw) [[Code]](https://lvdmaaten.github.io/tsne/)
