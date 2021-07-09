# tsne-grid
This is a python script for [t-SNE](https://lvdmaaten.github.io/tsne/) as used in the manuscript "A Computer Vision Algorithm for Interpreting Lacustrine Carbonate Textures at Searles Valley, USA.", submitted to Computers and Geosciences.
<p align="center">
<img src="./tsne_grid_smaller.jpg" width="290" height="290" />
</p>

### Setup
Dependencies:
* [tensorflow](https://www.tensorflow.org/install/)
* [keras](https://keras.io/)
* [scikit-learn](https://scikit-learn.org/stable/)

### Usage
Basic usage:
```bash
python tsne_nd_smallset.py --dir <path/to/example-directory/> --size 3
```
#### Options (required)
* `--dir`: Path to directory containing image collection.
* `--size`: should be set to 3

#### Options (optional)
* `--per`: Perplexity for t-SNE algorithm. Default is 50.
* `--iter`: Number of iterations for t-SNE algorithm. Default is 5000.

### Implementation details
VGG16 (without fc layers on top) is used to generate high dimensional feature representations of images. 2D representaions of these features are formed using scikit-learn's t-SNE implementation. These 2D representations are converted into a square grid using [Jonker-Volgenant](https://blog.sourced.tech/post/lapjv/) algorithm.

### Support
The script was tested with tensorflow (2.2.0) and keras (2.4.3) on a Nvidia GeForce GTX 1060

### References
* L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008. [PDF](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf) [[Supplemental material]](https://lvdmaaten.github.io/publications/misc/Supplement_JMLR_2008.pdf) [[Talk]](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw) [[Code]](https://lvdmaaten.github.io/tsne/)
