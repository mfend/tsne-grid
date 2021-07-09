import numpy as np
import os, argparse, pdb, pickle
import tensorflow as tf
import matplotlib as mlp
import matplotlib.pyplot as plt
import pylab as pl
from PIL import Image
from lapjv import lapjv
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Flatten

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, help="number of small images in a row/column in output image")
parser.add_argument('-d', '--dir', type=str, help="source directory for images")
parser.add_argument('-r', '--res', type=int, default=224, help="width/height of output square image")
parser.add_argument('-n', '--name', type=str, default='tsne_grid.jpg', help='name of output image file')
parser.add_argument('-p', '--path', type=str, default='./', help="destination directory for output image")
parser.add_argument('-x', '--per', type=int, default=50, help="tsne perplexity")
parser.add_argument('-i', '--iter', type=int, default=5000, help="number of iterations in tsne algorithm")
parser.add_argument('-z', '--numd', type=int, default=2, help="number of tsne dimensions")

# assign command line arguments to variable names
args = parser.parse_args()
out_res = args.res
out_name = args.name
out_dim = args.size
to_plot = np.square(out_dim)
perplexity = args.per
tsne_iter = args.iter
n = args.numd

# Raise error for only one image
if out_dim == 1:
    raise ValueError("Output grid dimension 1x1 not supported.")

# Assign image directory path/name to variable, raise error if the directory doesn't exist 
if os.path.exists(args.dir):
    in_dir = args.dir
else:
    raise argparse.ArgumentTypeError("'{}' not a valid directory.".format(in_dir))

# Assign output directory path/name to variable, raise error if the directory doesn't exist 
if os.path.exists(args.path):
    out_dir = args.path
else:
    raise argparse.ArgumentTypeError("'{}' not a valid directory.".format(out_dir))

# define function for VGG16 sequential model, using imagenet weights
def build_model():
    base_model = VGG16(weights='imagenet')
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    return Model(inputs=base_model.input, outputs=top_model(base_model.output))

# define function for loading all images in directory and returning them as a collection
def load_img(in_dir):
    pred_img = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    img_collection = []
    for idx, img in enumerate(pred_img):
        img = os.path.join(in_dir, img)
        try:
            img_collection.append(image.load_img(img, target_size=(out_res, out_res)))
        except:
            pdb.set_trace()
    if (np.square(out_dim) > len(img_collection)):
        raise ValueError("Cannot fit {} images in {}x{} grid".format(len(img_collection), out_dim, out_dim))
    return img_collection

# define function for getting activations for each image in collection
def get_activations(model, img_collection):
    activations = []
    for idx, img in enumerate(img_collection):
        #if idx == to_plot:
        #    break;
        print("Processing image {}".format(idx+1))
        img = img.resize((224, 224), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        activations.append(np.squeeze(model.predict(x)))
    return activations

# define function for creating t-SNE model based on activations
def generate_tsne(activations,n):
    tsne = TSNE(perplexity=perplexity, n_components=n, init='random', n_iter=tsne_iter,learning_rate=10)
    #X_nd = tsne.fit_transform(np.array(activations)[0:to_plot,:])
    X_nd = tsne.fit_transform(np.array(activations))

    X_nd -= X_nd.min(axis=0)
    X_nd /= X_nd.max(axis=0)
    return X_nd
 
# define main function
def main():
    model = build_model()
    img_collection = load_img(in_dir)
    activations = get_activations(model, img_collection)
    print("Generating 2D representation.")
    X_nd = generate_tsne(activations,n)

    return X_nd
    
# Run the main program, dump and safe output in pickles
if __name__ == '__main__':
    dats = main()
    fili = open('tsne_example_p'+str(perplexity)+'i'+str(tsne_iter)+'.pkl','wb')
    pickle.dump(dats,fili)
    fili.close()

# Fit k-means to output data, using 3 clusters
kinfo = KMeans(n_clusters=3,random_state=0).fit(dats)

# Assign cluster info and data labels to variable names
clusts = kinfo.cluster_centers_
labels = kinfo.labels_

# find indices of members of each cluster
cluster0 = np.nonzero(labels==0)[0]
cluster1 = np.nonzero(labels==1)[0] 
cluster2 = np.nonzero(labels==2)[0]

# make an iteratable list of cluster indices
cluster_sets = [cluster0,cluster1,cluster2]

i = 1

print('\n'+'='*10+'    Final Clusters    '+'='*10+'\n')

# print the file names of the images in each cluster
for inds in cluster_sets:

    images = np.array(os.listdir(in_dir))[inds]
    
    print('\nCluster '+str(i)+' contains:')
    for item in images:
        print(item)

    print('\n'+42*'-'+'\n')

    i+=1
    
# plot up the clusters
pl.scatter(dats[:,0],dats[:,1],c=labels)
pl.ion()
pl.show()
