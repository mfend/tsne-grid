#from skimage.util import img_as_float
#from skimage import data
#from skimage.transform import rescale
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
from collections import Counter
from scipy import ndimage as nd
from PIL import Image
import pylab as pl
import numpy as np
import pickle
import copy
import pdb 
import os

def pie_slices(labes,inds):
    return (sum(labes[inds]==0),sum(labes[inds]==1))

WRITE = False

# I feel like I haven't slept enough to achieve this task)
# what we want is % of each cluster that is from each label
def rightwrong(clusters,labels):

    labs = np.unique(labels)
    clus = np.unique(clusters)
    
    linds = []
    cinds = []
    
    for lab in labs:
        linds.append(np.nonzero(labels==lab)[0])
    for clu in clus:
        cinds.append(np.nonzero(clusters==clu)[0])

    output = {}
        
    for clu in clus:
        output[clu] = Counter(labels[clusters==clu]).most_common()

    return output

def radius(feats,labels,centroids):

    xs = feats[:,0]
    ys = feats[:,1]

    rms = []

    ls = np.unique(labels)

    for i in np.arange(0,len(ls),1):
        xl = xs[labels == ls[i]]
        yl = ys[labels == ls[i]]

        rs = np.sqrt((centroids[i][0] - xl)**2 + (centroids[i][1] - yl)**2)
        rms.append(np.mean(rs))

    return rms

def cents(centroids):

    x_mean = np.mean(centroids[:,0])
    y_mean = np.mean(centroids[:,1])

    return np.mean(np.sqrt((centroids[:,0]-x_mean)**2 + (centroids[:,1]-y_mean)**2))

def read_dats(fpath):
    have = {}
    have['fname'] = []
    have['alts'] = []
    have['lons'] = []
    have['lats'] = []
    filo = open(fpath,'r')
    for line in filo:
        if not line.startswith('fname'):
            have['fname'].append(line.strip().split(',')[0])
            have['alts'].append(np.float(line.strip().split(',')[1]))
            have['lats'].append(np.float(line.strip().split(',')[2]))
            have['lons'].append(np.float(line.strip().split(',')[3]))
    have['fname'] = np.array(have['fname'])
    have['alts'] = np.array(have['alts'])
    have['lons'] = np.array(have['lons'])
    have['lats'] = np.array(have['lats'])
    return have

def order_data(data,fnames):
    have = {}
    have['fname'] = []
    have['alts'] = []
    have['lons'] = []
    have['lats'] = []
    have['site'] = []
    for item in fnames:
        if item in data['fname']:
            have['fname'].append(item)
            have['alts'].append(data['alts'][data['fname']==item][0])
            have['lats'].append(data['lats'][data['fname']==item][0])
            have['lons'].append(data['lons'][data['fname']==item][0])
            if item.startswith('I'):
                date = item.split('_')[1]
            else:
                date = item.split('_')[0]
            if date == '20181013':
                have['site'].append('mono')
            else:
                have['site'].append('searles')
    have['site'] = np.array(have['site'])
    return have
    
def plot_tsne(feats,savename,pick=False,test=False,savek=True):

    #path = '/home/binuab/Documents/jp/tufa/searles/textures/cropped/'
    #path = '/media/binuab/ExtraDrive1/textureDB/dtd/images/'

    if pick:
        fili = open(feats,'rb')#'tsne_nd_tuf.pkl','rb')
        feats = pickle.load(fili)

    #fnames = os.listdir('/home/binuab/Documents/jp/tufa/searles/textures/tsne/scalesnbubs/')#[:324]

    fnames = os.listdir('/home/binuab/Documents/jp/tufa/searles/textures/cropped/')

    if test:
    
        clasif = []

        for item in fnames:
            clasif.append(item.split('_')[0])

        clasif = np.array(clasif)
    else:
        clasif = np.array(['b']*feats.shape[0])
    
    kinfo = KMeans(n_clusters=3,random_state=0).fit(feats)
    
    clusts = kinfo.cluster_centers_
    labels = kinfo.labels_

    if test:
    
        bub_inds = clasif == 'bubbly'
        sca_inds = clasif == 'scaly'
        str_inds = clasif == 'striped'
    
        cols = copy.deepcopy(clasif)
    
        cols[cols=='bubbly']= 'r'
        cols[cols=='scaly']= 'g'
        cols[cols=='striped']= 'b'

        percor = rightwrong(labels,clasif)

    else:
        cols = copy.deepcopy(clasif)
        percor = {}

    radii = radius(feats,labels,clusts)
    cent_r = cents(clusts)

    if savek:
        output = open(savename.split('.')[0]+'.txt','w')
        output.write('centroid radii = '+str(cent_r)+'\n')
        output.write('radii = '+str(radii)+'\n')
        for item in percor.keys():
            output.write(str(item)+': '+str(percor[item])+'\n')
        output.close()

    data1 = read_dats('/home/binuab/Documents/jp/tufa/searles/textures/'+\
                     'image_metadata/image_gps_all.txt')

    data = order_data(data1,fnames)

    cols = 2
    figy = 10
    
    if 'mono' in savename:
        cols = 3
        figy = 20

    else:
        inds = data['site'] == 'searles'
        for key in data.keys():
            data[key] = np.array(data[key])[inds]
        
    pl.ion()
    
    pl.figure(2,figsize=(figy,5))
    pl.clf()
    pl.subplot(1,cols,1)
    # fig = pl.figure1(2)
    # ax = pl.axes(projection='3d')
    # ax.scatter3D(feats[:,0],feats[:,1],feats[:,2],c=labels)
    pl.scatter(feats[:,0],feats[:,1],c=data['alts'])
    pl.colorbar()
    pl.title('Altitude (m)')

    # fig = pl.figure(3)
    # ax = pl.axes(projection='3d')
    # ax.scatter3D(feats[:,0],feats[:,1],feats[:,2],c=clasif)

    pl.subplot(1,cols,2)
    pl.scatter(feats[:,0],feats[:,1],c=data['lats'])
    #pl.suptitle('radius of centroids = '+str(cent_r))
    pl.colorbar()
    pl.title('Latitude')

    data['xs'] = feats[:,0]
    data['ys'] = feats[:,1]
    data['klusters'] = labels
    
    out = open(savename.split('.')[0]+"_data.pkl",'wb')
    pickle.dump(data,out)
    out.close()
    
    if 'mono' in savename:
        pl.subplot(1,cols,3)
        pl.plot(feats[:,0][data['site']=='searles'],feats[:,1]\
                [data['site']=='searles'],'ro',label='searles')
        pl.plot(feats[:,0][data['site']=='mono'],feats[:,1]\
                [data['site']=='mono'],'bo',label='mono')
        #pdb.set_trace()
        pl.title('Basin')
        pl.legend(loc=2)
    pl.tight_layout()
    pl.subplots_adjust(top=0.9)
    pl.savefig(savename)

#plot_tsne('tsne_searles_p17i250.pkl','tsne_searles_p17i250.png',pick=True,test=False,savek=False)

#fparts = ['p17i250','p18i20000','p19i250','p19i3000','p20i250','p9i250']


path = '/home/binuab/Documents/jp/tufa/searles/textures/tsne/cluster_data/'

# plot_tsne(feats,savename,pick=False,test=False,savek=True):

#for fpart in fparts:
#    plot_tsne(path+'tsne_searlesmono_'+fpart+'.pkl','tsne_searlesmono_'+fpart+\
#              '.png',pick=True,test=False,savek=False)

plot_tsne(path+'tsne_searles_p18i20000.pkl',\
          'tsne_searles_p18i20000.png',pick=True,test=False,savek=False)
