''' Tensorflow inception score code
Derived from https://github.com/openai/improved-gan
Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
THIS CODE REQUIRES TENSORFLOW 1.3 or EARLIER to run in PARALLEL BATCH MODE 

To use this code, run sample.py on your model with --sample_npz, and then 
pass the experiment name in the --experiment_name.
This code also saves pool3 stats to an npz file for FID calculation
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile
import math
from tqdm import tqdm, trange
from scipy import linalg
from argparse import ArgumentParser

import numpy as np
from six.moves import urllib
import tensorflow as tf

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

MODEL_DIR = './inception/'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def prepare_parser():
  usage = 'Parser for TF1.3- Inception Score scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--experiment_name', type=str, default='',
    help='Which experiment''s samples.npz file to pull and evaluate')
  parser.add_argument(
    '--experiment_root', type=str, default='samples',
    help='Default location where samples are stored (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=500,
    help='Default overall batchsize (default: %(default)s)')
  return parser


def run(config):
  # code for handling inception net derived from
  #   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
  def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3  

  def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
  
  def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr
  
  def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  
  # Inception with TF1.3 or earlier.
  # Call this function with list of images. Each of elements should be a 
  # numpy array with values ranging from 0 to 255.
  def get_inception_score(images, splits=10):
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    print('debug@tys: inception_score is max images[0] {}'.format(np.max(images[0])))
    # assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 100
    with tf.Session(config=config_tf) as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in tqdm(range(n_batches), desc="Calculate inception score"):
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))

        sess.close()
    return np.mean(scores), np.std(scores)     #, np.squeeze(np.concatenate(pools, 0))
  '''
  def get_inception_score(images, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
    bs = config['batch_size']
    with tf.Session() as sess:
      preds, pools = [], []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in trange(n_batches):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred, pool = sess.run([softmax, pool3], {'ExpandDims:0': inp})
        preds.append(pred)
        pools.append(pool)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
      return np.mean(scores), np.std(scores), np.squeeze(np.concatenate(pools, 0))
  '''
  
  # This function is called automatically.
  def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        print('debug@tys: data_url is {}, {}'.format(DATA_URL, filepath))
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config_tf) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)
        sess.close()

  # if softmax is None: # No need to functionalize like this.
  _init_inception()

  # fname = '%s/%s/samples.npz' % (config['experiment_root'], config['experiment_name'])
  fname = '%s/%s/samples.npz' % (config['experiment_root'], config['experiment_name'])
  print('loading %s ...'%fname)
  ims = np.load(fname)['x']
  import time
  t0 = time.time()
  #inc_mean, inc_std, pool_activations = get_inception_score(list(ims.swapaxes(1,2).swapaxes(2,3)), splits=10)
  print(f'image shape is {ims.shape}')
  inc_mean, inc_std = get_inception_score(list(ims.swapaxes(1,2).swapaxes(2,3)), splits=10)
  t1 = time.time()
  # Load statistic files of real images
  fname = './fid_stat/fid_stats_cifar10_train.npz'
  real_stat = np.load(fname)
  with tf.Session(config=config_tf) as sess:
    gen_mu, gen_sigma =  calculate_activation_statistics(ims.swapaxes(1,2).swapaxes(2,3), sess, batch_size=50, verbose=False)
  fid = calculate_frechet_distance(gen_mu, gen_sigma, real_stat['mu'], real_stat['sigma'], eps=1e-6)
  print('Saving pool to numpy file for FID calculations...')
  #np.savez('%s/%s/TF_pool.npz' % (config['experiment_root'], config['experiment_name']), **{'pool_mean': np.mean(pool_activations,axis=0), 'pool_var': np.cov(pool_activations, rowvar=False)})
  print('Inception took %3f seconds, score of %3f +/- %3f.'%(t1-t0, inc_mean, inc_std))
  print('FID is %3f'%(fid))

def main():
  # parse command line and run
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()
