from keras.models import Sequential
from keras.layers import Conv2D,Input
from keras.optimizers import SGD,Adam
from skimage.measure import compare_ssim as ssim 
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import os
import math


def psnr(target,ref):
  target_data=target.astype('float')
  ref_data=ref.astype('float')
  deff=target_data-ref_data
  deff=deff.flatten()
  rmse=math.sqrt(np.mean(deff**2.))
  return 20 * math.log10(255./rmse)

def rms(target,ref):
  err=np.sum((target.astype(float)-ref.astype(float))**2)
  err/=float(target.shape[0]*target.shape[1])
  return err

def compare_images(target,ref):
  score=[]
  score.append(psnr(target,ref)) 
  score.append(rms(target,ref)) 
  score.append(ssim(target,ref,multichannel=True))
  return score 

def prepare_images(path, factor):
    
    for file in os.listdir(path):
        
        img = cv2.imread(path + '/' + file)
        
        h, w, _ = img.shape
        new_height = int(h / factor)
        new_width = int(w / factor)
        
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
        
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
        
        print('Saving {}'.format(file))
        cv2.imwrite('images/{}'.format(file), img)


prepare_images('source/',2)


for file in os.listdir('images/'):
    
    # open target and reference images
    target = cv2.imread('images/{}'.format(file))
    ref = cv2.imread('source/{}'.format(file))
    
    # calculate score
    scores = compare_images(target, ref)

    # print all three scores with new line characters (\n) 
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    
    
    
def model():
  SRCNN=Sequential()
  SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
  SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
  SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
  adam=Adam(0.0003)

  SRCNN.compile(adam,loss='mean_squared_error',metrics=['mean_squared_error'])

  return SRCNN




def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img

def shave(image, border):
    img = image[border: -border, border: -border]
    return img


def predict(image_path):
  srcnn=model()
  srcnn.load_weights('3051crop_weight_200.h5')
  path,file=os.path.split(image_path)
  degraded=cv2.imread(image_path)
  ref=cv2.imread('source/{}'.format(file))

  ref=modcrop(ref,3)
  degraded=modcrop(degraded,3)

  temp=cv2.cvtColor(degraded,cv2.COLOR_BGR2YCrCb)

  Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
  Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    
  pre = srcnn.predict(Y, batch_size=1)
    
  pre *= 255
  pre[pre[:] > 255] = 255
 
  pre[pre[:] < 0] = 0
  pre = pre.astype(np.uint8)
    
  temp = shave(temp, 6)
  temp[:, :, 0] = pre[0, :, :, 0]
  output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    
  ref = shave(ref.astype(np.uint8), 6)
  degraded = shave(degraded.astype(np.uint8), 6)
    
  scores = []
  scores.append(compare_images(degraded, ref))
  scores.append(compare_images(output, ref))
    
  return ref, degraded, output, scores




for file in os.listdir('images'):
    
    # perform super-resolution
    ref, degraded, output, scores = predict('images/{}'.format(file))
    
    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[1].set(xlabel = 'PSNR: {}\nMSE: {} \nSSIM: {}'.format(scores[0][0], scores[0][1], scores[0][2]))
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    axs[2].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(scores[1][0], scores[1][1], scores[1][2]))

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
      
    print('Saving {}'.format(file))
    fig.savefig('output/{}.png'.format(os.path.splitext(file)[0])) 
    plt.close()


