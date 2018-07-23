import os
import matplotlib.pyplot as plt
import utils

# Vector quantization for compressing images
from quantize_image import Quantize

        img = utils.load_dataset('dog')['I']/255
        shape = np.shape(img)
        print(shape)
        img = np.reshape(img,(np.size(img)/3,3))
        #print(np.shape(img))
        model = Quantize(b=6)
        model.quantize(img)
        model.dequantize(img)
        img = np.reshape(img,shape)
        

        
        #utils.plot_2dclustering(img, model.dequantize(img))
        
        fname = os.path.join("..", "figs", "quantize.png")
        plt.imsave(fname, img)
        #plt.imshow(img)
        #plt.show
        #plt.savefig(fname)
        
print("\nFigure saved as '%s'" % fname)