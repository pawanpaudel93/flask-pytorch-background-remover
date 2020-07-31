import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from decouple import config
import torch
import numpy as np
import cv2

# Apply the transformations needed
import torchvision.transforms as T

def decode_segmap(image, source, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  
  rgb = np.stack([r, g, b], axis=2)

  # Load the foreground input image 
  foreground = np.array(source.convert("RGB"))

  # Change the color of foreground image to RGB 
  # and resize image to match shape of R-band in RGB output map
  # foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
  foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))

  # Create a background array to hold white pixels
  # with the same size as RGB output map
  background = 255 * np.ones_like(rgb).astype(np.uint8)

  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)

  # Create a binary mask of the RGB output map using the threshold value 0
  th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

  # Apply a slight blur to the mask to soften edges
  alpha = cv2.GaussianBlur(alpha, (7,7),0)

  # Normalize the alpha mask to keep intensity between 0 and 1
  alpha = alpha.astype(float)/255

  # Multiply the foreground with the alpha matte
  foreground = cv2.multiply(alpha, foreground)  
  
  # Multiply the background with ( 1 - alpha )
  background = cv2.multiply(1.0 - alpha, background)  
  
  # Add the masked foreground and background
  outImage = cv2.add(foreground, background)

  # Return a normalized output image for display
#   outImage = (outImage-outImage.min())/(outImage.max()-outImage.min())
  return outImage

def remove_background(net, file, file_path, DEFAULT_CONFIG, dev='cpu'):
    # removing the image with the same name if it is already present
    Path(file_path).unlink(missing_ok=True)
    if torch.cuda.is_available():
      dev='cuda'
    img = Image.open(file)
    # Comment the Resize and CenterCrop for better inference results
    resize_to = config("RESIZETO", cast=str)
    resize_to = resize_to.split(",")
    resize_to = int(resize_to[0]) if len(resize_to)==1 else tuple([int(v) for v in resize_to])
    trf = T.Compose([T.Resize(resize_to),
                    #T.CenterCrop(224), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    
    output_image = decode_segmap(om, img)
    if DEFAULT_CONFIG["BLACKnWHITE"]:
        output_image = cv2.cvtColor(np.array(output_image, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(file_path, output_image)
    else:
        output_image = (output_image-output_image.min())/(output_image.max()-output_image.min())
        plt.imsave(file_path, output_image, dpi=1000)
