import torch
from generate import net, main
import cv2

img_raw = cv2.imread('./test.jpg')

model = torch.jit.script(net, img_raw)

model.save('mobilenet0.25_script.pth')  

model = torch.jit.load('mobilenet0.25_script.pth')

main(model)

print(net)