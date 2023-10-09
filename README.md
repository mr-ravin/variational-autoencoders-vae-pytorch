# Variational Auto-encoders with CNN in Pytorch

```
|--Variational-AutoEncoders-Pytorch.ipynb # pytorch implementation of VAE.
|--dataset
|     |-train/
|     |-valid/
|     |-test/
|
|--models/   # weights are stored here
|--dataloader.py
|--vae_model.py
|--pred/     # predictions are stored here
```


- See `Variational-AutoEncoders-Pytorch.ipynb` for pytorch implementation of VAE.

### Generated Image Sample using VAE:

![image](https://github.com/mr-ravin/variational-autoencoders-vae-pytorch/blob/main/pred/pred_1.jpg?raw=true)

#### Note: VAEs do suffer from blurry generated samples and reconstructions compared to the images they have been trained on.


```
Copyright (c) 2023 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
