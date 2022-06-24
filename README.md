# Narrow U-Net  
By Samuel Li & Matthew French

### Installation  
1. Clone this repository (i.e. git clone ...) 
2. Create and activate a virtual environment:
 ```conda create -n narrow_unet```  
  ```conda activate narrow_unet``` 
  or  
  ```python3 -m venv .narrow_unet```
  ```source .narrow_unet/bin/activate```
3. Install requirements
  ``` pip install -r requirements.txt```

### Training  
1. Check GPU availability  
```nvidia-smi``` 
2. Run training script 
```python3 scripts/train_vgg_on_unstretched_ballies.py -g #gpu```
Note: the flag g corresponds to the GPU device to use for training
