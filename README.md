# UATR-CMoE
### Underwater acoustic target recognition -- Convolution-based Mixture of Experts


This is the PyTorch implementation of the paper:   
**"Unraveling Complex Data Diversity in Underwater Acoustic Target Recognition through Convolution-based Mixture of Experts"**,      
which has been published on **Expert Systems with Applications**.     

**DOI: https://doi.org/10.1016/j.eswa.2024.123431**      
**Arxiv: https://arxiv.org/abs/2402.11919**   

<br/>

<div style="display: flex; justify-content: space-between;">
    <img src="figs/framework.jpg" alt="First Figure" width="400" />
    <img src="figs/results.jpg" alt="Second Figure" width="250" />
</div>

<br/>

        
In addition to the model architecture (cmoe_model.py), this repository offers pre-extracted features of the Shipsear test set, accompanied by corresponding testing code.


## Steps of Inference

#### 1. Download pre-extracted features    
Download link:    
Save features to your own path:    

#### 2. Load models and print results

```
python get_confusion.py /path_features/
```

#### 3. (Optional) Reproduce the confusion matrix   

```
python draw_confusion.py /path_features/
```

## Citation

```
@article{xie2024unraveling,
  title={Unraveling complex data diversity in underwater acoustic target recognition through convolution-based mixture of experts},
  author={Xie, Yuan and Ren, Jiawei and Xu, Ji},
  journal={Expert Systems with Applications},
  pages={123431},
  year={2024},
  publisher={Elsevier}
}
```


