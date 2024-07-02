# NNG-Mix

This repository contains the implementation of the paper:

**NNG-Mix: Improving Semi-supervised Anomaly Detection with Pseudo-anomaly Generation**  
[Hao Dong](https://sites.google.com/view/dong-hao/), [Gaëtan Frusque](https://frusquegaetan.github.io/), [Yue Zhao](https://viterbi-web.usc.edu/~yzhao010/), [Eleni Chatzi](https://chatzi.ibk.ethz.ch/about-us/people/prof-dr-eleni-chatzi.html) and [Olga Fink](https://people.epfl.ch/olga.fink?lang=en)  
[Link](https://arxiv.org/abs/2311.11961) to the arXiv version of the paper is available.

We investigate improving semi-supervised anomaly detection performance from a novel viewpoint, by generating additional pseudo-anomalies based on the limited labeled anomalies and a large amount of unlabeled data. We introduce NNG-Mix, a simple and effective pseudo-anomaly generation algorithm, that optimally utilizes information from both labeled anomalies and unlabeled data.

<img src="pics/NNG-Mix.png" width="800">
Nearest Neighbor Gaussian Mixup (NNG-Mix) makes good use of information from both labeled anomalies and unlabeled data to generate pseudo-anomalies effectively.

## Dataset
Download `Classical`, `CV_by_ResNet18`, and `NLP_by_BERT` from [ADBench](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets) and put under `datasets/` folder.

## Code

Change `--ratio 1.0` to `--ratio 0.5` or `--ratio 0.1` for training with 5% or 1% available labeled anomalies.
### Classical Dataset
<details>
<summary>Click for details...</summary>


#### Train on Classical datasets with 10% available labeled anomalies using DeepSAD
```
python NNG_Mix.py --ratio 1.0 --method nng_mix --seed 0 --alg DeepSAD --dataset Classical --nn_k 10 --nn_k_anomaly 10 --nn_mix_gaussian --nn_mix_gaussian_std 0.01 --mixup_alpha 0.2 --mixup_beta 0.2
```

#### Train on Classical datasets with 10% available labeled anomalies using MLP
```
python NNG_Mix.py --ratio 1.0 --method nng_mix --seed 0 --alg MLP --dataset Classical --nn_k 10 --nn_k_anomaly 10 --nn_mix_gaussian --nn_mix_gaussian_std 0.01 --mixup_alpha 0.2 --mixup_beta 0.2
```

</details>

### CV Dataset
<details>
<summary>Click for details...</summary>


#### Train on CV with 10% available labeled anomalies using DeepSAD
```
python NNG_Mix.py --ratio 1.0 --method nng_mix --seed 0 --alg DeepSAD --dataset CV --nn_k 10 --nn_k_anomaly 10 --nn_mix_gaussian --nn_mix_gaussian_std 0.01 --mixup_alpha 0.2 --mixup_beta 0.2
```

#### Train on CV with 10% available labeled anomalies using MLP
```
python NNG_Mix.py --ratio 1.0 --method nng_mix --seed 0 --alg MLP --dataset CV --nn_k 10 --nn_k_anomaly 10 --nn_mix_gaussian --nn_mix_gaussian_std 0.3 --mixup_alpha 0.2 --mixup_beta 0.2
```

</details>


### NLP Dataset
<details>
<summary>Click for details...</summary>


#### Train on NLP with 10% available labeled anomalies using DeepSAD
```
python NNG_Mix.py --ratio 1.0 --method nng_mix --seed 0 --alg DeepSAD --dataset NLP --nn_k 10 --nn_k_anomaly 10 --nn_mix_gaussian --nn_mix_gaussian_std 0.01 --mixup_alpha 0.2 --mixup_beta 0.2
```

#### Train on NLP with 10% available labeled anomalies using MLP
```
python NNG_Mix.py --ratio 1.0 --method nng_mix --seed 0 --alg MLP --dataset NLP --nn_k 10 --nn_k_anomaly 10 --nn_mix_gaussian --nn_mix_gaussian_std 0.3 --mixup_alpha 0.2 --mixup_beta 0.2
```

</details>

## Contact
If you have any questions, please send an email to donghaospurs@gmail.com

## Citation

If you find our work useful in your research please consider citing our paper:

```
@article{dong2023nngmix,
	author   = {Hao Dong and Ga{\"e}tan Frusque and Yue Zhao and Eleni Chatzi and Olga Fink},
	title    = {{NNG-Mix: Improving Semi-supervised Anomaly Detection with Pseudo-anomaly Generation}},
	journal  = {arXiv preprint arXiv:2311.11961},
	year     = {2023},
}
```

## Related Projects

[MultiOOD](https://github.com/donghao51/MultiOOD): Scaling Out-of-Distribution Detection for Multiple Modalities

## Acknowledgement

Many thanks to the excellent open-source projects [ADBench](https://github.com/Minqi824/ADBench).
