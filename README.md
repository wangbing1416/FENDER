# FENDER

This repo is the official code of our work _Variety Is the Spice of Life: Detecting Fake News with Dynamic Contextual Representations_

This code acknowledges and follows to _Generalizing to the Future: Mitigating Entity Bias in Fake News Detection_

### Requirements
```shell
cudatoolkit==11.8.0
torch==1.13.1
numpy==1.23.5
```

### Train

1. Download datasets from [GITHUB LINK](https://github.com/ICTMCG/ENDEF-SIGIR2022), and place them to 
`./FENDER_en/data` and `./FENDER_ch/data`, respectively;
2. Run the scripts `./FENDER_en/run.sh` or `./FENDER_ch/run.sh` to rerun the experiments, or you can directly run
```shell
python main.py --para_len 32 --seed 1000
```

### Output
The log files are saved in `./logs/param`, and the checkpoints can be loaded from `./para_model`.

### Acknowledgement

We acknowledge the source code provided by [GITHUB LINK](https://github.com/ICTMCG/ENDEF-SIGIR2022).

### Citation
NONE