# Semantic Information Oriented No-Reference Video Quality Assessment

## Description
SIONR code for the following papers:

- Wei Wu, Qinyao Li, Zhenzhong Chen, Shan Liu. Semantic Information Oriented No-Reference Video Quality Assessment.

## Feature Extraction
```
python generate_CNNfeatures.py
```

## Test Demo
The model weights provided in `model/SIONR.pt` are the saved weights when running a random split of KoNViD-1k. The random split is shown in [data/train_val_test_split.xlsx](https://github.com/lorenzowu/SIONR/blob/master/data/train_val_test_split.xlsx), which contains video file names, scores, and train/validation/test split assignment (random).
```
python test_demo.py
```
The test results are shown in [result/test_result.xlsx](https://github.com/lorenzowu/SIONR/blob/master/result/test_result.xlsx).

## NR-VQA
|    Model   | Download            | Paper             |
|:------------:|:-------------------:|:-------------------:|
| TLVQM       | [nr-vqa-consumervideo](https://github.com/jarikorhonen/nr-vqa-consumervideo) | [Korhenen et al. TIP'19](https://ieeexplore.ieee.org/document/8742797)
