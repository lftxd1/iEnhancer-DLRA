## Indroduction

Welcome to the iEnhancer-RLA code repository. iEnhancer-RLA is a deep learning method based on a self-attentive mechanism to fuse local and global features of sequences to identify and classify enhancers.



## Performance

The experimental results on the independent test dataset indicate that iEnhancer-RLA performs better than nine existing state-of-the-art methods in both identification and classification of enhancers in  all almost metrics.

**Enhancer identifier :** 

| Method           | ACC (%)      | SP (%)       | SN (%)       | MCC          |
| ---------------- | ------------ | ------------ | ------------ | ------------ |
| EnhancerPred     | 74.00      | 74.50      | 73.50      | 0.480      |
| iEnhancer-2L     | 73.00      | 75.00      | 71.00      | 0.460      |
| iEnhancer-EL     | 74.75      | 78.50      | 71.00      | 0.496      |
| iEnhancer-5Step  | 82.30     | 83.50      | 81.10      | 0.650      |
| iEnhancer-ECNN   | 76.90      | 75.20      | 78.50      | 0.537      |
| iEnhancer-XG     | 75.75      | 77.50      | 74.50      | 0.515      |
| iEnhancer-EBLSTM | 77.20      | 79.50      | 75.50      | 0.272      |
| iEnhancer-GAN    | 78.40      | 75.80      | 81.10      | 0.567     |
| iEnhancer-RD     | 78.80      | 76.50      | 81.00      | 0.576      |
| iEnhancer-RLA    | **93.72** | **90.45** | **97.00** | **0.876** |
| Improvement      | +13.8%     | +8.3%      | +19.7%     | +34.7%     |

**Enhancer classifier :** 

| Method           | ACC (%)      | SP (%)  | SN (%)       | MCC          |
| ---------------- | ------------ | ------- | ------------ | ------------ |
| EnhancerPred     | 55.00      | 65.00 | 45.00      | 0.102      |
| iEnhancer-2L     | 60.50      | 74.00 | 47.00     | 0.218      |
| iEnhancer-EL     | 61.00      | 68.00 | 54.00     | 0.222     |
| iEnhancer-5Step  | 63.50     | 74.00 | 53.00     | 0.280     |
| iEnhancer-ECNN   | 67.80     | 56.40 | 79.10     | 0.368     |
| iEnhancer-XG     | 63.50     | 57.00 | 70.00     | 0.272     |
| iEnhancer-EBLSTM | 65.80     | 53.60 | 81.20     | 0.324     |
| iEnhancer-GAN    | 74.90     | 53.70 | 96.10     | 0.505     |
| iEnhancer-RD     | 70.50     | 57.00 | 84.00     | 0.426     |
| iEnhancer-RLA    | **84.40** | 70.00 | **98.80** | **0.718** |
| Improvement      | +12.6%       | -5.4%   | +2.8%        | +42.1%       |



## Environment requirements

1. Python 3.7+
2. Tensorflow 2.0+



## Usage

```bash
git clone https://github.com/lftxd1/iEnhancer-RLA.git
cd iEnhancer-RLA
pip install -r requirements.txt
python benchmark_identifier.py
python benchmark_classifier.py
python rice_identifier.py 
```

