## Tik-Tok DL Closed and Open-World Experiments

### Overview

The scripts in this directory can be used to evaluate the performance of different data 
representations (eg. direction, time, directional time) with the DeepFingerprinting model.

These experiments take [WANG14](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/wang_tao) structured plain-text data files.

The main runnable scripts are `cw_attack.py` and `ow_attack` for closed and open-world respectively.
The closed-world attack script evaluates closed world performance across several cross-validation folds.
The open-world attack script does not perform cross-validation, however examines performance when 
different thresholding values are used. 

### Requirements

* Python 3.X
  * Install required modules from the `requirements.txt`
  * Usage of python virtual environments is recommended
* CUDA 10.2 & CuDNN
* Undefended DF dataset
  * Download from: [gdrive](https://drive.google.com/file/d/1jUbKFUr048_4Zm0lLXcst-yepveFGYXS/view?usp=sharing)

### Usage Examples

Ex1. Setup virtual environment:  
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ex2. View argument help information: 
```bash
python cw_attack.py -h
```

Ex3. Execute closed-world attack using direction representation:  
```bash
python cw_attack.py -t /data/undefended/ -o df.h5 -a 0
```

Ex4. Execute open-world attack using time-only representation:  
```bash
python ow_attack.py -m /data/undefended/ -u /data/undefended_ow/ -o df.h5 -a 2
```
