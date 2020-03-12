## W-T Prototype Evaluation

### Overview

This script can be used to evaluate our Walkie-Talkie prototype dataset against different 
data representations (e.g. direction, time, directional time) with the DF model.

The main runnable script is `wt_attack.py`, and must be run using python on the commandline with arguments.
This script was designed to work specifically with our W-T prototype dataset. If you want to run the 
standard DF CW and OW experiments, use the scripts in the other directory instead.

### Requirements

* Python 3.X
  * Install required modules from the `requirements.txt`
  * Usage of python virtual environments are recommended
* CUDA 10.2 & CuDNN
* W-T proto dataset
  * Download from: [gdrive](https://drive.google.com/file/d/1TUv43I9E3Av1JwraB5mapId4WQ4_NQ4C/view?usp=sharing)

### Usage Examples

Ex1. Setup virtual environment:  
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ex2. View argument help information:  
```bash
python wt_attack.py -h
```

Ex3. Execute attack using directional time representation:  
```bash
python wt_attack.py -t /data/wt_proto/ -o dt.h5 -a 1
```

### W-T Dataset Information

The root of the W-T dataset contains several subdirectories whose names are of the form 
`{ID}-{type}` where `{ID}` identifies the crawl ID and `{type}` is either `normal` or `inverse`.
Each subdirectory contains the data samples captured from a crawl cycle. The `normal`-type crawls
contain data samples in which the true website is one of the monitored sites. The `inverse`-type 
crawls contains the reverse pairings of (i.e. true website is unmonitored) for sample pairs 
captured in the `normal`-type crawl of the same crawl ID.

Each crawl subdirectory contains one or more additional directories. Each of these subdirectories 
represent a real crawl instance, some crawls may contain several instances as the crawler may 
occasionally error and require that it be restarted from where it left off.
The filenames for samples in the dataset are of the form `{BatchNo.}_{TrueSiteName}_{CaptureID}`.
When loading `inverse`-type dataset as unmonitored, the true site names are ignored (the batch 
number and capture ids are ignored for both monitored and unmonitored data loading).
