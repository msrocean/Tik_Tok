[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tik-tok-the-utility-of-packet-timing-in/website-fingerprinting-attacks-on-website)](https://paperswithcode.com/sota/website-fingerprinting-attacks-on-website?p=tik-tok-the-utility-of-packet-timing-in)

:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Reasearch purposes ONLY. :warning: :warning: :warning:


This repository contains the code and data to demonstrate the ***Experiments*** and ***Reproduce*** the results of the **Privacy Enhancing Technologies Symposium (PETS) 2020** paper:
### Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks ([Read the Paper](https://petsymposium.org/popets/2020/popets-2020-0043.pdf))


#### Reference Format
```
@article{rahman2020tik,
  title={{Tik-Tok}: The utility of packet timing in website fingerprinting attacks},
  author={Rahman, Mohammad Saidur and Sirinam, Payap and Mathews, Nate and Gangadhara, Kantha Girish and Wright, Matthew},
  journal={Proceedings on Privacy Enhancing Technologies},
  volume={2020},
  number={3},
  pages={5--24},
  year={2020},
  publisher={Sciendo}
}
```

### Dataset
In this paper, we use **five datasets** for our experiments. 
Among those, four datasets are from previous research, and 
we have collected the Walkie-Talkie (Real) dataset. We list the datasets as follows with appropriate description and references:

1. **Undefended** [1]: Undefended dataset contains both closed-world (CW) \& open-world (OW) data, and collected in 2016.
 CW data contains 95 sites with 1000 instances each and OW data contain 40,716 sites with 1 instance each.
2. **WTF-PAD** [1]: WTF-PAD dataset contains both closed-world (CW) \& open-world (OW) data, and collected in 2016 as well.
 CW data contains 95 sites with 1000 instances each and OW data contain 40,716 sites with 1 instance each.
3. **Walkie-Talkie (Simulated)** [1]: Walkie-Talkie (Simulated) dataset contains only closed-world (CW) data, and collected in 2016 as well.
This dataset contains 100 sites with 900 instances each.
4. **Onion Sites** [2]: Onion Sites dataset contains only closed-world (CW) data, 
and collected in 2016 as well.
This dataset contains 538 sites with 77 instances each.
5. **Walkie-Talkie (Real)**: Walkie-Talkie (Real) dataset contains 100 sites with over 750 instances each.
 We collected this dataset using our implemented Walkie-Talkie prototype in 2019. 
 See the `W-T_Experiments` subdirectory for additional details.
 
 ```angular2
[1] Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018. 
Deep Fingerprinting: Undermining Website Fingerprinting Defenses 
with Deep Learning. In Proceedings of the 2018 ACM Conference on 
Computer and Communications Security (CCS). ACM.

[2] Rebekah Overdorf, Mark Juarez, Gunes Acar, Rachel Greenstadt, and Claudia
Diaz. 2017. How Unique is Your. onion?: An Analysis of the Fingerprintability
of Tor Onion Services. In Proceedings of the 2017 ACM Conference on Computer
and Communications Security (CCS). ACM.
```

### Data Representation

We have experiments with **four** types of **data representations**.
We explain each of the data representation as follows:

- **Timing Features**: Timing features consist of 160 feature values
                        (20 feature values from 8 feature categories).
                        In the model, timing features are represented as 
                        an 1-D array of [1x160].

- **Direction (D)**: We represent the direction information of an instance 
                    as a sequence of +1 and -1, **+1** representing an outgoing packet
                     and **-1** representing an incoming packet. 
                     The sequences are trimmed or padded with 0’s as needed to reach a fixed length of 5,000 packets.
                      Thus, the input forms an 1-D array of [1 x 5000].
                    
- **Raw Timing (RT**): We represent the raw timing information as a sequence of 
                        **raw timestamps** of an instance.
                        The sequences are trimmed or padded with 0’s as 
                        needed to reach a fixed length of 5,000 packets.
                        Thus, the input forms an 1-D array of [1 x 5000].
                        
- **Directional Timing (DT)**: We represent the directional timing information as 
                              a sequence of the **multiplication** of 
                              **raw timestamps** and the corresponding packet
                              **direction** (+1 (outgoing) or -1 (incoming)) of 
                              a particular packet of an instance. The sequences are trimmed or padded with 0’s as 
                        needed to reach a fixed length of 5,000 packets.
                        Thus, the input forms an 1-D array of [1 x 5000].

                              
### Reproducability of the Results

#### Dependencies & Required Packages
Please make sure you have all the dependencies available and installed before running the models.
- NVIDIA GPU should be installed in the machine, running on CPU will significantly increase time complexity.
- Ubuntu 16.04.5
- Python3-venv
- Keras version: 2.3.0
- TensorFlow version: 1.14.0
- CUDA Version: 10.2 
- CuDNN Version: 7 
- Python Version: 3.6.x 



Please install the required packages using:

```angular2
pip3 install -r requirements.txt
```

We explain the ways to reproduce each of 
experimental results one by one as the following:

#### 1. Timing Features 

- Traditional machine-learning (ML) classifier: For the experiments with 
   *k*-NN [3], SVM (CUMUL) [4], and *k*-FP [5], we refer to the classifier from the 
   respective repositories.
   ```angular2
    [3] Tao Wang, Xiang Cai, Rishab Nithyanand, Rob Johnson, and 
        Ian Goldberg. 2014. Effective attacks and provable defenses for 
        website fingerprinting. In Proceedings of the 23rd USENIX Conference 
        on Security Symposium.
    
    [4] Andriy Panchenko, Fabian Lanze, Jan Pennekamp, Thomas Engel, 
        Andreas Zinnen, Martin Henze, and Klaus Wehrle. 2016. Website 
        fingerprinting at Internet scale. In Proceedings of the 23rd Network and
        Distributed System Security Symposium (NDSS).
  
    [5] Jamie Hayes and George Danezis. 2016. k-Fingerprinting: A robust 
        scalable website fingerprinting technique. In Proceedings of the 25th 
        USENIX Conference on Security Symposium.
    ```
   
   
- Timing Features in *Deep Fingerprinting* [1] model:

    You can either 
    
     i) process raw data to get the features [(google drive url.)](https://drive.google.com/drive/folders/1k6X8PjKTXNalCiUQudx-HyqoAXVXRknL?usp=sharing), or 
     ii) use our processed data given in this [(google drive url.)](https://drive.google.com/drive/folders/13JY6QBk0Yb4D8K38oaNkZ0vGWPSYdK95?usp=sharing)
    If you are using our processed data, 
    please download the processed data and put them into the `Timing_Features/save_data/` directory.
    Please go to `Timing_Features` directory and run the following command. 
    In the place of ***dataset***, please write any of the following: 
         ***Undefended, WTF-PAD, W-T-Simulated, Onion-Sites***
    ```angular2
    python Tik_Tok_timing_features.py dataset
    ```
    Optional: We have also added a *jupyter notebook* (Tik_Tok_timing_features.ipynb) for a better interactive environment.
    
    A snippet of output for Undefended data:
    
    ```
    python Tik_Tok_timing_features.py Undefended
  
    Using TensorFlow backend.
    76000 train samples
    9500 validation samples
    9500 test samples
    Train on 76000 samples, validate on 9500 samples
    Epoch 1/100
     - 11s - loss: 4.1017 - acc: 0.0593 - val_loss: 2.9626 - val_acc: 0.1926
    Epoch 2/100
     - 7s - loss: 2.9497 - acc: 0.1976 - val_loss: 2.4673 - val_acc: 0.3026
    
    .....
  
   Epoch 99/100
     - 7s - loss: 0.3103 - acc: 0.9109 - val_loss: 0.7414 - val_acc: 0.8216
    Epoch 100/100
     - 7s - loss: 0.3096 - acc: 0.9104 - val_loss: 0.7639 - val_acc: 0.8239
    
    Testing accuracy: 0.843284285
    ```

#### 2. Closed and Open-world Experiments w/ Deep Fingerprinting

See the `DL_Experiments` directory for the scripts used to perform the Direction, Raw Timing, and Directional Timing experiments.

#### 3. W-T Prototype Experiments

Our W-T crawling software and instructions can be downloaded as a zip file from the following link: [gdrive](https://drive.google.com/file/d/1eMLzy0L83wCV5mFf9wEN4p-BMhgxXqmr/view?usp=sharing)

The scripts used to evaluate the dataset and related instructions are found in the `W-T_Experiments` subdirectory.

#### 4. Information Leakage Analysis:
For information leakage analysis, we refer to our re-implemented 
github repository of WeFDE: [https://github.com/notem/reWeFDE.](https://github.com/notem/reWeFDE) 

#### 5. Congestion Analysis
See the `Congestion_Analysis` directory for the scripts used to perform the experiments with the instances of `slow circuits as test set` and instances of `fast circuits as test set`.
We processed the data to feed into model. Please create a sub-directory named `datasets` inside the 
`Congestion_Analysis` directory. Download the data from this google drive [url.](https://drive.google.com/drive/folders/18dYNAq8bbkgG3XWy3wSRDpI9-pTbWhMA)
Extract the downloaded files to `datasets` sub-directory.

Parameters:
- `--congestion` : choices = ['slow', 'fast']\
                            **slow**: Instances of Slow cirtuits as test set.\
                            **fast**: Instances of fast circuits as test set.)
- `--dataset` : choices=['Undefended', 'WTF-PAD', 'Onion-Sites']
- `--data_rep` : choices = ['D', 'RT', 'DT']\
               Type of data representation to be used.\
               **D**: direction, **RT**: Raw Timing, and **DT**: Directional Timing
                
Example of Usage:\
    ```
    python Tik_Tok_Congestion.py --congestion slow --dataset Undefended --data_rep D 
    ```



### Questions, Comments, & Feedback
Please, address any questions, comments, or feedback to the authors of the paper.
The main developers of this code are:
 
* Mohammad Saidur Rahman ([saidur.rahman@mail.rit.edu](mailto:saidur.rahman@mail.rit.edu)) 
* Nate Mathews ([nate.mathews@mail.rit.edu](mailto:nate.mathews@mail.rit.edu))
* Payap Sirinam ([payap_siri@rtaf.mi.th](mailto:payap_siri@rtaf.mi.th))
* Kantha Girish Gangadhara ([kantha.gangadhara@mail.rit.edu](mailto:kantha.gangadhara@mail.rit.edu))
* Matthew Wright ([matthew.wright@rit.edu](mailto:matthew.wright@rit.edu))


### Acknowledgements
We thank the anonymous reviewers for their helpful feedback. We give special thanks to Tao Wang for providing details about the technical implementation of the W-T defense, and to Marc Juarez for providing guidelines on developing the W-T prototype. This material is based upon work supported in part by the **National Science Foundation (NSF)** under Grants No. **1722743** and **1816851**.
