:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Reasearch purposes ONLY. :warning: :warning: :warning:


This repository contains the code and data to demonstrate the ***Experiments*** and ***Reproduce*** the results of the **Privacy Enhancing Technologies Symposium (PETS) 2020** paper:
### Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks ([Read the Paper](https://arxiv.org/pdf/1902.06421.pdf))


#### Reference Format
```
@inproceedings{rahman2019tik,
  title={{Tik-Tok}: The Utility of Packet Timing in Website Fingerprinting Attacks},
  author={Rahman, Mohammad Saidur and Sirinam, Payap and Mathews, Nate 
          and Gangadhara, Kantha Girish and Wright, Matthew},
  journal={Proceedings on Privacy Enhancing Technologies (PETS)},
  volume={2020},
  number={3},
  year={2020},
  publisher={Sciendo}
}
```

### Dataset
In this paper, we use **five datasets** for our experiments. 
Among those, four datasets are from previous research, and 
we collect another dataset. We list the datasets as follows with appropriate references:

1. **Undefended** [1]: Undefended dataset contains both closed-world (CW) \& open-world (OW) data, and collected in 2016.
 CW data contains 95 sites with 1000 instances each and OW data contain 40,716 sites with 1 instance each.
2. **WTF-PAD** [1]: WTF-PAD dataset contains both closed-world (CW) \& open-world (OW) data, and collected in 2016.
 CW data contains 95 sites with 1000 instances each and OW data contain 40,716 sites with 1 instance each.
3. **Walkie-Talkie (Simulated)** [1]: Walkie-Talkie (Simulated) dataset contains only closed-world (CW) data, and collected in 2016.
This dataset contains 100 sites with 900 instances each.
4. **Onion Sites** [2]: Onion Sites dataset contains only closed-world (CW) data, 
and collected in 2016.
This dataset contains 538 sites with 77 instances each.
5. **Walkie-Talkie (Real)**: Walkie-Talkie (Real) dataset contains 100 sites with over 750 instances each.
 This dataset is collected using our implemented Walkie-Talkie prototype in 2019. 
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

- **Timing Features**: Timing features consists of 160 feature values
                        (20 feature values from 8 feature categories).
                        In the model, timing features are represented as 
                        an 1-D array of [1x160].

- **Direction (D)**: We represent the direction information of an instance 
                    as a sequences of +1 and -1, **+1** representing an outgoing packet
                     and **-1** representing an incoming packet. 
                     The sequences are trimmed or padded with 0’s as need to reach a fixed length of 5,000 packets.
                      Thus, the input forms an 1-D array of [1 x 5000].
                    
- **Raw Timing (RT**): We represent the raw timing information as a sequence of 
                        **raw timestamps** of an instance.
                        The sequences are trimmed or padded with 0’s as 
                        need to reach a fixed length of 5,000 packets.
                        Thus, the input forms an 1-D array of [1 x 5000].
                        
- **Directional Timing (DT)**: We represent the directional timing information as 
                              a sequence of the **multiplication** of 
                              **raw timestamps** and the corresponding packet
                              **direction** (+1 (outgoing) or -1 (incoming)) of 
                              a particular packet of an instance.

                              
### Reprodcability of the Results

#### Dependencies & Required Packages
Please make sure you have all the dependencies available and installed before running the models.
- NVIDIA GPU should be installed in the machine, running on CPU will signifcantly increase time complexity.
- Ubuntu 16.04.5
- Keras version: 2.2.2
- TensorFlow version: 1.10.0
- CUDA Version: 10.0 
- CuDNN Version: 7 
- Python Version: 3.6.9 

Please install the required packages using:

```angular2
pip3 install -r requirements.txt
```

We have in total **seven** sets of **experiments**. We explain the ways to reproduce each of 
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
    
     i) process raw data to get the features, or 
     ii) use our processed data given in this [google drive url.](https://drive.google.com/drive/folders/13JY6QBk0Yb4D8K38oaNkZ0vGWPSYdK95?usp=sharing)
    If you are using our processed data, 
    please download the processed data and put them into the `Timing_Features/save_data/` directory.
    Please go to `Timing_Features` directory and run the following command. 
    In the place of ***dataset***, please write any of the following: 
         ***Undefended, WTF-PAD, W-T-Simulated, Onion-Sites***
    ```angular2
    python Tik_Tok_timing_features.py dataset
    ```
    Optional: We have also added a *jupyter notebook* (Tik_Tok_timing_features.ipynb) for a better interactive environment.
    
    A snipet of output for Undefended data:
    
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
For information leakge analysis, we refer to our re-implemented 
github repository of WeFDE: [https://github.com/notem/reWeFDE.](https://github.com/notem/reWeFDE) 

#### 5. Congestion Analysis




### Questions, Comments, & Feedback
Please, address any questions, comments, or feedback to the authors of the paper.
The main developers of this code are:
 
* Mohammad Saidur Rahman ([saidur.rahman@mail.rit.edu](mailto:saidur.rahman@mail.rit.edu)) 
* Nate Mathews ([nate.mathews@mail.rit.edu](mailto:nate.mathews@mail.rit.edu))
* Payap Sirinam ([payap_siri@rtaf.mi.th](mailto:payap_siri@rtaf.mi.th))
* Kantha Girish Gangadhara ([kantha.gangadhara@mail.rit.edu](mailto:kantha.gangadhara@mail.rit.edu)
* Matthew Wright ([matthew.wright@rit.edu](mailto:matthew.wright@rit.edu)) 
