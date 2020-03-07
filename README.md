:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Reasearch purposes ONLY. :warning: :warning: :warning:


This repository contains the code and data to demonstrate the ***Experiments*** and ***Reproduce*** the results of the **PETS 2020** paper:
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
5. **Walkie-Talkie (Real)**: Walkie-Talkie (Real) dataset contains 100 sites with 750 instances each.
 This dataset is collected using our implemented Walkie-Talkie prototype in 2019.
 As this dataset size is more than the limit of *github*, we share the 
  the dataset in a google drive and the dataset can be downloaded from this [url](). 
  ***nate*** please describe the dataset.
 
 ```angular2
[1] Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018. 
DeepFingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.
In Proceedings of the 2018 ACM Conference on Computer and Communications Security (CCS). ACM

[2] Rebekah Overdorf, Mark Juarez, Gunes Acar, Rachel Greenstadt, and Claudia
Diaz. 2017. How Unique is Your. onion?: An Analysis of the Fingerprintability
of Tor Onion Services. In Proceedings of the 2017 ACM Conference on Computer
and Communications Security (CCS). ACM.
```

### Data Representation

We have experiments with **four** types of **data representation**.
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

                              

