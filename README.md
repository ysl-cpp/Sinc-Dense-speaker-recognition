
# SincNet
SincNet is a neural architecture for processing **raw audio samples**. It is a novel Convolutional Neural Network (CNN) that encourages the first convolutional layer to discover more **meaningful filters**. SincNet is based on parametrized sinc functions, which implement band-pass filters.

In contrast to standard CNNs, that learn all elements of each filter, only low and high cutoff frequencies are directly learned from data with the proposed method. This offers a very compact and efficient way to derive a **customized filter bank** specifically tuned for the desired application. 

On the basis of SINC filter, this project adopts Densenet to carry out deeper feature interaction, and obtains better effect than the original Sincnet


## Prerequisites
- Linux
- Python 3.6/2.7
- pytorch 1.0
- pysoundfile (``` conda install -c conda-forge pysoundfile```)
- We also suggest using the anaconda environment.



## How to run a TIMIT experiment
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

**1. Run TIMIT data preparation.**

This step is necessary to store a version of TIMIT in which start and end silences are removed and the amplitude of each speech utterance is normalized. To do it, run the following code:

``
python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored
- *data_lists/TIMIT_all.scp* is the list of the TIMIT files used for training/test the speaker id system.

**Run the speaker id experiment.**

- Modify the *[data]* section of *cfg/SincNet_TIMIT.cfg* file according to your paths. In particular, modify the *data_folder* with the *$OUTPUT_FOLDER* specified during the TIMIT preparation. The other parameters of the config file belong to the following sections:
 1. *[windowing]*, that defines how each sentence is split into smaller chunks.
 2. *[cnn]*,  that specifies the characteristics of the CNN architecture.
 3. *[dnn]*,  that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
 4. *[class]*, that specify the softmax classification part.
 5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

- Once setup the cfg file, you can run the speaker id experiments using the following command:

``
python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
``

The network might take several hours to converge (depending on the speed of your GPU card). In our case, using an *nvidia Tesla V100*, the full training took about 24 hours. 