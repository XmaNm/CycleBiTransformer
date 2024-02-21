# CycleBiTransformer
this is the implementation CycleBiTransformer
# Train CycleBiTransformer on medical image:
## Training First Stage for codebook
(optional) Configure Hyperparameters in training_vqgan.py
Set path to dataset in training_vqgan.py
python training_vqgan.py

## Training Second Stage for image synthesis
(optional) Configure Hyperparameters in training_transformer.py
Set path to dataset in training_transformer.py
python training_transformer.py
