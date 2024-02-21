# CycleBiTransformer
this is the implementation CycleBiTransformer
# Train CycleBiTransformer on medical image:

## Training First Stage for codebook

1.(optional) Configure Hyperparameters in ```training_vqgan.py```

2.Set path to dataset in ```training_vqgan.py```

3.python ```training_vqgan.py```

## Training Second Stage for image synthesis

1.(optional) Configure Hyperparameters in ```training_transformer.py```

2.Create the file  'C:\Users\dome\datasets\images' to save training images.

3.Set path to dataset in ```training_transformer.py```

4.python ```training_transformer.py```
