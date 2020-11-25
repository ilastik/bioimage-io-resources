# bioimage-io-models
Ilastik models for the bioimage.io model zoo

## How to add an ilastik model

- Prepare your model for the bioimage.io format
  - Write a model.yaml that describes your model following the [bioimage.io configuration specification](https://github.com/bioimage-io/configuration). 
  See also [the example pytorch model.yaml](https://github.com/bioimage-io/pytorch-bioimage-io/blob/master/specs/models/unet2d/nuclei_broad/UNet2DNucleiBroad.model.yaml).
  - Host your model.yaml on github.
  - Upload your model weights. Note that the weights must be stored as [state dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended). We recommend to host the weights on zenodo.
  - Host the zip containing your model.yaml and the other necessary files.
- Make a pull request to [the ilastik model repo](https://github.com/ilastik/bioimage-io-models) to add your model
  - In the pull request, add your model to the [`model` list](https://github.com/ilastik/bioimage-io-models/blob/master/manifest.bioimage.io.yaml#L24). In the list enry, you need to specify
    - `id`: the identifier of your model
    - `source`: where the model configuration file is hosted
    - `links`: the bioimage.io apps that can read this model (this is usually just `Ilastik`)
    - `download_url`: where the zipped model files are hosted

If you don't know how to proceed for any of these steps, please open an issue here or make a incomplete pull request and ask for help finishing it.

<!--
TODO:
- explain (or provide link) how to use the model checker locally to make sure the model works
- include a script with model checker that makes the model zip
- slightly unrelated: how do we specify arbitrary shapes in xy: https://github.com/hci-unihd/batchlib/blob/master/misc/models/torch/bioimageio/UNetCovidIf.model.yaml#L25
-->
