# Generating the data

We detail the steps to reproduce our [merge_train.py](..%2F..%2Fdata%2Fpretrain%2Fmerge_train.py)multilingual instruction mix and evaluation data.
Note that most scripts have to be updated with your path to the raw data used by the scripts.

## License
Use of the data has to comply with the licenses of the original datasets used to generate this data.

Translations are produced with [NLLB](https://huggingface.co/facebook/nllb-200-distilled-1.3B) so use has to comply with
their license.

* MSCOCO: [CC BY 4.0 for annotations, Flickr Terms of Use for images](https://cocodataset.org/#termsofuse)
* BLIP captions (Web CapFilt): [BSD 3](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt)
* LLaVA: [CC BY NC 4.0](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K). It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use
* VQAv2: [CC BY 4.0](https://visualqa.org/terms.html)
* A-OKVQA [Apache 2.0](https://github.com/allenai/aokvqa)
* ImageNet: [Non-Commercial](https://www.image-net.org/download), Babel-ImageNet: [BabelNet Non-Commercial License](https://babelnet.org/full-license)


## Training Instruction Mix
You will need to download the respective 'raw' data from the websites of MSCOCO (including images), A-OKVQA, LLaVA, ImageNet and the [BLIP captions](https://github.com/salesforce/BLIP).

### BLIP Web CapFilt
1. Run in [pretrain](data/pretrain) `filter.py` and `download_images.py` to sample captions from the full data and download the images.
2. Run `generate_train.py` to generate a English intermediate file, `translate_train.py`to generate the translations, and `generate_train.py` again for the final data file.

As exactly reproducing our sampling is impossible due to randomness, we include our result after step 1 [here](TODO).
This file also includes image URL which you can use to download the images. As of 06.2023, all links were still available.

#### Caption Matching
To generate the image-caption matching data, first run `hard_match.py` to generate the English examples (this takes a while),
`translate_match_train.py`to generate the translations, and then `generate_match_train.py` to produce the final file.

### Other tasks
Run `generate_train.py` once to generate an intermediate file, `translate_train.py`to generate the translations, and `generate_train.py` again for the final data files.

The translation step is not needed for A-OKVQA and the second `generate_train.py` is not needed for LLaVA.


For ImageNet examples, you need the label file from https://github.com/gregor-ge/Babel-ImageNet.

### Combine
Run `pretrain/merge_train.py` to combine the different files into one task mix file.



## Evaluation
Note for captioning: Both XM3600 and xFlickrCo also generate files used by the pycocoeval library
for evaluation - those files contain `coco` in their name.


### IGLUE (xGQA, XVNLI, MaRVL, xFlickrCo)
Download the data from the IGLUE repository along with the images and run the scripts in the folders.


### XM3600 and MaXM
Download the raw data and images from the CrossModal3600 and MaXM repositories and run the 
respective scripts in the folders.

### POPE and CHAIR
Clone the POPE repository (https://github.com/AoiDragon/POPE) and then run the respective scripts.