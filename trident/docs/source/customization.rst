.. |project| replace:: trident
.. _project: https://www.github.com/fdschmidt93/trident/

.. _customization:

Customization
=============

.. _add-path:

Adding your project to path
---------------------------

You can add entire projects or single python files to the environment of trident by providing your paths as `config.imports` as follows:

You most typically would have the following folder structure in your project `my_project`, where the file `my_model.py` has your :obj:`MyModel` you want to train.

.. code-block::

    my_project/
    └── project_code
        ├── __init__.py
        ├── __pycache__
        │   └── __init__.cpython-39.pyc
        └── src
            ├── __init__.py
            └──── my_model.py

For all intents and purposes, `MyModel` is yet another light wrapper around `Pytorch-Lightning <https://pytorch-lightning.readthedocs.io/>`_ and `transformers <https://huggingface.co/transformers/>`_ for a sequence classification task like MNLI.

.. code-block:: python

    class MyModel(LightningModule):
        def __init__( self, pretrained_model_name_or_path: str, num_labels: int):

            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path, num_labels=num_labels
            )

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

This allows you to later on natively instantiate your models as if they were part of the project, either via a dedicated separate `imports` config 

:obj:`/configs/imports/my_project.yaml`

.. code-block:: yaml

    # imports: Union[str, list[str]]
    - /home/my_user/.../my_project

or directly in your `experiment` config.

.. code-block:: bash
    
    # you could alternatively write a new `experiment`
    imports:
      - /home/my_user/.../my_project

    module:
      model: 
        # and then import your model as expected
        _target_: project_code.src.my_model.MyModel
        pretrained_model_name_or_path: "prajjwal1/bert-tiny"
        num_labels: 3

The paths passed to `config.imports` are prepended to your :obj:`sys.path`:

.. code-block:: python

    for path in config.imports:
        sys.path.insert(0, path)

You can find the full example experiment :repo:`here <configs/experiment/mnli_my_model.yaml>`\.

**Reminder:** you can combine importing your code with any of the below mechanisms to extend |project|\.


.. _link-function:

Linking your own function
-------------------------

Most often, you would link your own functions to customize the evaluation loop. The below example is a common pattern in customized evaluation.

.. code-block:: yaml

    # see also /configs/evaluation/sequence_classification.yaml
    apply:
      batch: null
      outputs: 
        # required _target_ for hydra.utils.instantiate
        _target_: src.utils.hydra.partial
        # actual function
        _partial_: src.evaluation.classification.get_preds
      step_outputs: null

:obj:`partial` is a wrapper around :obj:`functools.partial`.

.. _function-override:

Function Overrides
------------------

You can override functions of the model and datamodule explicitly as follows.

1. Write your custom function the project or python path (see :ref:`add-path`)
2. Provide a yaml configuration in `/configs/overrides/my_datamodule_overrides.yaml` like below

    .. code-block:: yaml
    
        setup: # name of function to override
          _target_: src.utils.hydra.partial # leverage partial
          _partial_: src.utils.hydra.setup_my_dataset # path to function

3. Add the override to your model or datamodule like, for instance, in `/configs/datamodules/my_datamodule.yaml`:     

    .. code-block:: yaml

        _target_: src.datamodules.base.BaseDataModule
        _recursive_: false
         
        defaults:
        - /collate_fn: my_collator
        # option 1: more applicable to extending TridentModules
        - /overrides: my_datamodule

        # option 2: a single fixed-case override can be concisely expressed
        overrides:
            setup: # name of function to override
              _target_: src.utils.hydra.partial # leverage partial
              _partial_: src.utils.hydra.setup_my_dataset # path to function
        
        batch_size: ???
        num_workers: ???
        pin_memory: ???
        seed: ${seed} # linked against global option

The most common use cases to override existing functions are:

a. Provide your own datamodule for :obj:`src.datamodules.base.BaseDataModule`
b. Override existing or add functions to :obj:`src.modules.base.TridentModule`

Should your dataset fit the project, please consider a PR!

.. _evaluation:

Evaluation
----------

The evaluation mixin diminishes the boilerplate when writing custom evaluation loops for custom models. The below example is an annotated variant of :repo:`sequence classification <configs/evaluation/sequence_classification.yaml>` (see also, :repo:`tatoeba <configs/evaluation/tatoeba.yaml>` for sentence translation retrieval).

The configuration separates on a high level into:

* **apply**: transformation functions applied to `batch`, `outputs`, and `step_outputs`
* **step_outputs**: what keys of (default: complete `batch` and `outputs`)
* **metric**: configure how to instantiate and compute your metric

.. code-block:: yaml

    # apply transformation function 
    apply:
      batch: null
      outputs:   
        _target_: src.utils.hydra.partial
        _partial_: src.evaluation.classification.get_preds

      step_outputs: null  # on flattened outputs of what's collected from steps

    # Which keys/attributes are supposed to be collected from `outputs` and `batch`
    # for {val, test} loop end
    step_outputs:
      outputs: "preds" # can be a str
      batch: # or a list[str]
        - labels

    # metrics config
    metrics:
      # name of the metric used eg for logging
      accuracy:
        # instructions to instantiate metric, preferrably torchmetrics.Metric
        metric:
          _target_: torchmetrics.Accuracy

        # either on_step: true or on_epoch: true
        on_step: true # torchmetrics compute on_step!

        # either on_step: true or on_epoch: true
        compute: 
          # function_argument: "from:key"
          # ... for `preds` of `torchmetrics.Accuracy` get `preds` from `outputs`
          preds: "outputs:preds"
          # ... for `targets` of `torchmetrics.Accuracy` get `labels` from `batch`
          target: "batch:labels"

      f1:
        metric:
          _target_: torchmetrics.F1
        on_step: true
        compute:
          preds: "outputs:preds"
          target: "batch:labels"


where `get_preds` is defined as follows: 

.. code-block:: python
    
    def get_preds(outputs):
        outputs.preds = outputs.logits.argmax(dim=-1)
        return outputs
