from collections import defaultdict, OrderedDict
from typing import Optional, Union, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, PrefixTuningConfig,
    PeftModel
)
from torch.nn import CrossEntropyLoss
from transformers import Blip2ForConditionalGeneration, Blip2PreTrainedModel, Blip2Config, AutoConfig, Blip2VisionModel, \
    Blip2QFormerModel, AutoModelForCausalLM, MT5ForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
import os

class mBLIPModule(LightningModule):
    def __init__(self,
                 blip_pretrained="Salesforce/blip2-flan-t5-xxl",
                 blip_pretrained_checkpoint="",
                 lm_pretrained="bigscience/mt0-xl",
                 huggingface_checkpoint=None,
                 random_init_projection=True,
                 train_checkpoint=None,
                 load_8bit=True,
                 freeze_vit=True,
                 freeze_qformer=True,
                 freeze_lm=True,
                 freeze_projection=False,
                 compile=False,
                 gradient_checkpoint=False,
                 use_lora=False,
                 lora_alpha=16,
                 lora_r=8,
                 lora_bias="none",
                 lora_dropout=0.05,
                 prefix_tokens=32,
                 lora_checkpoint=None,
        ):
        super().__init__()

        if huggingface_checkpoint is not None:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                huggingface_checkpoint,
                load_in_8bit=load_8bit,
                device_map="auto"
            )

        else:
            self.model = mBLIP(blip_pretrained=blip_pretrained,
                               blip_pretrained_checkpoint=blip_pretrained_checkpoint,
                               train_checkpoint=train_checkpoint,
                               lm_pretrained=lm_pretrained,
                               random_init_projection=random_init_projection,
                               load_8bit=load_8bit,
                               freeze_vit=freeze_vit,
                               freeze_qformer=freeze_qformer,
                               freeze_lm=freeze_lm,
                               freeze_projection=freeze_projection,
                               use_lora=use_lora,
                               lora_alpha=lora_alpha,
                               lora_r=lora_r,
                               lora_bias=lora_bias,
                               lora_dropout=lora_dropout,
                               prefix_tokens=prefix_tokens,
                               lora_checkpoint=lora_checkpoint,
                               )
        if gradient_checkpoint:
            self.model.language_model.gradient_checkpointing_enable()
        if compile:  # compile currently does not work with gradient checkpoint 2.0.1
            self.model = torch.compile(self.model)

    def forward(self, mode="forward", **kwargs):
        if mode == "forward":
            return self.model(**kwargs)
        else:
            return self.generate(**kwargs)

    def generate(self, **kwargs):
        generate_kwargs = kwargs.get("generate_kwargs", dict())
        generate_kwargs.pop("stage", None) # added by Trident but must go
        return self.model.generate(kwargs["pixel_values"], kwargs["input_ids"], kwargs["attention_mask"], **kwargs["generate_kwargs"])


class mBLIP(Blip2PreTrainedModel):
    def __init__(self,
                 blip_pretrained="Salesforce/blip2-flan-t5-xxl",
                 blip_pretrained_checkpoint="",
                 train_checkpoint=None,
                 lm_pretrained="bigscience/mt0-xl",
                 random_init_projection=True,
                 load_8bit=True,
                 freeze_vit=True,
                 freeze_qformer=True,
                 freeze_lm=True,
                 freeze_projection=False,
                 use_lora=False,
                 lora_alpha=16,
                 lora_r=8,
                 lora_dropout=0.05,
                 lora_bias="none",
                 prefix_tokens=32,
                 lora_checkpoint=None
                 ):
        config = Blip2Config.from_pretrained(blip_pretrained)
        if "bloom" in lm_pretrained or "llama" in lm_pretrained:
            config.use_decoder_only_language_model = True
        config.text_config = {}
        lm_config = AutoConfig.from_pretrained(lm_pretrained)
        config.text_config = lm_config
        super().__init__(config)

        # # For debugging
        # self.tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xl")

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        checkpoint = torch.load(blip_pretrained_checkpoint, map_location="cpu")
        if random_init_projection:
            checkpoint = {k:v for k,v in checkpoint.items() if "language_projection" not in k}
        msg = self.load_state_dict(checkpoint, strict=False)
        del checkpoint
        print(msg)

        self.vision_model = self.vision_model.to("cuda")

        rank = int(os.environ.get("LOCAL_RANK", 0))
        if "bloom" in lm_pretrained or "llama" in lm_pretrained or "poly" in lm_pretrained:
            self.llm_cast_dtype = torch.bfloat16
            if isinstance(load_8bit, str) and load_8bit == "4bit":
                print("Loading in 4bit")
                self.language_model = AutoModelForCausalLM.from_pretrained(lm_pretrained,
                                                                           # low_cpu_mem_usage=True,
                                                                           # offload_state_dict=True,
                                                                           torch_dtype="auto",
                                                                           load_in_4bit=True,
                                                                           bnb_4bit_quant_type="nf4",
                                                                           bnb_4bit_use_double_quant=False,
                                                                           bnb_4bit_compute_dtype=self.llm_cast_dtype,
                                                                           device_map={"": rank},
                                                                                  )
            else:
                self.language_model = AutoModelForCausalLM.from_pretrained(lm_pretrained,
                                                                           # low_cpu_mem_usage=True,
                                                                           # offload_state_dict=True,
                                                                           torch_dtype="auto",
                                                                           load_in_8bit=load_8bit,
                                                                           device_map="auto"
                                                                           )
        elif "flan-t5" in lm_pretrained:
            self.llm_cast_dtype = torch.bfloat16
            self.language_model = T5ForConditionalGeneration.from_pretrained(lm_pretrained,
                                                                              # low_cpu_mem_usage=True,
                                                                              # offload_state_dict=True,
                                                                              torch_dtype="auto",
                                                                              load_in_8bit=load_8bit,
                                                                              device_map="auto"
                                                                             )
        else:
            self.llm_cast_dtype = torch.bfloat16
            if isinstance(load_8bit, str) and load_8bit == "4bit":
                print("Loading in 4bit")
                self.language_model = MT5ForConditionalGeneration.from_pretrained(lm_pretrained,
                                                                           # low_cpu_mem_usage=True,
                                                                           # offload_state_dict=True,
                                                                           torch_dtype="auto",
                                                                           load_in_4bit=True,
                                                                           bnb_4bit_quant_type="nf4",
                                                                           bnb_4bit_use_double_quant=True,
                                                                           bnb_4bit_compute_dtype=self.llm_cast_dtype,
                                                                           device_map={"": rank},
                                                                                  )
            else:
                self.language_model = MT5ForConditionalGeneration.from_pretrained(lm_pretrained,
                                                                           # low_cpu_mem_usage=True,
                                                                           # offload_state_dict=True,
                                                                           torch_dtype="auto",
                                                                           load_in_8bit=load_8bit,
                                                                           device_map={"": rank},
                                                                                  )
        print("LM loaded")
        if freeze_vit:
            print("Freeze ViT")
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if freeze_lm:
            print("Freeze LLM")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if freeze_qformer:
            print("Freeze QFormer")
            self.query_tokens.requires_grad = False
            for param in self.qformer.parameters():
                param.requires_grad = False

        if freeze_projection:
            print("Freeze Projection")
            for param in self.language_projection.parameters():
                param.requires_grad = False

        if use_lora:
            print("Using LoRA")
            self.language_model = prepare_model_for_int8_training(self.language_model, use_gradient_checkpointing=True)
            if lora_checkpoint:
                print("Loading LoRA adapter ", lora_checkpoint)
                self.language_model = PeftModel.from_pretrained(self.language_model, lora_checkpoint)
            else:
                target_modules = ["query_key_value"] if "bloom" in lm_pretrained else ["q", "v"]
                task = "CAUSAL_LM" if "bloom" in lm_pretrained or "poly" in lm_pretrained else "SEQ_2_SEQ_LM"
                # we save those modules extra
                # modules_to_save = []
                # if not freeze_projection:
                #     modules_to_save.append("language_projection")
                # if not freeze_qformer:
                #     modules_to_save.append("query_tokens")
                #     modules_to_save.append("qformer")
                if isinstance(use_lora, bool) or use_lora=="lora":
                    config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias=lora_bias,
                        task_type=task
                    )
                elif use_lora == "lora_all":
                    if "mt0" in lm_pretrained:
                        target_modules = ["q", ".k", "v", ".o", "wi_0", "wi_1", "wo"]
                    elif "poly" in lm_pretrained:
                        target_modules = ["c_attn", "c_proj", "c_fc"]
                    elif "bloom" in lm_pretrained:
                        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
                    config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias=lora_bias,
                        task_type=task
                    )
                elif use_lora == "prefix":
                    config = PrefixTuningConfig(
                        task_type=task,
                        num_virtual_tokens=prefix_tokens,
                    )
                self.language_model = get_peft_model(self.language_model, config)

        if train_checkpoint is not None:
            print("Loading training checkpoint", train_checkpoint)
            checkpoint = torch.load(train_checkpoint, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
                checkpoint = {k.replace("model.model.", ""): v for k, v in checkpoint.items()}

            missing, unexpected = self.load_state_dict(checkpoint, strict=False)
            print(unexpected)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        We overwrite state_dict to remove language and vision model from it. Why? Because we do not have enough RAM
        :param args:
        :param destination:
        :param prefix:
        :param keep_vars:
        :return:
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            ### SKIPPING LM and ViT FOR STATE DICT (memory reasons)
            if name in {"language_model", "vision_model"}:
                continue
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        num_images = 1
        # handle multiple input images and resize them to the correct size
        if len(pixel_values.shape) == 5:
            num_images = pixel_values.shape[1]
            pixel_values = pixel_values.view(pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:])

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        if num_images > 1:
            query_output = query_output.view(input_ids.shape[0], -1, query_output.shape[2])

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        with torch.cuda.amp.autocast(dtype=self.llm_cast_dtype):
            lm_embedding = self.language_model.get_input_embeddings()
            inputs_embeds = lm_embedding(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            expected_device = language_model_attention_mask.device
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

            if self.config.use_decoder_only_language_model:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                logits = outputs.logits if return_dict else outputs[0]
                loss = None
                # we compute the loss here since we need to take into account the sequence length of the query embeds
                if labels is not None:
                    labels = labels.to(logits.device)
                    logits = logits[:, -labels.size(1):, :]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(logits.device)

                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduction="mean")

                    loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
            else:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    labels=labels,
                )
                loss = outputs.loss if return_dict else outputs[0]
                logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        num_images = 1
        orig_batch_size = pixel_values.shape[0]
        # handle multiple input images and resize them to the correct size
        if len(pixel_values.shape) == 5:
            orig_batch_size = pixel_values.shape[0]
            num_images = pixel_values.shape[1]
            pixel_values = pixel_values.view(pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:])

        # batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state
        if num_images > 1:
            query_output = query_output.view(orig_batch_size, -1, query_output.shape[2])
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(orig_batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        attention_mask = torch.cat([language_attention_mask, attention_mask], dim=1)

        with torch.cuda.amp.autocast(dtype=self.llm_cast_dtype):
            # concatenate query embeddings with prompt embeddings
            lm_embedding = self.language_model.get_input_embeddings()
            inputs_embeds = lm_embedding(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        return outputs

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            print(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for",
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility
