
import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.roberta import (RobertaModel,
                                         RobertaConfig,
                                         RobertaPreTrainedModel,
                                        )
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaLMHead
from transformers.configuration_utils import PretrainedConfig


from typing import Callable, Optional, Union, Any, List, Tuple, Iterable

class XLMRobertaForSeq2SeqConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        encoder_layers=12,
        decoder_layers=12,
        # encoder_num_hidden_layers=12,
        # encoder_num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_length=128,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=False,
        classifier_dropout=None,
        decoder_start_token_id=None,                     #写在初始构造config的时候
        share_encoder_decoder_embeddings=True,
        **kwargs):
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                         intermediate_size=intermediate_size, hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob,
                         attention_probs_dropout_prob=attention_probs_dropout_prob, max_position_embeddings=max_position_embeddings, type_vocab_size=type_vocab_size,
                         initializer_range=initializer_range, layer_norm_eps=layer_norm_eps, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         position_embedding_type=position_embedding_type, use_cache=use_cache, classifier_dropout=classifier_dropout, **kwargs)
        self.is_encoder_decoder = True
        self.share_encoder_decoder_embeddings = share_encoder_decoder_embeddings
        self.max_length = max_length
        self.decoder_start_token_id = self.eos_token_id if decoder_start_token_id else decoder_start_token_id

        

    def get_encoder_config(self):
        config_dict = self.to_dict()
        config_dict['num_hidden_layers'] = self.encoder_layers
        config_dict['is_encoder_decoder'] = False
        config_dict['is_decoder'] =False
        return RobertaConfig(**config_dict)
    
    def get_decoder_config(self):
        config_dict = self.to_dict()
        config_dict['is_decoder'] = True
        config_dict['add_cross_attention'] = True
        config_dict['decoder_start_token_id'] = self.eos_token_id
        config_dict['num_hidden_layers'] = self.decoder_layers
        return RobertaConfig(**config_dict)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class XLMRobertaForSeq2Seq(RobertaPreTrainedModel):

    def __init__(self, config: XLMRobertaForSeq2SeqConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.encoder_config = config.get_encoder_config()
        self.decoder_config = config.get_decoder_config()
        self.encoder = RobertaModel(self.encoder_config, add_pooling_layer=False)
        self.decoder = RobertaModel(self.decoder_config, add_pooling_layer=False)

        self.lm_head = RobertaLMHead(self.encoder_config)

        if self.config.share_encoder_decoder_embeddings:
            self.decoder.embeddings = self.encoder.embeddings

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] =None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,     
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ## TODO 要在作为decoder的模型里设为False
        use_cache = False
        
        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self.prepare_decoder_inputids_from_labels(labels)
            else:
                decoder_input_ids = input_ids

        ## ! encoder 输出
        if encoder_outputs is None:             # encoder 的输出，用于推理的时候自回归解码时
            encoder_outputs = self.encoder(input_ids, attention_mask, head_mask=head_mask,
                                           output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(input_ids = decoder_input_ids, 
            attention_mask = labels_attention_mask,
            encoder_hidden_states = encoder_outputs[0],
            encoder_attention_mask = attention_mask,
            past_key_values = past_key_values,
            head_mask = decoder_head_mask,
        )

        prediction_scores = self.lm_head(decoder_outputs[0])

        ## TODO 计算loss
        lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), 
                               labels.view(-1))

        if not return_dict:
            return (lm_loss, prediction_scores, decoder_outputs.hidden_states)
        
        return Seq2SeqLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            decoder_hidden_states=decoder_outputs.hidden_states,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    
    def prepare_decoder_inputids_from_labels(self, labels: torch.Tensor):
        """
        将<eos>设为解码开始的token
        decoder_attention_mask 不用变
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def can_generate(self):
        return True
    

if __name__ == "__main__":
    # from transformers import AutoModelForMaskedLM
    # print("  shdfoiasdnoifi")
    # pretrained_model = AutoModelForMaskedLM.from_pretrained("microsoft/xlm-align-base")
    # # bin_path = torch.save(pretrained_model.state_dict(), "./patameters.pth")
    # config = XLMRobertaForSeq2SeqConfig(**pretrained_model.config.to_dict())
    # model = XLMRobertaForSeq2Seq(config)
    # model_state_dict = torch.load('./patameters.pth')
    # model.lm_head.load_state_dict(model_state_dict)
    # print(model)
    pass