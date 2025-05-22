import logging

import torch
from einops import rearrange
from torch import nn

from .vit.vit import build_vit
from .bert.builder import build_bert


logger = logging.getLogger(__name__)


class UMT(nn.Module):
    """docstring for UMT"""

    def __init__(self, config, is_pretrain=False):
        super(UMT, self).__init__()

        self.config = config
        self.is_pretrain = is_pretrain
        self.vision_width = config.vision_encoder.d_model
        self.text_width = config.text_encoder.d_model
        self.embed_dim = config.embed_dim
        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.temp)
        self.itm_head = nn.Linear(self.text_width, 2)
        self.build_umt()
    def build_umt(self):
        state_dict = torch.load(self.config.vision_encoder.pretrained, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        keep_temporal=self.config.vision_encoder.keep_temporal
        
        vision_embeds, pooled_vision_embeds, _ = self.vision_encoder(
            image, None, use_image, keep_temporal,
        )
        return vision_embeds, pooled_vision_embeds
        

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
         
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds
   

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if "vit" in encoder_name:
            vision_encoder = build_vit(self.config)
        else:
            raise ValueError(f"not implemented: {encoder_name}")        
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.text_encoder.name
        logger.info(f"Build text_encoder {encoder_name}")

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config,
                self.is_pretrain,
                False,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder
