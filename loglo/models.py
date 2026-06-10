"""CLIP wrappers and LoGlo visual fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_LAYERS = (3, 6, 9, 12)


def _load_clip(name, attention=False):
    kwargs = {"attn_implementation": "eager"} if attention else {}
    try:
        return CLIPModel.from_pretrained(name, **kwargs)
    except TypeError:
        return CLIPModel.from_pretrained(name)


class CLIPWrapper(nn.Module):
    model_type = "clip"

    def __init__(
        self,
        clip_model=DEFAULT_CLIP_MODEL,
        layers=DEFAULT_LAYERS,
        alpha=0.3,
        attention=False,
    ):
        super().__init__()
        self.clip_model_name = clip_model
        self.clip = _load_clip(clip_model, attention=attention)
        self.clip.requires_grad_(False)
        self.layers = tuple(int(layer) for layer in layers)
        self.alpha = float(alpha)
        maximum = self.clip.config.vision_config.num_hidden_layers
        if self.layers and (
            min(self.layers) < 1
            or max(self.layers) > maximum
            or tuple(sorted(set(self.layers))) != self.layers
        ):
            raise ValueError(
                f"layers must be unique, sorted, and between 1 and {maximum}."
            )

    @property
    def projection_dim(self):
        return self.clip.config.projection_dim

    @property
    def hidden_dim(self):
        return self.clip.config.vision_config.hidden_size

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            projected = self.clip.text_projection(outputs.pooler_output)
        return F.normalize(projected, dim=-1)

    def _vision_outputs(self, pixel_values, attentions=False):
        with torch.no_grad():
            return self.clip.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=attentions,
                return_dict=True,
            )

    def _clip_image_embedding(self, vision_outputs):
        with torch.no_grad():
            projected = self.clip.visual_projection(vision_outputs.pooler_output)
        return F.normalize(projected, dim=-1)

    def fusion_embedding(self, vision_outputs):
        return None

    def encode_image(self, pixel_values):
        outputs = self._vision_outputs(pixel_values)
        original = self._clip_image_embedding(outputs)
        fusion = self.fusion_embedding(outputs)
        if fusion is None:
            return original
        return F.normalize(original + self.alpha * fusion, dim=-1)

    def forward(self, batch):
        return {
            "image_embeddings": self.encode_image(batch["pixel_values"]),
            "text_embeddings": self.encode_text(
                batch["input_ids"], batch["attention_mask"]
            ),
        }

    def checkpoint_config(self):
        return {
            "model_type": self.model_type,
            "clip_model": self.clip_model_name,
            "layers": list(self.layers),
            "alpha": self.alpha,
        }


class OriginalCLIP(CLIPWrapper):
    model_type = "clip"

    def __init__(self, clip_model=DEFAULT_CLIP_MODEL, **kwargs):
        super().__init__(clip_model=clip_model, layers=(), alpha=0.0, **kwargs)


class LinearL12Adapter(CLIPWrapper):
    model_type = "linear_l12"

    def __init__(self, clip_model=DEFAULT_CLIP_MODEL, alpha=0.3, **kwargs):
        super().__init__(clip_model=clip_model, layers=(12,), alpha=alpha, **kwargs)
        self.projector = nn.Linear(self.hidden_dim, self.projection_dim)

    def fusion_embedding(self, vision_outputs):
        feature = vision_outputs.hidden_states[12][:, 0]
        return F.normalize(self.projector(feature), dim=-1)


class StaticWeightedSum(CLIPWrapper):
    feature_type = "cls"

    def __init__(
        self,
        clip_model=DEFAULT_CLIP_MODEL,
        layers=DEFAULT_LAYERS,
        alpha=0.3,
        **kwargs,
    ):
        super().__init__(
            clip_model=clip_model, layers=layers, alpha=alpha, **kwargs
        )
        self.layer_logits = nn.Parameter(torch.zeros(len(self.layers)))
        self.projector = nn.Linear(self.hidden_dim, self.projection_dim)

    def layer_feature(self, hidden_state):
        raise NotImplementedError

    def normalized_weights(self):
        return F.softmax(self.layer_logits, dim=0)

    def fusion_embedding(self, vision_outputs):
        features = torch.stack(
            [
                self.layer_feature(vision_outputs.hidden_states[layer])
                for layer in self.layers
            ],
            dim=1,
        )
        fused = (features * self.normalized_weights()[None, :, None]).sum(dim=1)
        return F.normalize(self.projector(fused), dim=-1)


class StaticWeightedSumCLS(StaticWeightedSum):
    model_type = "weighted_sum_cls"
    feature_type = "cls"

    def layer_feature(self, hidden_state):
        return hidden_state[:, 0]


class StaticWeightedSumPatchMean(StaticWeightedSum):
    model_type = "weighted_sum_patch_mean"
    feature_type = "patch_mean"

    def layer_feature(self, hidden_state):
        return hidden_state[:, 1:].mean(dim=1)


class SelfAttentionFusion(CLIPWrapper):
    model_type = "self_attention"

    def __init__(
        self,
        clip_model=DEFAULT_CLIP_MODEL,
        layers=DEFAULT_LAYERS,
        alpha=0.3,
        num_heads=8,
        **kwargs,
    ):
        super().__init__(
            clip_model=clip_model, layers=layers, alpha=alpha, **kwargs
        )
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, batch_first=True
        )
        self.projector = nn.Linear(self.hidden_dim, self.projection_dim)

    def fusion_embedding(self, vision_outputs):
        features = torch.stack(
            [vision_outputs.hidden_states[layer][:, 0] for layer in self.layers],
            dim=1,
        )
        attended, _ = self.attention(features, features, features, need_weights=False)
        return F.normalize(self.projector(attended.mean(dim=1)), dim=-1)


class CrossAttentionFusion(CLIPWrapper):
    model_type = "cross_attention"

    def __init__(
        self,
        clip_model=DEFAULT_CLIP_MODEL,
        layers=DEFAULT_LAYERS,
        alpha=0.3,
        num_heads=8,
        **kwargs,
    ):
        super().__init__(
            clip_model=clip_model, layers=layers, alpha=alpha, **kwargs
        )
        if len(self.layers) < 2:
            raise ValueError("CrossAttentionFusion requires at least two layers.")
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, batch_first=True
        )
        self.projector = nn.Linear(self.hidden_dim, self.projection_dim)

    def fusion_embedding(self, vision_outputs):
        query = vision_outputs.hidden_states[self.layers[-1]][:, :1]
        context = torch.cat(
            [
                vision_outputs.hidden_states[layer]
                for layer in self.layers[:-1]
            ],
            dim=1,
        )
        attended, _ = self.attention(query, context, context, need_weights=False)
        return F.normalize(self.projector(attended[:, 0]), dim=-1)


MODEL_REGISTRY = {
    "clip": OriginalCLIP,
    "linear_l12": LinearL12Adapter,
    "weighted_sum_cls": StaticWeightedSumCLS,
    "weighted_sum_patch_mean": StaticWeightedSumPatchMean,
    "self_attention": SelfAttentionFusion,
    "cross_attention": CrossAttentionFusion,
}


def build_model(model_type, clip_model=DEFAULT_CLIP_MODEL, layers=DEFAULT_LAYERS, alpha=0.3):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choices: {sorted(MODEL_REGISTRY)}"
        )
    model_class = MODEL_REGISTRY[model_type]
    kwargs = {"clip_model": clip_model}
    if model_type != "clip":
        kwargs["alpha"] = alpha
    if model_type not in {"clip", "linear_l12"}:
        kwargs["layers"] = layers
    return model_class(**kwargs)
