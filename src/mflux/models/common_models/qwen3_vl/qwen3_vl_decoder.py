import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_attention import Qwen3VLKVCache
from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.common_models.qwen3_vl.qwen3_vl_rope import Qwen3VLRotaryEmbedding


class Qwen3VLDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: list[int] | None = None,
        attention_scaling: float = 1.0,
        image_token_id: int = 151655,
        visual=None,
        multimodal_position_ids: bool = False,
        deepstack: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        if mrope_section is None:
            mrope_section = [24, 20, 20]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=mrope_section,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=attention_scaling,
            mrope_section=mrope_section,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.visual = visual
        self.image_token_id = image_token_id
        # Both default off so the FIBO-VLM / Qwen-Image-2.1 text-encoder paths keep their exact
        # previous numerics; caption routes turn them on because Qwen3-VL was trained with them.
        self.multimodal_position_ids = multimodal_position_ids
        self.deepstack = deepstack
        # Per-row offset between flat token indices and the prompt's multimodal positions. After an
        # image the running position advances by max(llm_h, llm_w) instead of by the image token
        # count, so cached decode steps must continue from where the prompt stopped (HF's
        # `mrope_position_deltas`) rather than from the flat sequence length.
        self.mrope_position_delta: mx.array | None = None

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        use_cache: bool = False,
        past_key_values: list[Qwen3VLKVCache] | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, list[Qwen3VLKVCache]]:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        image_positions = (input_ids == self.image_token_id) if self.image_token_id is not None else None

        # Handle vision inputs if provided
        deepstack_image_embeds = None
        if pixel_values is not None and image_grid_thw is not None and self.visual is not None:
            image_embeds_split, deepstack_image_embeds = self._get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_embeds_split, axis=0)

            if image_positions is not None:
                n_image_tokens = int(mx.sum(image_positions).item())

                if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                    # Splice image rows into the flattened token sequence, in mask order.
                    inputs_embeds = Qwen3VLDecoder._replace_image_token_embeddings(
                        inputs_embeds=inputs_embeds,
                        image_positions=image_positions,
                        image_embeds=image_embeds.astype(inputs_embeds.dtype),
                    )

        # Create attention mask with causal masking (from QwenEncoder)
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        # Position embeddings
        if use_cache and past_key_values is not None:
            cached_seq_len = past_key_values[0][2] if len(past_key_values) > 0 else 0
            cache_position = mx.arange(cached_seq_len, cached_seq_len + seq_len, dtype=mx.int32)
            position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
            position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
            if self.mrope_position_delta is not None:
                delta = self.mrope_position_delta[:batch_size].reshape(1, -1, 1)
                position_ids = position_ids + delta
        elif self.multimodal_position_ids and image_grid_thw is not None and image_positions is not None:
            # Local import: qwen3_vl_util imports this module for type hints.
            from mflux.models.common_models.qwen3_vl.qwen3_vl_util import Qwen3VLUtil

            position_ids = Qwen3VLUtil.build_multimodal_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=int(getattr(self.visual, "spatial_merge_size", 2)),
                image_token_id=self.image_token_id,
                attention_mask=attention_mask,
            )
            self.mrope_position_delta = Qwen3VLDecoder._mrope_position_delta(
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        else:
            self.mrope_position_delta = None
            cache_position = mx.arange(seq_len, dtype=mx.int32)
            position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
            position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))

        # Create causal + padding mask
        mask_dtype = inputs_embeds.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        # Optimize: when seq_len=1 (cached iterations), causal mask is trivial (all zeros)
        if seq_len == 1:
            causal_tri_mask = mx.zeros((batch_size, 1, 1, 1), dtype=mask_dtype)
        else:
            idx = mx.arange(seq_len, dtype=mx.int32)
            j = mx.expand_dims(idx, axis=0)
            i = mx.expand_dims(idx, axis=1)
            tri_bool = j > i
            zeros_2d = mx.zeros((seq_len, seq_len), dtype=mask_dtype)
            neginf_2d = mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype)
            causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
            causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
            causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_tri_mask + padding_mask
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Qwen3-VL adds the deepstack merger outputs back onto the image-token rows after each of
        # the first len(deepstack_visual_indexes) decoder layers. Off by default (see __init__).
        deepstack_layers = None
        deepstack_offsets = None
        deepstack_mask = None
        if self.deepstack and deepstack_image_embeds:
            deepstack_mask = image_positions.reshape(-1)
            deepstack_offsets = mx.cumsum(deepstack_mask.astype(mx.int32), axis=0) - 1
            n_image_tokens = int(mx.sum(deepstack_mask).item())
            if n_image_tokens > 0:
                deepstack_offsets = mx.clip(deepstack_offsets, 0, n_image_tokens - 1)
                deepstack_layers = [embeds.astype(inputs_embeds.dtype) for embeds in deepstack_image_embeds]
            else:
                deepstack_mask = None

        present_key_values = [] if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            layer_past = None
            if use_cache and past_key_values is not None:
                layer_past = past_key_values[layer_idx] if layer_idx < len(past_key_values) else None

            layer_output = layer(
                hidden_states,
                attention_mask_4d,
                position_embeddings,
                past_key_value=layer_past,
                max_cache_length=max_cache_length,
            )

            if use_cache:
                hidden_states, layer_present = layer_output
                present_key_values.append(layer_present)
            else:
                hidden_states = layer_output

            if deepstack_layers is not None and layer_idx < len(deepstack_layers):
                hidden_states = Qwen3VLDecoder._add_deepstack_embeddings(
                    hidden_states=hidden_states,
                    flat_positions=deepstack_mask,
                    offsets=deepstack_offsets,
                    deepstack_embeds=deepstack_layers[layer_idx],
                )

        # Final norm
        hidden_states = self.norm(hidden_states)
        if isinstance(self.lm_head, nn.QuantizedLinear):
            # 4-bit checkpoints publish a packed lm_head (weight/scales/biases); the float32
            # matmul below cannot read the packed weight, so quantized heads go through the module.
            logits = self.lm_head(hidden_states)
        else:
            hidden_states_f32 = hidden_states.astype(mx.float32)
            weight_f32 = self.lm_head.weight.astype(mx.float32)
            logits_f32 = mx.matmul(hidden_states_f32, weight_f32.T)
            logits = logits_f32.astype(hidden_states.dtype)

        if use_cache:
            return logits, present_key_values
        return logits

    @staticmethod
    def _mrope_position_delta(position_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """Per-row `last multimodal position + 1 - flat token count` (HF's mrope_position_deltas)."""
        mask = attention_mask.astype(mx.float32)
        masked_positions = mx.where(mask[None] == 1, position_ids, -mx.ones_like(position_ids))
        row_max = mx.max(masked_positions, axis=(0, 2))
        row_length = mx.sum(mask, axis=1)
        return (row_max + 1 - row_length).astype(mx.int32)

    @staticmethod
    def _replace_image_token_embeddings(
        inputs_embeds: mx.array,
        image_positions: mx.array,
        image_embeds: mx.array,
    ) -> mx.array:
        """Replace image-token rows with image embeddings, in flattened mask order."""
        if image_embeds.shape[0] == 0:
            return inputs_embeds

        flat_positions = image_positions.reshape(-1)
        flat_inputs = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        offsets = mx.cumsum(flat_positions.astype(mx.int32), axis=0) - 1
        offsets = mx.clip(offsets, 0, image_embeds.shape[0] - 1)
        picked = image_embeds[offsets]
        replaced = mx.where(mx.expand_dims(flat_positions, axis=-1), picked, flat_inputs)
        return replaced.reshape(inputs_embeds.shape)

    @staticmethod
    def _add_deepstack_embeddings(
        hidden_states: mx.array,
        flat_positions: mx.array,
        offsets: mx.array,
        deepstack_embeds: mx.array,
    ) -> mx.array:
        """Add one deepstack merger output onto the image-token rows only."""
        rows = deepstack_embeds[offsets]
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        contribution = mx.where(mx.expand_dims(flat_positions, axis=-1), rows, mx.zeros_like(rows))
        return (flat_hidden + contribution).reshape(hidden_states.shape)

    def _get_image_features(
        self, pixel_values: mx.array, image_grid_thw: mx.array
    ) -> tuple[list[mx.array], list[mx.array] | None]:
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        # Vision model can handle the original dtype (usually float16)
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values, image_grid_thw, return_deepstack=self.deepstack
        )
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        spatial_merge_size = self.visual.spatial_merge_size
        split_sizes = (original_split_sizes // (spatial_merge_size**2)).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        split_sizes = [s for s in split_sizes if s > 0]

        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        return image_embeds_split, deepstack_image_embeds
