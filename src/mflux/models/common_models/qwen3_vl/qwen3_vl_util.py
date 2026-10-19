import mlx.core as mx
import numpy as np

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder


class Qwen3VLUtil:
    @staticmethod
    def sample_top_p(
        logits: mx.array,
        top_p: float,
        temperature: float = 1.0,
    ) -> mx.array:
        # Greedy decoding: temperature <= 0 takes the argmax instead of sampling. Dividing by a
        # zero temperature below would produce NaNs, so caption routes pass 0.0 for determinism.
        if temperature <= 0:
            return mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)
        if temperature != 1.0:
            logits = logits / temperature
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        force_keep_first = mx.arange(probs.shape[-1]) == 0
        sorted_keep = (cumulative_probs <= top_p) | force_keep_first
        filtered_probs = mx.where(sorted_keep, sorted_probs, mx.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)
        sorted_token_id = mx.random.categorical(mx.log(filtered_probs), axis=-1)
        return sorted_indices[sorted_token_id].astype(mx.int32)

    @staticmethod
    def build_multimodal_position_ids(
        input_ids: mx.array,
        image_grid_thw: mx.array,
        spatial_merge_size: int,
        image_token_id: int,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """3-axis (t, h, w) RoPE positions for a prompt that mixes text and image tokens.

        Qwen3-VL numbers image tokens by their (t, h, w) grid coordinate inside the merged patch
        grid and advances the running text position by `max(llm_grid_h, llm_grid_w)` after each
        image, instead of treating the whole sequence as one flat line. Caption/image-understanding
        quality depends on this: with a flat arange the spatial layout of the image region is lost.
        Returns an int32 array shaped (3, batch, seq_len).
        """
        ids = np.asarray(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        batch_size, seq_len = ids.shape
        if attention_mask is None:
            mask = np.ones((batch_size, seq_len), dtype=np.int32)
        else:
            mask = np.asarray(attention_mask)
            if mask.ndim == 1:
                mask = mask[None, :]
        grids = np.asarray(image_grid_thw).reshape(-1, 3).tolist()

        position_rows = np.zeros((3, batch_size, seq_len), dtype=np.int32)
        grid_index = 0
        for batch_idx in range(batch_size):
            active = [idx for idx in range(seq_len) if int(mask[batch_idx, idx]) == 1]
            token_types = [1 if int(ids[batch_idx, idx]) == image_token_id else 0 for idx in active]
            rows = [[], [], []]
            current_position = 0
            start_idx = 0
            while start_idx < len(token_types):
                modality = token_types[start_idx]
                end_idx = start_idx + 1
                while end_idx < len(token_types) and token_types[end_idx] == modality:
                    end_idx += 1

                if modality == 0:
                    length = end_idx - start_idx
                    for axis in range(3):
                        rows[axis].extend(range(current_position, current_position + length))
                    current_position += length
                else:
                    if grid_index >= len(grids):
                        raise ValueError("Image token groups exceed the provided image_grid_thw entries.")
                    grid_t, grid_h, grid_w = (int(value) for value in grids[grid_index])
                    llm_h = grid_h // spatial_merge_size
                    llm_w = grid_w // spatial_merge_size
                    expected = grid_t * llm_h * llm_w
                    if expected != end_idx - start_idx:
                        raise ValueError(
                            "Image token group length does not match image_grid_thw: "
                            f"{end_idx - start_idx} tokens for a grid-derived {expected}."
                        )
                    for time_idx in range(grid_t):
                        for height_idx in range(llm_h):
                            for width_idx in range(llm_w):
                                rows[0].append(current_position + time_idx)
                                rows[1].append(current_position + height_idx)
                                rows[2].append(current_position + width_idx)
                    current_position += max(llm_h, llm_w)
                    grid_index += 1

                start_idx = end_idx

            if any(len(row) != len(active) for row in rows):
                raise ValueError("Multimodal position construction produced an invalid sequence length.")
            for local_idx, source_idx in enumerate(active):
                for axis in range(3):
                    position_rows[axis, batch_idx, source_idx] = rows[axis][local_idx]

        if grid_index != len(grids):
            raise ValueError("image_grid_thw entries exceed the image token groups.")
        return mx.array(position_rows, dtype=mx.int32)

    @staticmethod
    def generate_text(
        decoder: Qwen3VLDecoder,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        max_new_tokens: int = 4096,
        top_p: float = 0.9,
        temperature: float = 0.2,
        stop_token_sequences: list[list[int]] | None = None,
        eos_token_id: int | None = None,
        seed: int | None = None,
    ) -> mx.array:
        if stop_token_sequences is None:
            stop_token_sequences = []
        if eos_token_id is not None:
            stop_token_sequences.append([eos_token_id])

        batch_size = input_ids.shape[0]

        # Avoid unnecessary conversion if already mx.array
        if isinstance(input_ids, mx.array):
            prompt_ids = input_ids
        else:
            prompt_ids = mx.array(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids).astype(mx.int32)

        # Set random seed for deterministic generation if provided
        if seed is not None:
            mx.random.seed(seed)

        # Pre-compute max stop sequence length (constant, don't recompute every iteration)
        max_stop_len = 0
        if stop_token_sequences:
            max_stop_len = max(len(seq) for seq in stop_token_sequences)

        # Pre-allocate reusable arrays for efficiency
        ones_1x1 = mx.ones((batch_size, 1), dtype=mx.int32)
        max_cache_length = int(prompt_ids.shape[1] + max_new_tokens)

        # Initialize KV cache
        past_key_values = None
        generated_token_columns: list[mx.array] = []
        generated_token_history: list[list[int]] = [[] for _ in range(batch_size)]
        current_input_ids = prompt_ids

        for iteration in range(max_new_tokens):
            # On first iteration: process full sequence
            # On subsequent iterations: only process new token (use cache)
            if past_key_values is None:
                # First iteration: process full input sequence
                decoder_input_ids = prompt_ids
                decoder_attention_mask = attention_mask
            else:
                # Subsequent iterations: only process the new token
                decoder_input_ids = current_input_ids
                decoder_attention_mask = ones_1x1  # Reuse pre-allocated array

            # Forward pass with KV cache
            # Only pass pixel_values and image_grid_thw on first iteration (they're part of the input sequence)
            decoder_kwargs = {
                "input_ids": decoder_input_ids,
                "attention_mask": decoder_attention_mask,
                "use_cache": True,
                "past_key_values": past_key_values,
                "max_cache_length": max_cache_length,
            }
            if past_key_values is None:
                # First iteration: include image data if present
                if pixel_values is not None and image_grid_thw is not None:
                    decoder_kwargs["pixel_values"] = pixel_values
                    decoder_kwargs["image_grid_thw"] = image_grid_thw

            decoder_output = decoder(**decoder_kwargs)

            if isinstance(decoder_output, tuple):
                logits, past_key_values = decoder_output
                # Evaluate KV cache to ensure it's computed and not accumulating
                if past_key_values is not None:
                    for kv_tuple in past_key_values:
                        mx.eval(kv_tuple[0], kv_tuple[1])
            else:
                logits = decoder_output
                past_key_values = None

            # Evaluate logits before sampling
            mx.eval(logits)

            # Get logits for the last position
            next_token_logits = logits[:, -1, :]
            next_tokens = []
            for i in range(batch_size):
                token_id = Qwen3VLUtil.sample_top_p(next_token_logits[i], top_p, temperature)
                next_tokens.append(token_id)

            next_tokens = mx.stack(next_tokens, axis=0)
            next_tokens = mx.expand_dims(next_tokens, axis=1)
            generated_token_columns.append(next_tokens)
            current_input_ids = next_tokens
            sampled_token_ids = next_tokens[:, 0].tolist()
            for i, token_id in enumerate(sampled_token_ids):
                generated_token_history[i].append(int(token_id))

            # Check for stop token sequences (only check last max_stop_len tokens for efficiency)
            should_stop = False
            matched_stop_sequence = None
            if stop_token_sequences and max_stop_len > 0:
                last_tokens_history = generated_token_history[0]
                if len(last_tokens_history) >= max_stop_len:
                    last_tokens_np = last_tokens_history[-max_stop_len:]
                    for stop_sequence in stop_token_sequences:
                        seq_len = len(stop_sequence)
                        if len(last_tokens_np) >= seq_len:
                            tokens_to_check = last_tokens_np[-seq_len:]
                            if tokens_to_check == stop_sequence:
                                should_stop = True
                                matched_stop_sequence = stop_sequence
                                break

            if should_stop:
                # Exclude the stop sequence from the output (match PyTorch behavior)
                if matched_stop_sequence is not None:
                    seq_len = len(matched_stop_sequence)
                    if seq_len > 0:
                        generated_token_columns = generated_token_columns[:-seq_len]
                        for history in generated_token_history:
                            del history[-seq_len:]
                break

        if not generated_token_columns:
            return prompt_ids

        generated_suffix = mx.concatenate(generated_token_columns, axis=1)
        return mx.concatenate([prompt_ids, generated_suffix], axis=1)
