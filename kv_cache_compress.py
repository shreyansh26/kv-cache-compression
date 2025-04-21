import torch
import torch.nn as nn

class KVCacheAttentionSink(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16, global_tokens=2, sliding_window=8):
        super().__init__()

        self.sliding_window = sliding_window
        self.global_tokens = global_tokens
        self.max_cache_size = sliding_window + global_tokens

        cache_shape = (max_batch_size, n_heads, self.max_cache_size, head_dim)
        pos = torch.full((max_batch_size, 1, self.max_cache_size), -1, dtype=torch.long)

        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('pos', pos)
    
    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        total_len = input_pos.shape[0]
        assert total_len == k_val.shape[2]

        # Prefill
        if total_len > 1:
            if total_len <= self.max_cache_size:
                self.pos[:, :, input_pos] = input_pos
                self.k_cache[:, :, input_pos] = k_val
                self.v_cache[:, :, input_pos] = v_val
            else:
                # Reduce prompt size when initial prompt is longer than max cache size
                # Always keep the first `global_tokens` tokens and the last `sliding_window` tokens.
                global_idxs = torch.arange(self.global_tokens, device=input_pos.device)
                recent_idxs = torch.arange(total_len - self.sliding_window, total_len, device=input_pos.device)
                keep_idxs = torch.cat([global_idxs, recent_idxs])
                new_pos = input_pos[keep_idxs]

                self.pos = new_pos.unsqueeze(0).unsqueeze(0).expand_as(self.pos)
                self.k_cache = k_val.index_select(dim=2, index=keep_idxs)
                self.v_cache = v_val.index_select(dim=2, index=keep_idxs)
        # Decode
        else:
            idx_to_pop = (self.global_tokens + torch.argmin(self.pos[:, :, self.global_tokens :], dim=-1)).flatten()
            self.pos[:, :, idx_to_pop] = input_pos.long()
            self.k_cache[:, :, idx_to_pop] = k_val
            self.v_cache[:, :, idx_to_pop] = v_val

        return self.k_cache, self.v_cache

    def compression_ratio(self, seq_len):
        return ((seq_len - self.max_cache_size) / seq_len)

    def get_cache_stats(self, seq_len):
        stats = {}
        for layer_idx, layer in enumerate(self.layers):
            stats[f"compression_ratio_{layer_idx}"] = layer.attention.kv_cache.compression_ratio(seq_len)
        stats["compression_ratio"] = sum(stats.values()) / len(stats)
        return stats

class KVCacheL2Norm(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16, keep_ratio=1.0, prune_after=1024):
        super().__init__()

        self.keep_ratio = keep_ratio
        self.prune_after = prune_after
        self.max_cache_size = min(prune_after, max_seq_length)
        self.curr_len = 0

        cache_shape = (max_batch_size, n_heads, self.max_cache_size, head_dim)
        pos = torch.full((max_batch_size, n_heads, self.max_cache_size), -1, dtype=torch.long)

        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('pos', pos)
    
    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        total_len = input_pos.shape[0]
        assert total_len == k_val.shape[2]

        # Prefill
        if total_len > 1:
            if total_len <= self.max_cache_size:
                self.pos[:, :, input_pos] = input_pos
                self.k_cache[:, :, input_pos] = k_val
                self.v_cache[:, :, input_pos] = v_val
                self.curr_len = total_len
            else:
                # Reduce prompt size when initial prompt is longer than max cache size
                # Always keep the first `global_tokens` tokens and the last `sliding_window` tokens.
                key_norm = torch.norm(k_val, p=2, dim=-1)
                key_norm_diff = key_norm.max() - key_norm
                scoring_priority = key_norm_diff.masked_fill(self.pos == -1, float('inf'))
                scoring_sorted_idx = torch.argsort(scoring_priority, dim=-1) # [B, H, S] -> in ascending order
                num_toks_to_remove = total_len - self.max_cache_size
                scoring_sorted_idx_selcted = scoring_sorted_idx[num_toks_to_remove:]
                self.k_cache = k_val.index_select(dim=2, index=scoring_sorted_idx_selcted)
                self.v_cache = v_val.index_select(dim=2, index=scoring_sorted_idx_selcted)
                self.pos = input_pos[scoring_sorted_idx_selcted]
                self.curr_len = self.max_cache_size
        # Decode
        else:
            if self.curr_len == self.max_cache_size:
                key_norm = torch.norm(self.k_cache, p=2, dim=-1)
                key_norm_diff = key_norm.max() - key_norm
                scoring_priority = key_norm_diff.masked_fill(self.pos == -1, float('inf'))
                scoring_sorted_idx = torch.argsort(scoring_priority, dim=-1) # [B, H, S] -> in ascending order
                num_toks_to_remove = int((1 - self.keep_ratio) * self.max_cache_size)
                scoring_sorted_idx_selcted = scoring_sorted_idx[:, :, num_toks_to_remove:]
                
                # Use gather instead of index_select
                self.k_cache = torch.cat([
                    torch.gather(self.k_cache, dim=2, index=scoring_sorted_idx_selcted.unsqueeze(-1).expand(-1, -1, -1, self.k_cache.shape[-1])),
                    torch.zeros(self.k_cache.shape[0], self.k_cache.shape[1], num_toks_to_remove, self.k_cache.shape[-1], device=self.k_cache.device, dtype=self.k_cache.dtype)
                ], dim=2)
                
                self.v_cache = torch.cat([
                    torch.gather(self.v_cache, dim=2, index=scoring_sorted_idx_selcted.unsqueeze(-1).expand(-1, -1, -1, self.v_cache.shape[-1])),
                    torch.zeros(self.v_cache.shape[0], self.v_cache.shape[1], num_toks_to_remove, self.v_cache.shape[-1], device=self.v_cache.device, dtype=self.v_cache.dtype)
                ], dim=2)
                
                self.pos = torch.cat([
                    torch.gather(self.pos.expand(-1, scoring_sorted_idx_selcted.shape[1], -1), dim=2, index=scoring_sorted_idx_selcted),
                    torch.ones(self.pos.shape[0], scoring_sorted_idx_selcted.shape[1], num_toks_to_remove, device=self.pos.device, dtype=self.pos.dtype) * -1
                ], dim=2)
                self.curr_len = self.max_cache_size - num_toks_to_remove

            if self.curr_len < self.max_cache_size:
                self.k_cache[:, :, [self.curr_len]] = k_val
                self.v_cache[:, :, [self.curr_len]] = v_val
                self.pos[:, :, [self.curr_len]] = input_pos.long()
                self.curr_len += 1

        return self.k_cache, self.v_cache

    def compression_ratio(self, seq_len):
        compressed_effective = self.max_cache_size * (1 + self.keep_ratio) / 2
        compressed_actual = self.max_cache_size
        return ((seq_len - compressed_effective) / seq_len), ((seq_len - compressed_actual) / seq_len)

    def get_cache_stats(self, seq_len):
        stats_effective = {}
        stats_actual = {}
        for layer_idx, layer in enumerate(self.layers):
            stats_effective[f"compression_ratio_effective_{layer_idx}"], stats_actual[f"compression_ratio_actual_{layer_idx}"] = layer.attention.kv_cache.compression_ratio(seq_len)
        stats_effective["compression_ratio_effective"] = sum(stats_effective.values()) / len(stats_effective)
        stats_actual["compression_ratio_actual"] = sum(stats_actual.values()) / len(stats_actual)
        return stats_effective, stats_actual


class KVCacheSnapKV(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16, window_size=8, compress_length=40, kernel_size=5):
        super().__init__()
        self.window_size = window_size
        self.compress_length = compress_length
        self.max_cache_size = max_seq_length
        self.kernel_size = kernel_size
        
        cache_shape = (max_batch_size, n_heads, self.max_cache_size, head_dim)
        pos = torch.full((max_batch_size, n_heads, self.max_cache_size), -1, dtype=torch.long)

        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('pos', pos)
        self.ctr = 0

        self.pool = torch.nn.AvgPool1d(
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
    
    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        total_len = input_pos.shape[0]
        assert total_len == k_val.shape[2]

        self.pos[:, :, input_pos] = input_pos.long()    
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val

        # Count number of prefill tokens
        if total_len > 1:
            self.ctr = total_len

        return self.k_cache, self.v_cache

    def prune_cache(self, input_pos: torch.Tensor, attn_scores: torch.Tensor, n_kv_head: int):
        total_len = input_pos.shape[0]
        head_dim = self.k_cache.shape[-1]
        b, nh, lq, lk = attn_scores.shape
        n_rep = nh // n_kv_head
        
        if total_len > self.compress_length:
            attn_scores_grouped = attn_scores.view(b, n_kv_head, n_rep, lq, lk)
            attn_scores_agg = attn_scores_grouped.sum(dim=2) # Sum over query heads sharing KV head. Shape: [B, n_kv_heads, L_obs, L_prefix]
            attn_weights_sum = attn_scores_agg[:, :, -self.window_size:, : total_len-self.window_size].sum(dim=-2)
            attn_cache = self.pool(attn_weights_sum)

            indices = attn_cache.topk(self.compress_length - self.window_size, dim=-1).indices
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            pos_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, 1).squeeze(-1)

            k_past_compress = self.k_cache[:, :, :total_len-self.window_size, :].gather(dim = 2, index = indices_expanded)
            v_past_compress = self.v_cache[:, :, :total_len-self.window_size, :].gather(dim = 2, index = indices_expanded)
            pos_compress = self.pos[:, :, :total_len-self.window_size].gather(dim = 2, index = pos_expanded)
            k_cur = self.k_cache[:, :, total_len-self.window_size:total_len, :]
            v_cur = self.v_cache[:, :, total_len-self.window_size:total_len, :]
            pos_cur = self.pos[:, :, total_len-self.window_size:total_len]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            pos_states = torch.cat([pos_compress, pos_cur], dim = 2)

            self.k_cache[:, :, torch.arange(self.compress_length), :] = key_states
            self.v_cache[:, :, torch.arange(self.compress_length), :] = value_states
            self.pos[:, :, torch.arange(self.compress_length)] = pos_states

    def compression_ratio(self, seq_len):
        compressed_effective = self.max_cache_size - self.ctr + self.compress_length
        compressed_actual = self.max_cache_size
        return abs((seq_len - compressed_effective) / seq_len), abs((seq_len - compressed_actual) / seq_len)

    def get_cache_stats(self, seq_len):
        stats_effective = {}
        stats_actual = {}
        for layer_idx, layer in enumerate(self.layers):
            stats_effective[f"compression_ratio_effective_{layer_idx}"], stats_actual[f"compression_ratio_actual_{layer_idx}"] = layer.attention.kv_cache.compression_ratio(seq_len)
        stats_effective["compression_ratio_effective"] = sum(stats_effective.values()) / len(stats_effective)
        stats_actual["compression_ratio_actual"] = sum(stats_actual.values()) / len(stats_actual)
        return stats_effective, stats_actual


class KVCacheHeavyHitter(nn.Module):
    """
    Adapted from https://github.com/AnswerDotAI/cold-compress/blob/15bb3005c5661e0d97e295c71c6e5fced1a863cc/cache.py#L615
    """
    def __init__(self, max_batch_size, max_cache_size, n_heads, head_dim, dtype=torch.bfloat16, history_window_size=1, attn_thresholding=False):
        super().__init__()

        self.max_cache_size = max_cache_size
        self.history_window_size = history_window_size
        self.attn_thresholding = attn_thresholding
        self.curr_len = 0
        self.n_heads = n_heads

        cache_shape = (max_batch_size, n_heads, self.max_cache_size, head_dim)
        pos_shape = (max_batch_size, n_heads, self.max_cache_size) # Head specific positions

        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('pos', torch.full(pos_shape, -1, dtype=torch.long))

        # Attention History Buffers (Mimicking KVCacheHeavyHitter)
        history_num_shape = (
            max_batch_size,
            n_heads,
            self.max_cache_size,
            self.history_window_size,
        )
        history_denom_shape = (max_batch_size, n_heads, self.max_cache_size)
        history_num_dtype = (
            torch.bool
            if self.attn_thresholding
            else torch.float64 # Use float64 for accumulation if history_window_size=1
            if self.history_window_size == 1
            else dtype
        )
        self.register_buffer("attn_history_num", torch.zeros(history_num_shape, dtype=history_num_dtype))
        self.register_buffer("attn_history_denom", torch.zeros(history_denom_shape, dtype=torch.int32))
        self.register_buffer("attn_counter", torch.zeros((max_batch_size, n_heads), dtype=torch.int64)) # Track per head/batch

    def _calculate_eviction_scores(self):
        """Calculates importance scores based on attention history."""
        # Identify the token with consistently "lowest" attention
        # Ensure float division
        numerator = self.attn_history_num.sum(dim=-1).float()

        if (self.history_window_size == 1):
            # We use the full history (there is no clamping around a fixed window)
            denominator = self.attn_history_denom.clamp_min(1)
        else:
            # The denominator is the number of times this token's history has been recorded
            # We only record most self.history_window_size recent scores so need to clamp it
            denominator = self.attn_history_denom.clamp(min=1, max=self.history_window_size)

        avg_attn = numerator / denominator # Lower score means less important

        # Evict unfilled slots first by giving them lowest score
        scores = avg_attn.masked_fill(self.pos == -1, float('-inf'))

        return scores # Higher score = More important

    def update_attn_history(self, attn: torch.Tensor):
        """
        Updates the attention history buffers.
        attn shape: [B, H, 1, S_cache] - Attention from current query to KVs in cache.
        """
        assert attn.shape[-1] <= self.max_cache_size, f"Attention scores ({attn.shape[-1]}) exceed cache size ({self.max_cache_size})"
        assert attn.shape[0] == self.k_cache.shape[0]
        
        b, nh, lq, lk = attn.shape
        n_rep = nh // self.n_heads

        attn_scores_grouped = attn.view(b, self.n_heads, n_rep, lq, lk)
        attn = attn_scores_grouped.sum(dim=2) # Sum over query heads sharing KV head. Shape: [B, n_kv_heads, L_obs, L_prefix]

        assert attn.shape[1] == self.n_heads

        batch_size = attn.shape[0]
        attn_len = attn.shape[-1]

        # attn comes in as [B, H, 1, S_cache], needs to be [B, H, S_cache, 1] for history
        processed_attn = attn.squeeze(2) # Shape: [B, H, S_cache]

        # Apply thresholding if needed
        if self.attn_thresholding:
            # Simplified threshold: assume uniform attention over current cache content
            # A more accurate threshold might consider only valid cache entries (pos != -1)
            threshold = 1.0 / self.pos.ne(-1).sum(dim=-1, keepdim=True).clamp_min(1) # Shape [B, H, 1]
            processed_attn = (processed_attn >= threshold).to(self.attn_history_num.dtype)
        else:
            processed_attn = processed_attn.to(self.attn_history_num.dtype)

        processed_attn = processed_attn.unsqueeze(-1) # Shape: [B, H, S_cache, 1]

        # Get indices for the circular buffer
        history_idx = (self.attn_counter % self.history_window_size).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, attn_len, 1) # Shape: [B, H, S_cache, 1]

        if self.history_window_size == 1:
            # Accumulate if window size is 1
            update_indices = torch.zeros_like(history_idx) # Always index 0
            # Pad attention update if needed
            padded_attn = torch.zeros_like(self.attn_history_num[:,:,:,0:1]) # Target shape: [B, H, max_cache_size, 1]
            padded_attn[:,:,:attn_len,:] = processed_attn
            self.attn_history_num.scatter_add_(3, update_indices, padded_attn)

        else:
            # Overwrite if window size > 1
            update_indices = history_idx
            # Pad attention update if needed
            padded_attn = torch.zeros_like(self.attn_history_num[:,:,:,0:1]) # Target shape: [B, H, max_cache_size, 1]
            padded_attn[:,:,:attn_len,:] = processed_attn
            self.attn_history_num.scatter_(3, update_indices, padded_attn)

        # Increment denominator only for positions that received attention
        denom_update_mask = torch.zeros_like(self.attn_history_denom) # Shape: [B, H, max_cache_size]
        denom_update_mask[:, :, :attn_len] = 1
        self.attn_history_denom += denom_update_mask.int()

        self.attn_counter += 1

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D], v_val: [B, H, S, D]
        batch_size, n_heads, seq_len, head_dim = k_val.shape
        assert seq_len == input_pos.shape[0]
        assert n_heads == self.n_heads
        device = k_val.device

        # Prefill
        if seq_len > 1:
            # Simple prefill: Fill cache up to max_cache_size. Overwrite if longer.
            fill_len = min(seq_len, self.max_cache_size)
            fill_indices = torch.arange(fill_len, device=device)
            # Use slicing for prefill as it's contiguous
            self.k_cache[:, :, :fill_len] = k_val[:, :, :fill_len]
            self.v_cache[:, :, :fill_len] = v_val[:, :, :fill_len]
            # Broadcast input_pos across heads for head-specific pos buffer
            self.pos[:, :, :fill_len] = input_pos[:fill_len].view(1, 1, fill_len).expand(batch_size, n_heads, fill_len)
            self.curr_len = fill_len
            # Reset attention history for prefill
            self.attn_history_num.zero_()
            self.attn_history_denom.zero_()
            self.attn_counter.zero_()

        # Decode
        else:
            assert seq_len == 1
            input_pos_val = input_pos.item() # Single value scalar

            if self.curr_len < self.max_cache_size:
                # Cache not full, append to the end using slicing
                fill_idx = self.curr_len
                self.k_cache[:, :, fill_idx:fill_idx+1] = k_val # k_val is [B, H, 1, D]
                self.v_cache[:, :, fill_idx:fill_idx+1] = v_val # v_val is [B, H, 1, D]
                self.pos[:, :, fill_idx] = input_pos_val
                # Initialize attention history for the new token (already zero from potential prefill reset or init)
                # self.attn_history_num[:, :, fill_idx, :].zero_() # Already zero
                # self.attn_history_denom[:, :, fill_idx].zero_() # Already zero
                # Counter is updated in update_attn_history

                self.curr_len += 1
            else:
                # Cache is full, evict least important token
                scores = self._calculate_eviction_scores() # Shape: [B, H, max_cache_size]
                # Argmin finds the *least* important token (lowest score)
                eviction_idx = torch.argmin(scores, dim=-1) # Shape: [B, H]

                # Expand eviction_idx for scatter operations
                # Target shape for K/V cache scatter: [B, H, 1, D]
                eviction_idx_kv = eviction_idx.view(batch_size, n_heads, 1, 1).expand(-1, -1, 1, head_dim)
                # Target shape for Pos cache scatter: [B, H, 1]
                eviction_idx_pos = eviction_idx.view(batch_size, n_heads, 1)

                # Scatter new K, V, and Pos values into the evicted slots
                self.k_cache.scatter_(2, eviction_idx_kv, k_val)
                self.v_cache.scatter_(2, eviction_idx_kv, v_val)
                # Need to broadcast input_pos_val for scatter
                input_pos_tensor = torch.tensor([input_pos_val], device=device, dtype=self.pos.dtype).view(1, 1, 1).expand(batch_size, n_heads, 1)
                self.pos.scatter_(2, eviction_idx_pos, input_pos_tensor)

                # Reset the attention history for the evicted slots (now filled with new token)
                # Target shape for attn_history_num scatter: [B, H, 1, history_window_size]
                eviction_idx_hist_num = eviction_idx.view(batch_size, n_heads, 1, 1).expand(-1, -1, 1, self.history_window_size)
                zeros_hist_num = torch.zeros(batch_size, n_heads, 1, self.history_window_size, dtype=self.attn_history_num.dtype, device=device)
                self.attn_history_num.scatter_(2, eviction_idx_hist_num, zeros_hist_num)

                # Target shape for attn_history_denom scatter: [B, H, 1]
                eviction_idx_hist_denom = eviction_idx_pos # Same shape as pos index
                zeros_hist_denom = torch.zeros_like(input_pos_tensor, dtype=self.attn_history_denom.dtype)
                self.attn_history_denom.scatter_(2, eviction_idx_hist_denom, zeros_hist_denom)

                # self.curr_len remains self.max_cache_size

        return self.k_cache, self.v_cache

    def compression_ratio(self, seq_len):
        return ((seq_len - self.max_cache_size) / seq_len)

    def get_cache_stats(self, seq_len):
        stats = {}
        for layer_idx, layer in enumerate(self.layers):
            stats[f"compression_ratio_{layer_idx}"] = layer.attention.kv_cache.compression_ratio(seq_len)
        stats["compression_ratio"] = sum(stats.values()) / len(stats)
        return stats