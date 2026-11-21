import torch


def corrupt_tokens(tokens, keypoints, raw_xyz, scenario, generator):
    noisy_tokens = tokens.clone()
    noisy_keypoints = keypoints.clone()
    noisy_raw_xyz = raw_xyz.clone()

    if "feature_sigma" in scenario:
        noisy_keypoints = noisy_keypoints + torch.randn(
            noisy_keypoints.shape, generator=generator, device=noisy_keypoints.device,
            dtype=noisy_keypoints.dtype,
        ) * scenario["feature_sigma"]
        noisy_raw_xyz = noisy_raw_xyz + torch.randn(
            noisy_raw_xyz.shape, generator=generator, device=noisy_raw_xyz.device,
            dtype=noisy_raw_xyz.dtype,
        ) * scenario["raw_sigma"]
    if "visual_sigma" in scenario:
        noisy_tokens = noisy_tokens + torch.randn(
            noisy_tokens.shape, generator=generator, device=noisy_tokens.device,
            dtype=noisy_tokens.dtype,
        ) * scenario["visual_sigma"]
    if "visual_token_dropout" in scenario:
        drop_prob = scenario["visual_token_dropout"]
        keep = (torch.rand(
            noisy_tokens.size(0), noisy_tokens.size(1), 1,
            generator=generator, device=noisy_tokens.device,
        ) > drop_prob).to(noisy_tokens.dtype)
        noisy_tokens = noisy_tokens * keep
    if "visual_patch_occlusion" in scenario:
        drop_prob = scenario["visual_patch_occlusion"]
        token_count = noisy_tokens.size(1)
        grid_size = int(round(token_count ** 0.5))
        if grid_size * grid_size != token_count:
            raise ValueError(f"Patch occlusion requires square token grid, got {token_count} tokens.")
        tokens_2d = noisy_tokens.reshape(noisy_tokens.size(0), grid_size, grid_size, -1)
        patch_size = max(1, int(round((token_count * drop_prob) ** 0.5)))
        patch_size = min(patch_size, grid_size)
        for row in range(tokens_2d.size(0)):
            y = torch.randint(0, grid_size - patch_size + 1, (1,), generator=generator,
                              device=noisy_tokens.device).item()
            x = torch.randint(0, grid_size - patch_size + 1, (1,), generator=generator,
                              device=noisy_tokens.device).item()
            tokens_2d[row, y:y + patch_size, x:x + patch_size] = 0
        noisy_tokens = tokens_2d.reshape_as(noisy_tokens)
    return noisy_tokens, noisy_keypoints, noisy_raw_xyz


def stable_scenario_seed(base_seed, experiment, scenario_name):
    text = f"{experiment}:{scenario_name}"
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
    return int(base_seed + offset)
