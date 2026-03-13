#!/usr/bin/env python3
"""Verify Alpamayo-R1 GRPO configuration.

This script validates the GRPO configuration files and checks for common issues.

Usage:
    python examples/embodiment/verify_alpamayo_grpo.py
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf


def verify_config():
    """Verify GRPO configuration."""
    print("=" * 60)
    print("Alpamayo-R1 GRPO Configuration Verification")
    print("=" * 60)
    
    embodied_path = Path(__file__).parent
    config_path = embodied_path / "config"
    
    # Check if config files exist
    grpo_config = config_path / "alpamayo_grpo.yaml"
    env_config = config_path / "env" / "physical_ai_av.yaml"
    model_config = config_path / "model" / "alpamayo.yaml"
    
    print("\n1. Checking configuration files...")
    for cfg_file, name in [
        (grpo_config, "GRPO config"),
        (env_config, "Environment config"),
        (model_config, "Model config"),
    ]:
        if cfg_file.exists():
            print(f"   ✅ {name}: {cfg_file}")
        else:
            print(f"   ❌ {name} NOT FOUND: {cfg_file}")
            return False
    
    # Load and validate GRPO config
    print("\n2. Loading GRPO configuration...")
    try:
        cfg = OmegaConf.load(grpo_config)
        print("   ✅ Configuration loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load configuration: {e}")
        return False
    
    # Check required keys
    print("\n3. Checking required configuration keys...")
    required_keys = {
        "algorithm": ["adv_type", "group_size", "kl_beta"],
        "actor": ["model", "optim", "fsdp_config"],
        "env": ["total_num_envs", "dataset"],
        "runner": ["max_epochs", "save_interval"],
    }
    
    all_valid = True
    for section, keys in required_keys.items():
        if not hasattr(cfg, section):
            print(f"   ❌ Missing section: {section}")
            all_valid = False
            continue
        
        section_cfg = getattr(cfg, section)
        for key in keys:
            if key not in section_cfg:
                print(f"   ❌ Missing key: {section}.{key}")
                all_valid = False
    
    if all_valid:
        print("   ✅ All required keys present")
    
    # Check algorithm configuration
    print("\n4. Checking algorithm configuration...")
    if cfg.algorithm.adv_type == "grpo":
        print("   ✅ Algorithm type: GRPO")
    else:
        print(f"   ⚠️  Algorithm type: {cfg.algorithm.adv_type} (expected: grpo)")
    
    print(f"   - Group size: {cfg.algorithm.group_size}")
    print(f"   - KL beta: {cfg.algorithm.kl_beta}")
    print(f"   - Normalize advantages: {cfg.algorithm.normalize_advantages}")
    
    # Check model configuration
    print("\n5. Checking model configuration...")
    model_cfg = cfg.actor.model
    print(f"   - Model path: {model_cfg.model_path}")
    print(f"   - Model type: {model_cfg.model_type}")
    print(f"   - Precision: {model_cfg.precision}")
    print(f"   - Value head: {model_cfg.add_value_head}")
    
    # Check environment configuration
    print("\n6. Checking environment configuration...")
    env_cfg = cfg.env
    print(f"   - Total envs: {env_cfg.total_num_envs}")
    print(f"   - Group size: {env_cfg.group_size}")
    print(f"   - Max episode steps: {env_cfg.max_episode_steps}")
    
    # Check reward configuration
    if hasattr(env_cfg, "dataset") and hasattr(env_cfg.dataset, "reward_config"):
        reward_cfg = env_cfg.dataset.reward_config
        print("\n7. Reward configuration:")
        print(f"   - Smoothness weight: {reward_cfg.get('smoothness_weight', 0)}")
        print(f"   - Goal weight: {reward_cfg.get('goal_weight', 0)}")
        print(f"   - Collision weight: {reward_cfg.get('collision_weight', 0)}")
    
    # Check cluster configuration
    print("\n8. Checking cluster configuration...")
    print(f"   - Num nodes: {cfg.cluster.num_nodes}")
    print(f"   - Component placement: {cfg.cluster.component_placement}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou can now run GRPO training:")
        print("   bash examples/embodiment/run_alpamayo_grpo.sh alpamayo_grpo 0-3")
    else:
        print("⚠️  SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before running training.")
    print("=" * 60)
    
    return all_valid


if __name__ == "__main__":
    success = verify_config()
    sys.exit(0 if success else 1)
