## Implementation for Robust Decision Transformer

### Usage

1. First, generate the downsampled dataset:
   ```
   python ratio_dataset --env_name "walker2d-medium-replay-v2" --ratio 0.1
   ```

2. Reproduce the results from the paper:

   For random state corruption:
   ```
   python RDT --seed 0 --env walker2d-medium-replay-v2 \
              --corruption_seed 0 \
              --corruption_mode random \
              --corruption_obs 1.0 \
              --corruption_rate 0.3 \
              --dataset_path your_dataset_path
   ```

   For adversarial state corruption:
   ```
   python RDT --seed 0 --env walker2d-medium-replay-v2 \
              --corruption_agent IQL \
              --corruption_mode adversarial \
              --corruption_obs 1.0 \
              --corruption_rate 0.3 \
              --dataset_path your_dataset_path
   ```
