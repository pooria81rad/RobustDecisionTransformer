## Implementation for Robust Decision Transformer

### Usage

1. **Generate the Downsampled Dataset:**
   ```
   cd utils
   python ratio_dataset.py --env_name walker2d-medium-replay-v2 --ratio 0.1
   ```

2. **Reproduce the Results from the Paper:**

   **For Random State Corruption:**
   ```
   python -m RDT --seed 0 --env walker2d-medium-replay-v2 \
              --corruption_seed 0 \
              --corruption_mode random \
              --corruption_obs 1.0 \
              --corruption_rate 0.3 \
              --dataset_path your_dataset_path
   ```

   **For Adversarial State Corruption:**
   ```
   python -m RDT --seed 0 --env walker2d-medium-replay-v2 \
              --corruption_agent IQL \
              --corruption_mode adversarial \
              --corruption_obs 1.0 \
              --corruption_rate 0.3 \
              --dataset_path your_dataset_path
   ```

3. **Run Baseline Results:**

   The implementation of all baselines (BC, RBC, DeFog, CQL, UWMSG, RIQL, DT) compared in our paper is provided in the `algos` folder. For example, to run RIQL:
   ```
   python -m algos.RIQL --seed 0 --env walker2d-medium-replay-v2 \
              --corruption_agent IQL \
              --corruption_mode adversarial \
              --corruption_obs 1.0 \
              --corruption_rate 0.3 \
              --dataset_path your_dataset_path
   ```

4. **Additional Instructions:**

   - Use `--corruption_mode` to set the data corruption type (`random` or `adversarial`).
   - Use `--corruption_rate` to set the corruption ratio of the whole dataset.
   - Use `--corruption_obs` to set the corruption scale on the state.
   - Similarly, use `--corruption_act` and `--corruption_rew` to set the corruption scale on the action and reward, respectively.
