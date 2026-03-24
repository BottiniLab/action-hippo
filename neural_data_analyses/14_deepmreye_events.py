import os
import re
import json
import pandas as pd

# Configuration
PREPROC_EVENTS_DIR = '/data/pt_02747/action_hippo/data/derivatives'
DEEP_RESULTS_DIR = '/data/pt_02747/action_hippo/deepmreye/derivatives'
TRIAL_TYPES_OF_INTEREST = [1, 2, 3, 4, '1', '2', '3', '4']
ALL_TRIALS_OUTPUT = os.path.join(DEEP_RESULTS_DIR, "all_trials_flat.csv")

# Remove previous global file to overwrite
if os.path.exists(ALL_TRIALS_OUTPUT):
    os.remove(ALL_TRIALS_OUTPUT)
    print(f"[INFO] Cleared previous file: {ALL_TRIALS_OUTPUT}")

# Loop over subject folders in derivatives
for subject in os.listdir(PREPROC_EVENTS_DIR):
    if not subject.startswith("sub-"):
        continue

    sub_id = subject.replace("sub-", "")
    func_dir = os.path.join(PREPROC_EVENTS_DIR, subject, "func")
    results_dir = os.path.join(DEEP_RESULTS_DIR, subject)

    if not os.path.exists(func_dir) or not os.path.exists(results_dir):
        continue

    print(f"[INFO] Processing subject {sub_id}")

    for file in os.listdir(func_dir):
        if not file.endswith("_events.tsv") or "run-" not in file:
            continue

        run_part = file.split("run-")[1].split("_")[0]
        run_str = f"run-{run_part}"
        events_path = os.path.join(func_dir, file)

        try:
            events = pd.read_csv(events_path, sep='\t', encoding='utf-8', engine='c')
        except (UnicodeDecodeError, pd.errors.ParserError):
            print(f"[WARNING] Falling back to Latin-1 for: {events_path}")
            events = pd.read_csv(events_path, sep='\t', encoding='latin1', engine='c')

        if 'trial_type' not in events.columns:
            print(f"[WARNING] No 'trial_type' column in {events_path}")
            continue

        filtered = events[events['trial_type'].isin(TRIAL_TYPES_OF_INTEREST)]
        if filtered.empty:
            print(f"[INFO] No matching trials in {events_path}")
            continue

        # Load DeepMReye flat predictions
        flat_csv = os.path.join(results_dir, f"sub-{sub_id}_{run_str}_desc-predictions_flat.csv")
        if not os.path.exists(flat_csv):
            print(f"[WARNING] No DeepMReye flat predictions found for {sub_id}, {run_str}")
            continue

        preds = pd.read_csv(flat_csv)

        rows = []
        trial_rows = []

        for idx, row in filtered.iterrows():
            onset = row['onset'] - 0.5
            offset = onset + row['duration'] + 1.1
            trial_type = row['trial_type']

            trial_preds = preds[(preds['time'] >= onset) & (preds['time'] <= offset)]
            if trial_preds.empty:
                continue

            x_vals = trial_preds['x'].tolist()
            y_vals = trial_preds['y'].tolist()
            median_x = pd.Series(x_vals).median()
            median_y = pd.Series(y_vals).median()

            # Create structured trial ID
            trial_id = f"{sub_id.zfill(2)}{run_part.zfill(2)}{str(idx).zfill(3)}"

            # Save trial summary
            rows.append({
                'onset': onset,
                'duration': row['duration'],
                'trial_type': trial_type,
                'median_x': median_x,
                'median_y': median_y,
                'x_values': x_vals,
                'y_values': y_vals,
                'trial_id': trial_id
            })

            # Save per-timepoint rows
            time_step = 0.1
            for rel_idx, (_, pred_row) in enumerate(trial_preds.iterrows()):
                time = pred_row['time']
                time_rel = round(rel_idx * time_step, 3) - 0.5
                time_rel_acc = time - onset

                trial_rows.append({
                    'subject': sub_id,
                    'run': run_str,
                    'trial_type': trial_type,
                    'trial_id': trial_id,
                    'time': time,
                    'time_rel': time_rel,
                    'time_rel_acc': time_rel_acc,
                    'x': pred_row['x'],
                    'y': pred_row['y']
                })

        if rows:
            # Create per-run event folder
            events_output_dir = os.path.join(results_dir, "events")
            os.makedirs(events_output_dir, exist_ok=True)

            # Save TSV
            tsv_path = os.path.join(events_output_dir, f"sub-{sub_id}_{run_str}_median_positions.tsv")
            tsv_df = pd.DataFrame(rows)
            tsv_df.to_csv(tsv_path, sep='\t', index=False)
            print(f"[OK] Wrote TSV: {tsv_path}")

            # Save JSON
            json_path = tsv_path.replace('.tsv', '.json')
            with open(json_path, 'w') as jf:
                json.dump(rows, jf, indent=2)
            print(f"[OK] Wrote JSON: {json_path}")

            # Save flat format to global file (overwrite mode already handled above)
            trial_df = pd.DataFrame(trial_rows)
            mode = 'a' if os.path.exists(ALL_TRIALS_OUTPUT) else 'w'
            trial_df.to_csv(ALL_TRIALS_OUTPUT, mode=mode, index=False, header=(mode == 'w'))
            print(f"[OK] Wrote {len(trial_rows)} rows to: {ALL_TRIALS_OUTPUT}")

        else:
            print(f"[INFO] No valid events matched predictions for {sub_id}, {run_str}")
