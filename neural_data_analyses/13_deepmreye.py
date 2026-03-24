#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepMReye BIDS-compatible pipeline with subject-wise output structure:
- Predictions go in:   /.../derivatives/sub-XX/sub-XX_run-YY_desc-predictions.csv
- Masks go in:         /.../derivatives/sub-XX/masks/
- Global summary:      /.../derivatives/model_error_summary.csv
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/dev/null'

print("[Check] TensorFlow is working")
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import shutil
import pandas as pd
from deepmreye import preprocess, train
from deepmreye.util import data_generator, model_opts

# ==== CONFIGURATION ====
PREPROC_PATH = '/data/pt_02747/action_hippo/data/derivatives'
OUTPUT_BASE = '/data/pt_02747/action_hippo/deepmreye'
DERIVATIVES_DIR = os.path.join(OUTPUT_BASE, 'derivatives')
MODEL_PATH = os.path.join(OUTPUT_BASE, 'models/model_weights/datasets_1to6.h5')
TASK_NAME = 'probe'

# ==== FUNCTIONS ====

def preprocess_eye_mask(sub_id, run_number, subject_output_dir):
    run_str = f"run-{run_number:02d}"
    bold_path = os.path.join(PREPROC_PATH, f"sub-{sub_id}", "func",
        f"sub-{sub_id}_task-{TASK_NAME}_{run_str}_space-T1w_desc-preproc_bold.nii.gz")

    mask_dir = os.path.join(subject_output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    mask_fname = os.path.join(mask_dir, f"sub-{sub_id}_{run_str}_mask_norm.npz")

    if not os.path.exists(mask_fname):
        print(f"[Preprocessing] {bold_path}")

        print("[DEBUG] Getting eye masks/templates")
        eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges = preprocess.get_masks()

        print("[DEBUG] Running participant-specific coregistration")
        mask, _ = preprocess.run_participant(
            bold_path, dme_template, eyemask_big, eyemask_small,
            x_edges, y_edges, z_edges,
            transforms=['Affine', 'Affine', 'SyNAggro']
        )

        print("[DEBUG] Moving ANTs outputs")
        base_name = os.path.basename(bold_path).replace('.nii.gz', '')
        shutil.move(f"{os.path.dirname(bold_path)}/mask_{base_name}.p", mask_dir)
        shutil.move(f"{os.path.dirname(bold_path)}/report_{base_name}.html", mask_dir)

        print("[DEBUG] Normalizing extracted mask")
        mask_norm = preprocess.normalize_img(mask)
        labels_dummy = np.zeros((mask.shape[3], 10, 2))

        participant_data = [mask_norm]
        participant_labels = [labels_dummy]
        participant_ids = [([sub_id] * labels_dummy.shape[0], [run_str] * labels_dummy.shape[0])]

        print("[DEBUG] Saving .npz data")
        preprocess.save_data(
            f"sub-{sub_id}_{run_str}",
            participant_data,
            participant_labels,
            participant_ids,
            mask_dir,
            center_labels=False
        )

        os.rename(
            os.path.join(mask_dir, f"sub-{sub_id}_{run_str}.npz"),
            mask_fname
        )
        print(f"[OK] Saved mask to: {mask_fname}")
    else:
        print(f"[Skip] Mask already exists: {mask_fname}")


def predict_and_save_csv(sub_id, run_number, subject_output_dir):
    run_str = f"run-{run_number:02d}"
    mask_dir = os.path.join(subject_output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(subject_output_dir, exist_ok=True)
    os.makedirs(DERIVATIVES_DIR, exist_ok=True)

    npz_path = os.path.join(mask_dir, f"sub-{sub_id}_{run_str}_mask_norm.npz")
    csv_path = os.path.join(subject_output_dir, f"sub-{sub_id}_{run_str}_desc-predictions.csv")
    csv_flat_path = os.path.join(subject_output_dir, f"sub-{sub_id}_{run_str}_desc-predictions_flat.csv")
    error_csv_path = os.path.join(DERIVATIVES_DIR, "model_error_summary.csv")

    TR = 1.0
    n_subTR = 10
    dt = TR / n_subTR

    if not os.path.exists(csv_path):
        print(f"[Predicting] sub-{sub_id}, {run_str}")
        print(f"[DEBUG] Loading npz: {npz_path}")

        try:
            npz = np.load(npz_path)
            #print("[DEBUG] .npz keys:", npz.files)
        except Exception as e:
            print(f"[ERROR] Could not read .npz: {e}")
            return

        data_list = [npz_path]
        print("[DEBUG] Creating data generators")
        generators = data_generator.create_generators(data_list, data_list)
        generators = (*generators, data_list, data_list)

        print("[DEBUG] Training model (untrained)")
        _, model_inference = train.train_model(
            dataset=sub_id,
            generators=generators,
            opts=model_opts.get_opts(),
            return_untrained=True
        )

        print("[DEBUG] Loading weights")
        model_inference.load_weights(MODEL_PATH)
        print("[DEBUG] Weights loaded")

        print("[DEBUG] Running inference")
        evaluation, scores = train.evaluate_model(
            dataset=sub_id,
            model=model_inference,
            generators=generators,
            save=False,
            verbose=2,
            percentile_cut=80,
            model_path=MODEL_PATH
        )
        print("[DEBUG] Evaluation complete")

        preds = evaluation[npz_path]['pred_y']  # shape: (n_vol, 10, 2)
        n_vols = preds.shape[0]

        print(f"[DEBUG] Prediction shape: {preds.shape}")

        # Wide CSV (original)
        df = pd.DataFrame(
            preds.reshape(n_vols, -1),
            columns=[f'subTR{st}_{axis}' for st in range(10) for axis in ['x', 'y']]
        )
        df.insert(0, 'volume', np.arange(n_vols))
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved wide-format predictions to: {csv_path}")

        # Flat long-format CSV
        flat_data = []
        for vol in range(n_vols):
            for subtr in range(n_subTR):
                t = vol * TR + subtr * dt
                x, y = preds[vol, subtr, :]
                flat_data.append([t, vol, subtr, x, y])

        flat_df = pd.DataFrame(flat_data, columns=['time', 'volume', 'subTR', 'x', 'y'])

        # Add proxy confidence (variance across subTRs)
        subtr_var = np.var(preds, axis=1)  # shape: (n_vols, 2)
        flat_df['subtr_var'] = np.repeat(np.mean(subtr_var, axis=1), n_subTR)
        flat_df.to_csv(csv_flat_path, index=False)
        print(f"[OK] Saved flat-format predictions to: {csv_flat_path}")

        # ---- Error and confidence summary ----
        score = scores.get(npz_path, {})
        summary = {'subject': sub_id, 'run': run_str}

        summary.update({
            'mean_subtr_variance': np.mean(subtr_var),
            'median_subtr_variance': np.median(subtr_var),
            'std_subtr_variance': np.std(subtr_var)
        })

        if 'MAE_x' in score:
            summary.update({
                'MAE_x': np.mean(score['MAE_x']),
                'MAE_y': np.mean(score['MAE_y']),
                'corr_x': np.mean(score['corr_x']),
                'corr_y': np.mean(score['corr_y']),
                'R2_x': np.mean(score['R2_x']),
                'R2_y': np.mean(score['R2_y'])
            })

        error_df = pd.DataFrame([summary])
        if not os.path.exists(error_csv_path):
            error_df.to_csv(error_csv_path, index=False)
        else:
            error_df.to_csv(error_csv_path, mode='a', index=False, header=False)
        print(f"[OK] Wrote summary to: {error_csv_path}")

    else:
        print(f"[Skip] CSV already exists: {csv_path}")


# ==== MAIN ====

import re

# Auto-detect subject folders in BIDS derivatives dir
subj_list = []
for name in os.listdir(PREPROC_PATH):
    if os.path.isdir(os.path.join(PREPROC_PATH, name)) and name.startswith('sub-'):
        match = re.match(r'sub-(.+)', name)
        if match:
            subj_list.append(match.group(1))

subj_list.sort()

print(f"[INFO] Found {len(subj_list)} subjects: {subj_list}")

run_range = range(1, 9)  # Replace with desired run numbers

for sub_id in subj_list:
    subject_output_dir = os.path.join(DERIVATIVES_DIR, f"sub-{sub_id}")
    for run_number in run_range:
        try:
            print(f"\n=== Processing sub-{sub_id}, run-{run_number:02d} ===")
            preprocess_eye_mask(sub_id, run_number, subject_output_dir)
            predict_and_save_csv(sub_id, run_number, subject_output_dir)
        except Exception as e:
            print(f"[ERROR] sub-{sub_id} run-{run_number:02d} failed: {e}")
