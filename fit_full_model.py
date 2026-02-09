# load packages
import os
import time
import logging
import socket
import pickle
import json
import math
import numpy as np
import pandas as pd
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import IPython.display as ipd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from datetime import datetime
import multiprocessing as mp
import librosa
import copy
import sys
import umap
import hdbscan
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
import torch.fft as tfft
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.ndimage import median_filter

#### PITCH ESTIMATION ####
def find_spectral_peaks_prominence(y, vmask, fs, hop_length, win_length, n_fft, 
                                   f0_min=300.0, f0_max=8000.0, 
                                   noise_margin_db=6.0, 
                                   min_prominence_db=10.0):
    """
    Find prominent peaks in the spectrum for each frame.
    """
    # Compute STFT
    f, t, Zxx = signal.stft(y, fs=fs, window='hann', nperseg=win_length, 
                            noverlap=win_length-hop_length, nfft=n_fft, 
                            return_onesided=True, boundary='zeros', padded=True)
    
    # Magnitude in dB
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-10)
    
    # Frequency mask for search range
    freq_mask = (f >= f0_min) & (f <= f0_max)
    f_valid = f[freq_mask]
    mag_db_valid = mag_db[freq_mask, :]
    
    T = mag_db.shape[1]
    
    # Align vmask
    vmask_aligned = vmask[:T] if len(vmask) >= T else np.pad(vmask, (0, T - len(vmask)), constant_values=False)
    
    # Estimate noise floor per frequency bin
    unvoiced_frames = ~vmask_aligned
    if unvoiced_frames.sum() > 0:
        noise_floor = np.median(mag_db_valid[:, unvoiced_frames], axis=1)
    else:
        noise_floor = np.percentile(mag_db_valid, 10, axis=1)
    
    # Threshold
    threshold = noise_floor[:, np.newaxis] + noise_margin_db
    
    # Find peaks with prominence for each frame
    peaks_list = []
    
    for t_idx in range(T):
        if vmask_aligned[t_idx]:
            spectrum = mag_db_valid[:, t_idx]
            
            # Find peaks with prominence requirement
            peak_indices, properties = signal.find_peaks(spectrum, 
                                                         prominence=min_prominence_db)
            
            # Filter by noise threshold
            above_threshold = spectrum[peak_indices] > threshold[peak_indices, 0]
            valid_peak_indices = peak_indices[above_threshold]
            peak_freqs = f_valid[valid_peak_indices]
            
            peaks_list.append((valid_peak_indices, peak_freqs))
        else:
            peaks_list.append((np.array([]), np.array([])))
    
    return mag_db, f, t, peaks_list, vmask_aligned, f_valid

def get_pitch(y, vmask, fs, hop_length, win_length, n_fft,
              f0_min=300.0, f0_max=4000.0,
              noise_margin_db=6.0,
              min_prominence_db=10.0,
              transition_weight=0.01,
              harmonic_bonus=5.0,
              harmonic_tolerance=0.15,
              n_harmonics_check=4,
              freq_boost_exp=0.5,
              smooth=True,
              jump_threshold_hz=500,
              smooth_window=5):
    """
    Estimate pitch contour using Viterbi tracking over prominent spectral peaks.
    
    Parameters:
    ----------
    y : np.ndarray
        Audio waveform (1D)
    vmask : np.ndarray
        Voicing mask (boolean, same length as STFT frames)
    fs : int
        Sample rate
    hop_length : int
        STFT hop length
    win_length : int
        STFT window length
    n_fft : int
        FFT size
    f0_min : float
        Minimum frequency to search (Hz)
    f0_max : float
        Maximum frequency to search (Hz)
    noise_margin_db : float
        dB above noise floor to threshold peaks
    min_prominence_db : float
        Minimum peak prominence in dB
    transition_weight : float
        Viterbi transition cost weight
    harmonic_bonus : float
        dB bonus per detected harmonic
    harmonic_tolerance : float
        Tolerance for harmonic matching (0.15 = ±15%)
    n_harmonics_check : int
        Number of harmonics to check (2f, 3f, ...)
    freq_boost_exp : float
        Exponent for low-frequency bias (0.5 = moderate)
    smooth : bool
        Whether to apply adaptive smoothing
    jump_threshold_hz : float
        Threshold for detecting jumps in smoothing
    smooth_window : int
        Median filter window size (must be odd)
    
    Returns:
    -------
    f0 : np.ndarray
        Pitch trajectory (Hz), NaN for unvoiced frames
    """
    
    # Detect prominent peaks
    mag_db, freqs, times, peaks_list, vmask_aligned, f_valid = find_spectral_peaks_prominence(
        y=y, vmask=vmask, fs=fs, hop_length=hop_length,
        win_length=win_length, n_fft=n_fft,
        f0_min=f0_min, f0_max=f0_max,
        noise_margin_db=noise_margin_db,
        min_prominence_db=min_prominence_db
    )
    
    T = len(peaks_list)
    freq_mask = (freqs >= f0_min) & (freqs <= f0_max)
    mag_db_valid = mag_db[freq_mask, :]
    
    # Get voiced frames with peaks
    voiced_frames = [t for t in range(T) if vmask_aligned[t] and len(peaks_list[t][0]) > 0]
    
    if len(voiced_frames) == 0:
        return np.full(T, np.nan)
    
    # Helper: detect harmonic support
    def get_harmonic_bonus(peak_freq, all_peak_freqs):
        bonus = 0
        for k in range(2, n_harmonics_check + 2):
            harmonic_freq = peak_freq * k
            for other_freq in all_peak_freqs:
                if abs(other_freq - harmonic_freq) / harmonic_freq < harmonic_tolerance:
                    bonus += harmonic_bonus
                    break
        return bonus
    
    # Viterbi forward pass
    viterbi_scores = [[] for _ in range(T)]
    backpointers = [[] for _ in range(T)]
    
    # Initialize first voiced frame
    first_frame = voiced_frames[0]
    peak_indices, peak_freqs = peaks_list[first_frame]
    
    for peak_idx in range(len(peak_freqs)):
        freq_bin = peak_indices[peak_idx]
        curr_freq = peak_freqs[peak_idx]
        emission_score = mag_db_valid[freq_bin, first_frame]
        bonus = get_harmonic_bonus(curr_freq, peak_freqs)
        freq_boost = (f0_max / curr_freq) ** freq_boost_exp
        emission_score = emission_score * freq_boost + bonus
        viterbi_scores[first_frame].append(emission_score)
        backpointers[first_frame].append((None, None))
    
    # Forward through voiced frames
    for t_idx in range(1, len(voiced_frames)):
        t_curr = voiced_frames[t_idx]
        t_prev = voiced_frames[t_idx - 1]
        
        curr_peak_indices, curr_peak_freqs = peaks_list[t_curr]
        prev_peak_indices, prev_peak_freqs = peaks_list[t_prev]
        
        if len(prev_peak_freqs) == 0 or len(curr_peak_freqs) == 0:
            continue
        
        for curr_idx in range(len(curr_peak_freqs)):
            curr_freq = curr_peak_freqs[curr_idx]
            freq_bin = curr_peak_indices[curr_idx]
            emission_score = mag_db_valid[freq_bin, t_curr]
            bonus = get_harmonic_bonus(curr_freq, curr_peak_freqs)
            freq_boost = (f0_max / curr_freq) ** freq_boost_exp
            emission_score = emission_score * freq_boost + bonus
            
            best_score = -np.inf
            best_prev_idx = 0
            
            for prev_idx in range(len(prev_peak_freqs)):
                prev_freq = prev_peak_freqs[prev_idx]
                freq_jump = abs(curr_freq - prev_freq)
                transition_cost = -transition_weight * freq_jump
                total_score = viterbi_scores[t_prev][prev_idx] + transition_cost + emission_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_prev_idx = prev_idx
            
            viterbi_scores[t_curr].append(best_score)
            backpointers[t_curr].append((t_prev, best_prev_idx))
    
    # Backtrack
    path = {}
    last_frame = voiced_frames[-1]
    if len(viterbi_scores[last_frame]) > 0:
        best_final_idx = np.argmax(viterbi_scores[last_frame])
        path[last_frame] = best_final_idx
        
        curr_frame = last_frame
        curr_peak_idx = best_final_idx
        
        while True:
            prev_frame, prev_peak_idx = backpointers[curr_frame][curr_peak_idx]
            if prev_frame is None:
                break
            path[prev_frame] = prev_peak_idx
            curr_frame = prev_frame
            curr_peak_idx = prev_peak_idx
    
    # Convert to frequency trajectory
    f0 = np.full(T, np.nan)
    for frame, peak_idx in path.items():
        _, peak_freqs = peaks_list[frame]
        if peak_idx < len(peak_freqs):
            f0[frame] = peak_freqs[peak_idx]
    
    # Adaptive smoothing
    if smooth:
        valid = ~np.isnan(f0)
        valid_indices = np.where(valid)[0]
        
        if len(valid_indices) >= 2:
            jumps = np.abs(np.diff(f0[valid]))
            large_jumps = jumps > jump_threshold_hz
            jump_positions = np.where(large_jumps)[0]
            
            segment_starts = [0] + list(jump_positions + 1)
            segment_ends = list(jump_positions + 1) + [len(valid_indices)]
            
            for start, end in zip(segment_starts, segment_ends):
                if end - start >= smooth_window:
                    segment_indices = valid_indices[start:end]
                    segment_values = f0[segment_indices]
                    smoothed = median_filter(segment_values, size=smooth_window, mode='nearest')
                    f0[segment_indices] = smoothed
    
    return f0, mag_db, freqs, times


#### AMPLITUDE ESTIMATOR ####
def estimate_amplitude(y, fs, win_length, hop_length, n_fft, f_min=200.0, f_max=8000.0, E_floor=-61):
    y = torch.as_tensor(y, dtype=torch.float32)
    if y.dim() == 1: y = y.unsqueeze(0)
    dev = y.device

    # STFT power
    win = torch.hann_window(win_length, periodic=False, device=dev, dtype=torch.float32)
    S = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=win, center=True, return_complex=True)
    P = (S.real * S.real + S.imag * S.imag)
    k0 = int(math.floor(f_min * n_fft / fs))
    k1 = int(math.ceil (f_max * n_fft / fs)) + 1
    E = P[:, k0:k1, :].sum(dim=1)
    A = torch.sqrt(E)
    return A.squeeze()


#### LOAD PITCH GRID ####
b_lo = -1/3 + 1e-3

n=500
d_max = 0.995
a_grid = np.linspace(-0.5, -0.12, n)
d_grid = 1.0 - np.logspace(np.log10(1 - d_max), 0, n)

with open('/mnt/cube/lo/chronic_ephys_pipeline/analysis/vocal_organ_model/final/a_b_sweep.pkl', 'rb') as f:
    results = pickle.load(f)
    pitch_grid = results['pitch_grid']
    amplitude_grid = results['amp_grid']

A = amplitude_grid.astype(float)
norm_amp = (A - np.nanmin(A)) / (np.nanmax(A) - np.nanmin(A))

#### DEFAULT PARAMETERS (can be overridden) ####
DEFAULT_SYS_PARS = {
    'gamma': 24000., 'Ch_inv': 4.5E10, 'Lb_inv': 1.E-4, 'Lg_inv': 1/82.,
    'Rb': 5E6, 'Rh': 6E5, 'V_ext': 0., 'dV_ext': 0., 'envelope': 0.,
}
DEFAULT_V_SOUND = 35000

#### PARAMETER EXTRACTION ####
from scipy.stats import rankdata

class BiomechanicalSongFitter:
    def __init__(self, a_grid, d_grid, pitch_grid, amplitude_grid):
        self.a_grid = a_grid
        self.d_grid = d_grid
        self.pitch_grid = pitch_grid
        self.amplitude_grid = amplitude_grid
        self.amp_interpolator = RegularGridInterpolator(
            (a_grid, d_grid), 
            amplitude_grid,
            bounds_error=False,
            fill_value=None
        )
        
    def extract_isopitch_contour(self, target_pitch):
        fig, ax = plt.subplots(figsize=(1, 1))
        cs = ax.contour(self.a_grid, self.d_grid, self.pitch_grid.T, levels=[target_pitch])
        plt.close(fig)
        
        if len(cs.collections[0].get_paths()) == 0:
            raise ValueError(f"No contour found for pitch {target_pitch} Hz")
        
        contour_path = cs.collections[0].get_paths()[0]
        contour_vertices = contour_path.vertices
        
        a_contour = contour_vertices[:, 0]
        d_contour = contour_vertices[:, 1]
        
        return a_contour, d_contour
    
    def get_amplitude_along_contour(self, a_contour, d_contour):
        points = np.column_stack([a_contour, d_contour])
        amplitudes = self.amp_interpolator(points)
        return amplitudes
    
    def get_percentile_position(self, amplitude_contour, target_percentile):
        sorted_indices = np.argsort(amplitude_contour)
        target_idx = int(target_percentile * (len(amplitude_contour) - 1))
        position = sorted_indices[target_idx] / (len(amplitude_contour) - 1)
        return np.clip(position, 0.0, 1.0)
    
    def _extract_contour_worker(self, t, pitch):
        a_cont, d_cont = self.extract_isopitch_contour(pitch)
        amp_cont = self.get_amplitude_along_contour(a_cont, d_cont)
        return t, a_cont, d_cont, amp_cont
    
    @staticmethod
    def _optimize_worker(trial, T_voiced, init_position, voiced_indices, 
                        amplitude_contours, birdsong_amp_voiced, lambda_smooth):
        def objective(voiced_positions):
            model_amp_voiced = np.zeros(T_voiced)
            for idx, t in enumerate(voiced_indices):
                amp_cont = amplitude_contours[t]
                pos = voiced_positions[idx]
                cont_idx = pos * (len(amp_cont) - 1)
                idx_low = int(np.floor(cont_idx))
                idx_high = min(idx_low + 1, len(amp_cont) - 1)
                weight = cont_idx - idx_low
                model_amp_voiced[idx] = (1 - weight) * amp_cont[idx_low] + weight * amp_cont[idx_high]
            
            if np.std(model_amp_voiced) < 1e-10 or np.std(birdsong_amp_voiced) < 1e-10:
                corr = -1.0
            else:
                corr = np.corrcoef(birdsong_amp_voiced, model_amp_voiced)[0, 1]
                if np.isnan(corr):
                    corr = -1.0

            smoothness = 0.0
            for i in range(len(voiced_indices) - 1):
                if voiced_indices[i+1] == voiced_indices[i] + 1:
                    smoothness += np.abs(voiced_positions[i+1] - voiced_positions[i])
            
            return -corr + lambda_smooth * smoothness
        
        if trial == 0:
            if np.isscalar(init_position):
                x0 = np.full(T_voiced, init_position)
            else:
                x0 = np.array(init_position)
        else:
            x0 = np.random.uniform(0.2, 0.8, T_voiced)
            x0 = np.convolve(x0, np.ones(5)/5, mode='same')
        
        bounds = [(0.0, 1.0) for _ in range(T_voiced)]
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        return result
    
    def fit_song(self, birdsong_pitch, birdsong_amplitude, voicing_mask,
                 lambda_smooth=0.0001, use_percentile_init=True, 
                 batch_size=30, max_batches=10, convergence_threshold=0.001,
                 n_jobs=-1, verbose=True):
        T = len(birdsong_pitch)
        
        birdsong_pitch = np.asarray(birdsong_pitch)
        birdsong_amplitude = np.asarray(birdsong_amplitude)
        voicing_mask = np.asarray(voicing_mask)
        
        voiced_indices = np.where(voicing_mask)[0]
        T_voiced = len(voiced_indices)
        
        if verbose: print("Extracting isopitch contours...")
        
        contour_results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_contour_worker)(t, birdsong_pitch[t])
            for t in tqdm(voiced_indices, disable=not verbose)
        )
        
        contours = {}
        amplitude_contours = {}
        for t, a_cont, d_cont, amp_cont in contour_results:
            contours[t] = (a_cont, d_cont)
            amplitude_contours[t] = amp_cont
        
        birdsong_amp_voiced = birdsong_amplitude[voiced_indices]
        
        if use_percentile_init:
            if verbose: print("Computing percentile-based initialization...")
            amp_percentiles = rankdata(birdsong_amp_voiced, method='average') / len(birdsong_amp_voiced)
            init_positions = np.array([
                self.get_percentile_position(amplitude_contours[voiced_indices[idx]], percentile)
                for idx, percentile in enumerate(amp_percentiles)
            ])
        else:
            init_positions = 0.5
        
        if verbose:
            print(f"Optimizing with adaptive batching (batch_size={batch_size})...")
        
        best_result = None
        best_corr = -np.inf
        batch_correlations = []
        batches_without_improvement = 0
        patience = 3
        
        for batch_num in range(max_batches):
            trial_offset = batch_num * batch_size
            
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(self._optimize_worker)(
                    trial_offset + i, T_voiced, init_positions, voiced_indices,
                    amplitude_contours, birdsong_amp_voiced, lambda_smooth
                )
                for i in range(batch_size)
            )
            
            batch_best = min(batch_results, key=lambda r: r.fun)
            batch_corr = -batch_best.fun + lambda_smooth * np.sum([
                np.abs(batch_best.x[i+1] - batch_best.x[i]) 
                for i in range(len(voiced_indices)-1) 
                if voiced_indices[i+1] == voiced_indices[i] + 1
            ])
            
            batch_correlations.append(batch_corr)
            
            improvement = batch_corr - best_corr
            if improvement > 0:
                best_corr = batch_corr
                best_result = batch_best
            
            if improvement > convergence_threshold:
                batches_without_improvement = 0
            else:
                batches_without_improvement += 1
            
            if verbose:
                print(f"  Batch {batch_num+1}: batch best = {batch_corr:.4f}, overall best = {best_corr:.4f}")
            
            if batches_without_improvement >= patience:
                if verbose:
                    print(f"  Converged! No improvement for {patience} batches")
                break
        
        alpha_traj = np.full(T, 0.15)
        d_traj = np.full(T, 0.99)
        positions = np.full(T, np.nan)
        model_amplitude = np.zeros(T)
        
        for idx, t in enumerate(voiced_indices):
            a_cont, d_cont = contours[t]
            amp_cont = amplitude_contours[t]
            
            pos = best_result.x[idx]
            positions[t] = pos
            
            cont_idx = pos * (len(amp_cont) - 1)
            idx_low = int(np.floor(cont_idx))
            idx_high = min(idx_low + 1, len(amp_cont) - 1)
            weight = cont_idx - idx_low
            
            model_amplitude[t] = (1 - weight) * amp_cont[idx_low] + weight * amp_cont[idx_high]
            alpha_traj[t] = (1 - weight) * a_cont[idx_low] + weight * a_cont[idx_high]
            d_traj[t] = (1 - weight) * d_cont[idx_low] + weight * d_cont[idx_high]
        
        final_corr = np.corrcoef(birdsong_amplitude[voiced_indices], model_amplitude[voiced_indices])[0, 1]
        
        if verbose:
            print(f"Optimization complete!")
            print(f"Final correlation: {final_corr:.4f}")
            smoothness_total = np.sum([np.abs(best_result.x[i+1] - best_result.x[i]) 
                                      for i in range(len(voiced_indices)-1) 
                                      if voiced_indices[i+1] == voiced_indices[i] + 1])
            print(f"Smoothness: {smoothness_total:.4f}")
            print(f"Total trials: {(batch_num+1) * batch_size}")
        
        return {
            'alpha': alpha_traj,
            'beta': d_traj,
            'positions': positions,
            'model_amplitude': model_amplitude,
            'correlation': final_corr,
            'success': best_result.success,
            'optimizer_result': best_result,
            'batch_correlations': batch_correlations
        }

    def plot_fit(self, birdsong_pitch, birdsong_amplitude, fit_result, voicing_mask, time=None):
        birdsong_amplitude = np.asarray(birdsong_amplitude)
        voicing_mask = np.asarray(voicing_mask)
        
        if time is None:
            time = np.arange(len(birdsong_amplitude))
        
        voiced_song_amp = birdsong_amplitude[voicing_mask]
        voiced_model_amp = fit_result['model_amplitude'][voicing_mask]
        
        model_amp_scaled = np.zeros(len(birdsong_amplitude))
        if np.std(voiced_model_amp) > 1e-10:
            scale = np.std(voiced_song_amp) / np.std(voiced_model_amp)
            offset = np.mean(voiced_song_amp) - scale * np.mean(voiced_model_amp)
            model_amp_scaled[voicing_mask] = scale * fit_result['model_amplitude'][voicing_mask] + offset
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
        
        ax = axes[0]
        ax.plot(time, birdsong_amplitude, 'k-', label='Original', linewidth=2)
        ax.plot(time, model_amp_scaled, 'r-', label='Recon', linewidth=2, alpha=0.7)
        ymin, ymax = ax.get_ylim()[0], ax.get_ylim()[1]
        ax.fill_between(time, ymin, ymax, where=~voicing_mask, alpha=0.2, color='gray', label='Silence')
        ax.set_ylabel('Amplitude')
        ax.set_xlim([min(time), max(time)])
        ax.set_ylim([ymin, ymax])
        ax.legend()
        ax.set_title(f"Amplitude Fit (correlation = {fit_result['correlation']:.4f})")
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(time, fit_result['alpha'], linewidth=2)
        ax.set_ylabel('α')
        ax.set_xlim([min(time), max(time)])
        ax.invert_yaxis()
        ax.grid(True, alpha=0)
        
        ax = axes[2]
        ax.plot(time, fit_result['d'], linewidth=2, color='orange')
        ax.set_ylabel('β')
        ax.set_xlabel('Time')
        ax.set_xlim([min(time), max(time)])
        ax.invert_yaxis()
        ax.grid(True, alpha=0.)
        
        plt.tight_layout()
        return fig

    def plot_parameter_heatmap(self, fit_result, voicing_mask=None):
        alpha_traj = np.asarray(fit_result['alpha'])
        d_traj = np.asarray(fit_result['d'])
        
        if voicing_mask is not None:
            voicing_mask = np.asarray(voicing_mask)
            alpha_voiced = alpha_traj[voicing_mask]
            d_voiced = d_traj[voicing_mask]
        else:
            alpha_voiced = alpha_traj
            d_voiced = d_traj
        
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.contourf(self.a_grid, self.d_grid, self.amplitude_grid.T, 
                         levels=20, cmap='inferno')
        time_indices = np.arange(len(alpha_voiced))
        scatter = ax.scatter(alpha_voiced, d_voiced, c=time_indices, 
                            s=5, cmap='cool')
        ax.set_xlabel('α')
        ax.set_ylabel('β ratio (proportional distance to SNILC cusp)')
        ax.set_title('Trajectory on Amplitude Landscape')
        plt.colorbar(im, ax=ax, label='Amplitude')
        cbar2 = plt.colorbar(scatter, ax=ax, label='Time')
        
        plt.tight_layout()
        return fig

fitter = BiomechanicalSongFitter(
    a_grid=a_grid,
    d_grid=d_grid, 
    pitch_grid=pitch_grid,
    amplitude_grid=norm_amp
)

class SNILCBounds(nn.Module):
    """
    β upper bound from the SNILC cusp (lower branch)
     - For α < 0: interpolate cusp
     - For α ≥ 0: return β_max_phys (no cusp constraint)
    """
    def __init__(self, beta_max_phys=100, n_grid=4096):
        super().__init__()
        b = torch.linspace(-1/3 + 1e-6, beta_max_phys, n_grid)  # valid β
        # Lower-branch cusp alpha for each beta (a_plus)
        x_plus = (1.0 + torch.sqrt(1.0 + 3.0 * b)) / 3.0
        a_plus = x_plus**3 - x_plus**2 - b * x_plus  # boundary alpha
        mask = a_plus < 0  # left of Hopf
        a_tab = a_plus[mask]
        b_tab = b[mask]
        idx = torch.argsort(a_tab)  # ascending α
        a_tab = a_tab[idx]
        b_tab = b_tab[idx]
        self.register_buffer("alpha_tab", a_tab)  # shape [M]
        self.register_buffer("beta_tab",  b_tab)  # shape [M]
        self.beta_max_phys = beta_max_phys

    def beta_cusp(self, alpha):
        """
        Interpolate β_cusp(α) for α < 0. For α ≥ 0, return β_max_phys
        so the cusp does not constrain in the silent regime
        """
        # for α < 0, interpolate
        a0, a1 = self.alpha_tab[0], self.alpha_tab[-1]
        a_clamped = torch.clamp(alpha, min=a0, max=a1)
        flat = a_clamped.reshape(-1)
        
        idx = torch.searchsorted(self.alpha_tab, flat, right=True)
        idx1 = torch.clamp(idx, min=1, max=self.alpha_tab.numel()-1)
        idx0 = idx1 - 1

        a_lo, a_hi = self.alpha_tab[idx0], self.alpha_tab[idx1]
        b_lo, b_hi = self.beta_tab[idx0],  self.beta_tab[idx1]
        t = (flat - a_lo) / (a_hi - a_lo + 1e-12)
        b_interp = (b_lo + t * (b_hi - b_lo)).reshape_as(alpha)
        
        # for α >= 0, return β max at α=0
        b_at_zero = self.beta_tab[-1].detach().to(b_interp)
        
        return torch.where(alpha >= 0, b_at_zero, b_interp)

    def upper_bound(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Return β_high: min(β_cusp(α), β_max_phys) for α < 0, else β_α=0
        """
        return self.beta_cusp(alpha)

snilc = SNILCBounds()


#### VOCAL SYNTHESIZER ####
class ZebraFinchVocalTract(nn.Module):
    def __init__(self, steps_per_sample, hop_length, fs_audio,
                 sys_pars=None, v_sound=None, jitter_sigma=0.10, integrator: str = "rk4"):
        super().__init__()
        
        # Use defaults if not provided
        if sys_pars is None:
            sys_pars = DEFAULT_SYS_PARS
        if v_sound is None:
            v_sound = DEFAULT_V_SOUND
            
        self.integrator = integrator.lower()
        self.hop_length = int(hop_length)
        self.jitter_sigma = float(jitter_sigma)
        self.steps_per_sample = int(steps_per_sample)
        self.fs_audio = int(fs_audio)

        # cache constants as buffers
        def buf(x):
            return torch.as_tensor(x, dtype=torch.float32)

        self.register_buffer("gamma",  buf(sys_pars["gamma"]))
        self.register_buffer("Ch_inv", buf(sys_pars["Ch_inv"]))
        self.register_buffer("Lb_inv", buf(sys_pars["Lb_inv"]))
        self.register_buffer("Lg_inv", buf(sys_pars["Lg_inv"]))
        self.register_buffer("Rb",     buf(sys_pars["Rb"]))
        self.register_buffer("Rh",     buf(sys_pars["Rh"]))

        # tube constants
        self.register_buffer("t_in", buf(0.5))
        self.register_buffer("r",    buf(0.1))

        # time step
        dt = 1.0 / (self.fs_audio * self.steps_per_sample)
        self.register_buffer("dt", buf(dt))
        self.register_buffer("inv_dt", buf(1.0 / dt))

        # delay buffer length (tau + 1)
        l_cm = 3.5
        tau = int(l_cm / (v_sound * dt))
        self.tau = int(tau)
        self.buf_len = self.tau + 1

    @staticmethod
    def _zoh_to_len(batch, L):
        # batch: [B, T] -> [B, L] via nearest upsample
        return F.interpolate(batch.unsqueeze(1), size=L, mode="nearest").squeeze(1)

    def forward(self, alpha, beta, gain, n_frames):
        dev = alpha.device
        sp = self.steps_per_sample
        Lw = int(n_frames * self.hop_length)

        # upsample controls to audio rate
        alpha = self._zoh_to_len(alpha, Lw)
        beta = self._zoh_to_len(beta, Lw)
        envelope = self._zoh_to_len(gain, Lw)

        # compensate envelope for SPS
        factor = {2: 5.952507019042969, 8: 4.259281158447266, 20: 1.0}.get(sp)
        if factor is None:
            raise ValueError(f"Unsupported steps_per_sample={sp}: no SPS compensation defined")
        envelope *= factor

        # jitter per audio sample (broadcast over substeps)
        if self.training and self.jitter_sigma > 0:
            eps = torch.randn(alpha.size(0), Lw, device=dev)
            jitter = torch.exp(self.jitter_sigma * eps - 0.5 * self.jitter_sigma * self.jitter_sigma)
            # clamp rare extremes
            jmax = math.exp(3.0 * self.jitter_sigma)
            jitter = jitter.clamp(1.0 / jmax, jmax)
        else:
            jitter = 1.0  # fast path

        B = alpha.size(0)
        tb = torch.zeros(B, self.buf_len, 2, device=dev, dtype=torch.float32)
        FWD, BWD = 0, 1  # delay-line buffer [B, buf_len, 2] (FWD=0, BWD=1)
        
        x0 = torch.full((B,), 1e-11, device=dev)
        x1 = torch.full((B,), 1e-11, device=dev)
        x2 = torch.full((B,), 1e-11, device=dev)
        x3 = torch.full((B,), 1e-11, device=dev)
        x4 = torch.full((B,), 1e-11, device=dev)

        # local constants
        g = self.gamma
        gg = g * g
        dt = self.dt
        inv_dt = self.inv_dt
        Lb_inv, Lg_inv, Ch_inv = self.Lb_inv, self.Lg_inv, self.Ch_inv
        Rb, Rh = self.Rb, self.Rh
        t_in, r = self.t_in, self.r
        tau = self.tau

        # precompute constants used in ODE
        C1 = Lg_inv * Ch_inv
        C2 = Rh * (Lb_inv + Lg_inv)
        C3 = C1 - Rb * Rh * Lb_inv * Lg_inv
        C4 = Rh * Lb_inv * Lg_inv
        C5 = Lb_inv / Lg_inv
        C6 = Rb * Lb_inv
        dt6 = dt / 6.0

        out = torch.empty(B, Lw, device=dev, dtype=torch.float32)
        A = envelope

        for t in range(Lw):
            a_t = alpha[:, t]
            b_t = beta[:,  t]
            if isinstance(jitter, torch.Tensor):
                A_1 = A[:, t] * jitter[:, t]
            else:
                A_1 = A[:, t]

            for s in range(sp):
                step   = t * sp + s
                i      = step % self.buf_len
                i_prev = (step - 1) % self.buf_len
                i_tau  = (step - tau) % self.buf_len
                i_tau_prev = (step - 1 - tau) % self.buf_len

                # reads
                f_tau  = tb[:, i_tau,  FWD]  # forward wave arriving at beak after delay
                b_tau  = tb[:, i_tau,  BWD]  # backward wave arriving at glottis after delay

                # scattering
                u_in  = t_in * A_1 * x1
                Vext  = u_in + b_tau
                b_new = -r * f_tau

                # write current slot
                tb[:, i, FWD] = Vext
                tb[:, i, BWD] = b_new

                # derivative of the delayed backward wave (constant across RK stages within this substep)
                b_tau_prev = tb[:, i_tau_prev, BWD]
                dbtau_dt   = (b_tau - b_tau_prev) * inv_dt
                
                # ---------- ODE integrator for the coupled syringeal source + upper vocal tract resonator ----------
                if self.integrator == "euler":
                    # very fast
                    k10 = x1
                    k11 = (gg*a_t) + (gg*b_t*x0) + (gg*x0*x0) - (gg*x0*x0*x0) - (g*x0*x1) - (g*x0*x0*x1)
                    k12 = x3
                    dU_dt_1 = t_in * A_1 * k11
                    dVext_1 = dU_dt_1 + dbtau_dt
                    Vext_1  = Vext
                    k13 = -C1*x2 - C2*x3 + C3*x4 + Lg_inv*dVext_1 + C4*Vext_1
                    k14 = -C5*x3 - C6*x4 + Lb_inv*Vext_1
                    x0 = x0 + dt*k10; x1 = x1 + dt*k11; x2 = x2 + dt*k12; x3 = x3 + dt*k13; x4 = x4 + dt*k14

                elif self.integrator == "heun":
                    # fast, decent quality
                    k10 = x1
                    k11 = (gg*a_t) + (gg*b_t*x0) + (gg*x0*x0) - (gg*x0*x0*x0) - (g*x0*x1) - (g*x0*x0*x1)
                    k12 = x3
                    dU_dt_1 = t_in * A_1 * k11
                    dVext_1 = dU_dt_1 + dbtau_dt
                    Vext_1  = Vext
                    k13 = -C1*x2 - C2*x3 + C3*x4 + Lg_inv*dVext_1 + C4*Vext_1
                    k14 = -C5*x3 - C6*x4 + Lb_inv*Vext_1
                    x0p = x0 + dt*k10; x1p = x1 + dt*k11; x2p = x2 + dt*k12; x3p = x3 + dt*k13; x4p = x4 + dt*k14
                    k20 = x1p
                    k21 = (gg*a_t) + (gg*b_t*x0p) + (gg*x0p*x0p) - (gg*x0p*x0p*x0p) - (g*x0p*x1p) - (g*x0p*x0p*x1p)
                    k22 = x3p
                    Vext_2  = t_in * A_1 * x1p + b_tau
                    dU_dt_2 = t_in * A_1 * k21
                    dVext_2 = dU_dt_2 + dbtau_dt
                    k23 = -C1*x2p - C2*x3p + C3*x4p + Lg_inv*dVext_2 + C4*Vext_2
                    k24 = -C5*x3p - C6*x4p + Lb_inv*Vext_2
                    x0 = x0 + 0.5*dt*(k10 + k20)
                    x1 = x1 + 0.5*dt*(k11 + k21)
                    x2 = x2 + 0.5*dt*(k12 + k22)
                    x3 = x3 + 0.5*dt*(k13 + k23)
                    x4 = x4 + 0.5*dt*(k14 + k24)

                else: ## RK4
                    # best quality
                    k10 = x1
                    k11 = (gg*a_t) + (gg*b_t*x0) + (gg*x0*x0) - (gg*x0*x0*x0) - (g*x0*x1) - (g*x0*x0*x1)
                    k12 = x3
                    dU_dt_1 = t_in * A_1 * k11
                    dVext_1 = dU_dt_1 + dbtau_dt
                    Vext_1  = Vext
                    k13 = -C1 * x2 - C2 * x3 + C3 * x4 + Lg_inv * dVext_1 + C4 * Vext_1
                    k14 = -C5 * x3 - C6 * x4 + Lb_inv * Vext_1
                    x0_2 = x0 + 0.5 * dt * k10; x1_2 = x1 + 0.5 * dt * k11
                    x2_2 = x2 + 0.5 * dt * k12; x3_2 = x3 + 0.5 * dt * k13; x4_2 = x4 + 0.5 * dt * k14

                    k20 = x1_2
                    k21 = (gg*a_t) + (gg*b_t*x0_2) + (gg*x0_2*x0_2) - (gg*x0_2*x0_2*x0_2) - (g*x0_2*x1_2) - (g*x0_2*x0_2*x1_2)
                    k22 = x3_2
                    Vext_2  = t_in * A_1 * x1_2 + b_tau
                    dU_dt_2 = t_in * A_1 * k21
                    dVext_2 = dU_dt_2 + dbtau_dt
                    k23 = -C1 * x2_2 - C2 * x3_2 + C3 * x4_2 + Lg_inv * dVext_2 + C4 * Vext_2
                    k24 = -C5 * x3_2 - C6 * x4_2 + Lb_inv * Vext_2
                    x0_3 = x0 + 0.5 * dt * k20; x1_3 = x1 + 0.5 * dt * k21
                    x2_3 = x2 + 0.5 * dt * k22; x3_3 = x3 + 0.5 * dt * k23; x4_3 = x4 + 0.5 * dt * k24

                    k30 = x1_3
                    k31 = (gg*a_t) + (gg*b_t*x0_3) + (gg*x0_3*x0_3) - (gg*x0_3*x0_3*x0_3) - (g*x0_3*x1_3) - (g*x0_3*x0_3*x1_3)
                    k32 = x3_3
                    Vext_3  = t_in * A_1 * x1_3 + b_tau
                    dU_dt_3 = t_in * A_1 * k31
                    dVext_3 = dU_dt_3 + dbtau_dt
                    k33 = -C1 * x2_3 - C2 * x3_3 + C3 * x4_3 + Lg_inv * dVext_3 + C4 * Vext_3
                    k34 = -C5 * x3_3 - C6 * x4_3 + Lb_inv * Vext_3
                    x0_4 = x0 + dt * k30; x1_4 = x1 + dt * k31
                    x2_4 = x2 + dt * k32; x3_4 = x3 + dt * k33; x4_4 = x4 + dt * k34

                    k40 = x1_4
                    k41 = (gg*a_t) + (gg*b_t*x0_4) + (gg*x0_4*x0_4) - (gg*x0_4*x0_4*x0_4) - (g*x0_4*x1_4) - (g*x0_4*x0_4*x1_4)
                    k42 = x3_4
                    Vext_4  = t_in * A_1 * x1_4 + b_tau
                    dU_dt_4 = t_in * A_1 * k41
                    dVext_4 = dU_dt_4 + dbtau_dt
                    k43 = -C1 * x2_4 - C2 * x3_4 + C3 * x4_4 + Lg_inv * dVext_4 + C4 * Vext_4
                    k44 = -C5 * x3_4 - C6 * x4_4 + Lb_inv * Vext_4
                    x0 = x0 + dt6 * (k10 + 2*k20 + 2*k30 + k40)
                    x1 = x1 + dt6 * (k11 + 2*k21 + 2*k31 + k41)
                    x2 = x2 + dt6 * (k12 + 2*k22 + 2*k32 + k42)
                    x3 = x3 + dt6 * (k13 + 2*k23 + 2*k33 + k43)
                    x4 = x4 + dt6 * (k14 + 2*k24 + 2*k34 + k44)
                # ---------- end ODE integrator ----------

            out[:, t] = Rb * x4  # radiated pressure sample

        return out  # [B, Lw]

def _init_vt_worker():
    torch.set_num_threads(1)

def _vt_worker(payload):
    (state_dict, sys_pars, v_sound, steps_per_sample, hop_length, integrator, fs_audio,
     a, b, g, Lm, off_mel, win_mel, idx) = payload

    vt = ZebraFinchVocalTract(
        sys_pars=sys_pars,
        v_sound=v_sound,
        steps_per_sample=steps_per_sample,
        hop_length=hop_length,
        fs_audio=fs_audio,
        jitter_sigma=0.0,
        integrator=integrator
    ).to("cpu").eval()

    if state_dict:
        vt.load_state_dict(state_dict, strict=False)

    with torch.inference_mode():
        y = vt(a.unsqueeze(0), b.unsqueeze(0), g.unsqueeze(0), Lm)[0].contiguous()

    start_samp = off_mel * hop_length
    length_samp = win_mel * hop_length
    seg = y[start_samp:start_samp + length_samp].contiguous()
    
    return (idx, seg)

def interpolate_nan_pitch(pitch, vmask):
    """
    Interpolate NaN pitch values in voiced frames from neighbors.
    - Between two valid frames: average them
    - Next to one valid frame: copy it
    - Isolated: mark as unvoiced
    
    Returns: interpolated_pitch, updated_vmask
    """
    pitch = pitch.copy()
    vmask = vmask.copy()
    
    nan_voiced = np.where(vmask & np.isnan(pitch))[0]
    
    for idx in nan_voiced:
        left_valid = None
        for i in range(idx-1, -1, -1):
            if vmask[i] and np.isfinite(pitch[i]):
                left_valid = pitch[i]
                break
        
        right_valid = None
        for i in range(idx+1, len(pitch)):
            if vmask[i] and np.isfinite(pitch[i]):
                right_valid = pitch[i]
                break
        
        if left_valid is not None and right_valid is not None:
            pitch[idx] = (left_valid + right_valid) / 2.0
        elif left_valid is not None:
            pitch[idx] = left_valid
        elif right_valid is not None:
            pitch[idx] = right_valid
        else:
            vmask[idx] = False
    
    return pitch, vmask

def song_to_parameters(waveform, fs_audio, vmask, hop_length, win_length, n_fft, pitch_kwargs=None):
    """
    Extract biomechanical parameters from birdsong -- uses model fit to regular air
    """
    pitch_kwargs = {} if pitch_kwargs is None else pitch_kwargs
    song_pitch, _, _, _ = get_pitch(waveform.astype(float), vmask, fs_audio, hop_length, win_length, n_fft, **pitch_kwargs)
    song_pitch, vmask = interpolate_nan_pitch(song_pitch, vmask)  # interpolate any NaN values
    song_amplitude = estimate_amplitude(waveform.astype(float), fs_audio, win_length, hop_length, n_fft)
    song_amplitude = song_amplitude / song_amplitude.max()

    fit_result = fitter.fit_song(
        birdsong_pitch=song_pitch,
        birdsong_amplitude=song_amplitude,
        voicing_mask=vmask,
        lambda_smooth=0.001,
        use_percentile_init=True,
        batch_size=30,
        max_batches=10,
        convergence_threshold=0.001,
        n_jobs=-1,
        verbose=True
    )
    
    return fit_result['alpha'], fit_result['beta'], fit_result

def synthesize_song(alpha, beta_or_d, fs_audio, len_targ, gain=None, 
                    steps_per_sample=8, integrator='heun', device=torch.device('cpu'),
                    sys_pars=None, v_sound=None, input_is_d=True):
    """
    Synthesize birdsong from biomechanical parameters.
    
    Parameters:
    - sys_pars: System parameters dict (optional, uses defaults if None)
    - v_sound: Speed of sound in cm/s (optional, uses DEFAULT_V_SOUND=35000 if None)
    - input_is_d: If True, convert d (0-1 proportional distance) to β before synthesis
    
    Examples:
    # Use defaults (air)
    rec_wav = synthesize_song(alpha, beta, fs_audio, len(y))
    
    # Use heliox
    rec_wav = synthesize_song(alpha, beta, fs_audio, len(y), v_sound=58450)
    
    # Custom system parameters
    custom_pars = DEFAULT_SYS_PARS.copy()
    custom_pars['gamma'] = 25000.
    rec_wav = synthesize_song(alpha, beta, fs_audio, len(y), sys_pars=custom_pars)
    """
    
    # Use defaults if not provided
    if sys_pars is None:
        sys_pars = DEFAULT_SYS_PARS
    if v_sound is None:
        v_sound = DEFAULT_V_SOUND
    
    hop_length = int(0.004 * fs_audio)
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    if input_is_d:
        d = torch.as_tensor(beta_or_d, dtype=torch.float32)
        b_lo = -1/3 + 1e-3
        b_hi = snilc.upper_bound(alpha).detach()
        beta = b_lo + d * (b_hi - b_lo)
    else:
        beta = torch.as_tensor(beta_or_d, dtype=torch.float32)
    
    if gain is None:
        gain = torch.ones_like(alpha)
    else:
        gain = torch.as_tensor(gain, dtype=torch.float32)
    
    vt = ZebraFinchVocalTract(
        sys_pars=sys_pars,
        v_sound=v_sound,
        steps_per_sample=steps_per_sample,
        integrator=integrator,
        hop_length=hop_length,
        fs_audio=fs_audio,
        jitter_sigma=0.0
    ).to(device)
    
    def compute_burn_in(vt, hop_length):
        substeps_per_mel = hop_length * vt.steps_per_sample
        return int(math.ceil(vt.tau / substeps_per_mel)) + 4
    
    def make_chunks(alpha, beta, gain, n_frames=128, burn_in=8):
        Tm = alpha.numel()
        chunks, idx = [], 0
        for s in range(0, Tm, n_frames):
            e = min(s + n_frames, Tm)
            ps = max(0, s - burn_in)
            off = s - ps
            a = alpha[ps:e].contiguous()
            b = beta[ps:e].contiguous()
            g = gain[ps:e].contiguous()
            chunks.append({
                "idx": idx, "a": a, "b": b, "g": g,
                "off_mel": off, "win_mel": (e - s), "Lm": a.numel()
            })
            idx += 1
        return chunks
    
    # GPU path: batch processing
    def synthesizer_gpu(alpha, beta, gain, vt, hop_length, n_frames=128, batch_size=4):
        burn_in = compute_burn_in(vt, hop_length)
        chunks = make_chunks(alpha, beta, gain, n_frames=n_frames, burn_in=burn_in)
        
        if len(chunks) == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        
        results = []
        t0 = time.time()
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Stack batch inputs
            batch_a = torch.stack([c["a"] for c in batch_chunks]).to(device)
            batch_b = torch.stack([c["b"] for c in batch_chunks]).to(device)
            batch_g = torch.stack([c["g"] for c in batch_chunks]).to(device)
            max_Lm = max(c["Lm"] for c in batch_chunks)
            
            # Pad sequences to same length if needed
            if batch_a.size(1) < max_Lm:
                pad_size = max_Lm - batch_a.size(1)
                batch_a = F.pad(batch_a, (0, pad_size))
                batch_b = F.pad(batch_b, (0, pad_size))
                batch_g = F.pad(batch_g, (0, pad_size))
            
            with torch.inference_mode():
                batch_y = vt(batch_a, batch_b, batch_g, max_Lm)
            
            # Extract and trim each segment
            for j, c in enumerate(batch_chunks):
                y = batch_y[j]
                start_samp = c["off_mel"] * hop_length
                length_samp = c["win_mel"] * hop_length
                seg = y[start_samp:start_samp + length_samp].contiguous()
                results.append((c["idx"], seg))
        
        # Sort and concatenate
        results.sort(key=lambda x: x[0])
        segments = [seg for _, seg in results]
        pred_wave = torch.cat(segments, dim=0)
        
        # Ensure correct length
        expected = alpha.numel() * hop_length
        if pred_wave.numel() != expected:
            pred_wave = pred_wave[:expected]
            if pred_wave.numel() < expected:
                pad = expected - pred_wave.numel()
                pred_wave = F.pad(pred_wave, (0, pad))
        
        print(f"Batch VT GPU: {len(chunks)} windows @ {n_frames} frames | "
              f"batch_size={batch_size} | {time.time()-t0:.3f}s | len={pred_wave.numel()}")
        return pred_wave
    
    # CPU path: multiprocessing
    def synthesizer_cpu(alpha, beta, gain, vt, sys_pars, v_sound, fs_audio, hop_length,
                        n_frames=128, num_workers=None):
        burn_in = compute_burn_in(vt, hop_length)
        chunks = make_chunks(alpha, beta, gain, n_frames=n_frames, burn_in=burn_in)
        
        if len(chunks) == 0:
            return torch.empty(0, dtype=torch.float32)
        
        vt_state = vt.state_dict()
        steps_per_sample = vt.steps_per_sample
        integrator = vt.integrator
    
        tasks = []
        for c in chunks:
            tasks.append((
                vt_state, sys_pars, v_sound, steps_per_sample, hop_length, integrator, fs_audio,
                c["a"], c["b"], c["g"], c["Lm"], c["off_mel"], c["win_mel"], c["idx"]
            ))
    
        if num_workers is None:
            num_workers = max(1, min(len(tasks), os.cpu_count() or 1))
    
        ctx = mp.get_context("fork")
        t0 = time.time()
        with ctx.Pool(processes=num_workers, initializer=_init_vt_worker) as pool:
            results = pool.map(_vt_worker, tasks)
    
        results.sort(key=lambda x: x[0])
        segments = [seg for _, seg in results]
        pred_wave = torch.cat(segments, dim=0)
    
        expected = alpha.numel() * hop_length
        if pred_wave.numel() != expected:
            pred_wave = pred_wave[:expected]
            if pred_wave.numel() < expected:
                pad = expected - pred_wave.numel()
                pred_wave = F.pad(pred_wave, (0, pad))
        
        print(f"Parallel VT CPU: {len(tasks)} windows @ {n_frames} frames | "
              f"workers={num_workers} | {time.time()-t0:.3f}s | len={pred_wave.numel()}")
        return pred_wave
    
    N_FRAMES = 128
    if device.type == 'cuda':
        BATCH_SIZE = 32  # tune based on GPU memory
        y = synthesizer_gpu(alpha, beta, gain, vt=vt, hop_length=hop_length,
                           n_frames=N_FRAMES, batch_size=BATCH_SIZE)
    else:
        NUM_WORKERS = min(len(alpha) // N_FRAMES + 1, (os.cpu_count() or 1))
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        
        y = synthesizer_cpu(alpha, beta, gain, vt=vt, sys_pars=sys_pars, v_sound=v_sound,
                            fs_audio=fs_audio, hop_length=hop_length, n_frames=N_FRAMES, 
                            num_workers=NUM_WORKERS)
    
    rec_wav = y[:len_targ].cpu().numpy()
    return rec_wav
