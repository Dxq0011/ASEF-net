"""Efficiency benchmark: parameters, latency, throughput, VRAM and module FLOPs.

Follows the measurement protocol of Sec. 4.9:
  - Latency is the mean wall-clock time of independent single-frame forward
    inferences (after warm-up, with device synchronization).
  - FPS is throughput from a separate continuous-batch run; because scheduling
    differs, it is measured independently and is not the reciprocal of latency.
  - Peak VRAM is the peak CUDA memory allocated by one forward pass.
  - FLOPs are profiled per module with torch.profiler and attributed following
    the manuscript's accounting: encoder-associated work to the encoder, the
    repeated dynamics entry to the ODE module, per-stage decoding to the
    decoder and the fusion attention to reconstruction.

All measured values depend on the GPU, driver and PyTorch build; report the
numbers produced by your own hardware.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from asef.model import ASEFConfig, ASEFNet
from asef.utils import load_config, make_model, make_device, load_checkpoint


def profile_flops(fn) -> int:
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with torch.inference_mode(), profile(activities=acts, with_flops=True) as prof:
        fn()
    return int(sum(e.flops for e in prof.key_averages()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/lol_v1.yaml')
    ap.add_argument('--checkpoint', help='optional weights; random weights otherwise')
    ap.add_argument('--stages', type=int, help='override model.stages (T)')
    ap.add_argument('--rk4-steps', type=int, help='override model.rk4_steps (N)')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=100, help='timed single-frame inferences')
    ap.add_argument('--throughput-seconds', type=float, default=10.0,
                    help='wall-clock duration of the continuous-batch FPS run')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no-flops', action='store_true', help='skip per-module FLOPs profiling')
    ap.add_argument('--out', default='results/benchmark.json')
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stages:
        cfg['model']['stages'] = args.stages
    if args.rk4_steps:
        cfg['model']['rk4_steps'] = args.rk4_steps
    device = make_device(args.device)
    model = make_model(cfg).to(device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    stages, steps = model.cfg.stages, model.cfg.rk4_steps
    nominal = 4 * stages * steps
    image = torch.rand(1, 3, args.size, args.size, device=device)

    params = sum(p.numel() for p in model.parameters())
    report = {'device': str(device), 'input': f'{args.size}x{args.size}',
              'stages_T': stages, 'rk4_steps_N': steps,
              'parameters': params, 'parameters_millions': round(params / 1e6, 6),
              'nominal_rk4_slope_evaluations': nominal}

    # Latency: independent single-frame inferences.
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(image)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    report['latency_ms_mean'] = 1e3 * sum(times) / len(times)

    # Throughput: separate continuous-batch run.
    with torch.inference_mode():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        count = 0
        while time.perf_counter() - t0 < args.throughput_seconds:
            model(image)
            count += 1
            if device.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    report['fps_continuous'] = count / elapsed

    # Peak VRAM of one forward pass.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(image)
        report['peak_vram_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20

    # General-entry RHS count of one forward pass (equals the nominal 4TN here).
    with torch.inference_mode():
        model(image)
    report['general_entry_rhs_calls'] = model.count_rhs_calls

    if not args.no_flops:
        with torch.inference_mode():
            z = model.encoder(image)
            v = torch.nn.functional.softplus(model.vbranch(z))
            n = torch.sigmoid(model.nbranch(z))
            c = torch.sigmoid(model.cbranch(z))
            p = model.position(0.5, z)
            stage_images = [model.decoder(z) for _ in range(stages)]
            logits = model.attention(torch.cat(stage_images, dim=1))
        f_encoder = profile_flops(lambda: model.encoder(image))
        f_dynamics = profile_flops(lambda: model.dynamics(z, n, c, p))
        f_piemf = profile_flops(lambda: model.piemf(z))
        f_position = profile_flops(lambda: model.position(0.5, z))
        f_decoder = profile_flops(lambda: model.decoder(z))
        f_attention = profile_flops(lambda: model.attention(
            torch.cat(stage_images, dim=1)))
        g = 1e9
        ode_flops = (f_dynamics + f_piemf + f_position) * nominal
        total = f_encoder + ode_flops + f_decoder * stages + f_attention
        report['flops_G'] = {
            'encoder': f_encoder / g,
            'ode': ode_flops / g,
            'decoder': f_decoder * stages / g,
            'fusion': f_attention / g,
            'total': total / g,
            'nominal_rk4_evaluations': nominal}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
