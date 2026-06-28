#!/usr/bin/env python3
import os, sys, json, time, subprocess, types, inspect, threading, re
from contextlib import contextmanager
import numpy as np

# ══════════════════════════════════════════════════════════════
# TF2 → TF1 COMPREHENSIVE PATCH
# ══════════════════════════════════════════════════════════════
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 1) TF1 API aliases
for _a in ['variable_scope', 'get_variable', 'GraphKeys', 'constant_initializer',
           'zeros_initializer', 'stop_gradient', 'identity', 'get_variable_scope',
           'control_dependencies', 'random_uniform']:
    if not hasattr(tf, _a) and hasattr(tf.compat.v1, _a):
        setattr(tf, _a, getattr(tf.compat.v1, _a))

# 2) Image resize
if not hasattr(tf.image, 'resize_bilinear'):
    tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear
if not hasattr(tf.image, 'resize_nearest_neighbor'):
    tf.image.resize_nearest_neighbor = tf.compat.v1.image.resize_nearest_neighbor

# 3) extract_image_patches
if not hasattr(tf, 'extract_image_patches'):
    def _extract_patches(images, ksizes, strides, rates, padding, **kw):
        return tf.image.extract_patches(
            images, sizes=ksizes, strides=strides, rates=rates, padding=padding)
    tf.extract_image_patches = _extract_patches

# 4) py_func
if not hasattr(tf, 'py_func'):
    tf.py_func = tf.compat.v1.py_func

# 5) keep_dims → keepdims
for _fn in ['reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min']:
    _orig = getattr(tf, _fn)
    def _make_patched(orig):
        def _patched(*args, **kwargs):
            if 'keep_dims' in kwargs:
                kwargs['keepdims'] = kwargs.pop('keep_dims')
            return orig(*args, **kwargs)
        return _patched
    setattr(tf, _fn, _make_patched(_orig))

# 6) tf.layers hybrid (function-style TF1 + class-style Keras3)
class _LayersCompat:
    Conv2D = tf.keras.layers.Conv2D
    Dense  = tf.keras.layers.Dense
    BatchNormalization = tf.keras.layers.BatchNormalization

    @staticmethod
    def conv2d(inputs, filters, kernel_size, strides=1, padding='valid',
               dilation_rate=1, activation=None, use_bias=True,
               kernel_initializer=None, bias_initializer=None,
               name=None, reuse=None, trainable=True, **kwargs):
        if isinstance(kernel_size, int): kernel_size = [kernel_size, kernel_size]
        if isinstance(strides, int):     strides = [strides, strides]
        if isinstance(dilation_rate, int): dilation_rate = [dilation_rate, dilation_rate]
        pad = padding.upper() if isinstance(padding, str) else padding
        in_ch = inputs.get_shape().as_list()[-1]
        with tf.compat.v1.variable_scope(name or 'conv2d', reuse=tf.compat.v1.AUTO_REUSE):
            W = tf.compat.v1.get_variable('kernel',
                shape=[kernel_size[0], kernel_size[1], in_ch, filters],
                initializer=kernel_initializer or tf.keras.initializers.GlorotUniform(),
                trainable=trainable)
            out = tf.nn.conv2d(inputs, W,
                strides=[1, strides[0], strides[1], 1],
                padding=pad, dilations=dilation_rate)
            if use_bias:
                b = tf.compat.v1.get_variable('bias', shape=[filters],
                    initializer=bias_initializer or tf.keras.initializers.Zeros(),
                    trainable=trainable)
                out = tf.nn.bias_add(out, b)
            if activation is not None:
                out = activation(out)
            return out

    @staticmethod
    def conv2d_transpose(inputs, filters, kernel_size, strides=1, padding='valid',
                         activation=None, use_bias=True,
                         name=None, reuse=None, trainable=True, **kwargs):
        if isinstance(kernel_size, int): kernel_size = [kernel_size, kernel_size]
        if isinstance(strides, int):     strides = [strides, strides]
        pad = padding.upper() if isinstance(padding, str) else padding
        in_shape = inputs.get_shape().as_list()
        in_ch = in_shape[-1]
        with tf.compat.v1.variable_scope(name or 'conv2d_transpose', reuse=tf.compat.v1.AUTO_REUSE):
            W = tf.compat.v1.get_variable('kernel',
                shape=[kernel_size[0], kernel_size[1], filters, in_ch],
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=trainable)
            bs = tf.shape(inputs)[0]
            h  = (in_shape[1] or tf.shape(inputs)[1]) * strides[0]
            w  = (in_shape[2] or tf.shape(inputs)[2]) * strides[1]
            out = tf.nn.conv2d_transpose(inputs, W,
                output_shape=tf.stack([bs, h, w, filters]),
                strides=[1, strides[0], strides[1], 1], padding=pad)
            if use_bias:
                b = tf.compat.v1.get_variable('bias', shape=[filters],
                    initializer=tf.keras.initializers.Zeros(), trainable=trainable)
                out = tf.nn.bias_add(out, b)
            if activation is not None:
                out = activation(out)
            return out

    @staticmethod
    def batch_normalization(inputs, training=False, name=None, reuse=None, **kwargs):
        with tf.compat.v1.variable_scope(name or 'batch_norm', reuse=tf.compat.v1.AUTO_REUSE):
            return tf.keras.layers.BatchNormalization()(inputs, training=training)

if not hasattr(tf, 'layers'):
    tf.layers = _LayersCompat()

# 7) arg_scope
@contextmanager
def _arg_scope(list_ops, **kwargs):
    saved = {}
    for func in list_ops:
        saved[func] = (func.__defaults__, func.__kwdefaults__)
        sig = inspect.signature(func)
        default_params = [(n, p) for n, p in sig.parameters.items()
                          if p.default is not inspect.Parameter.empty
                          and p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        new_defaults = list(func.__defaults__ or [])
        for i, (n_, _) in enumerate(default_params):
            if n_ in kwargs:
                new_defaults[i] = kwargs[n_]
        func.__defaults__ = tuple(new_defaults) if new_defaults else func.__defaults__
        new_kw = dict(func.__kwdefaults__ or {})
        for n_ in kwargs:
            if n_ in [k for k, p in sig.parameters.items()
                      if p.kind == inspect.Parameter.KEYWORD_ONLY]:
                new_kw[n_] = kwargs[n_]
        func.__kwdefaults__ = new_kw if new_kw else None
    try:
        yield
    finally:
        for func in list_ops:
            func.__defaults__, func.__kwdefaults__ = saved[func]

# 8) tensorflow.contrib fake module
if 'tensorflow.contrib' not in sys.modules:
    _c   = types.ModuleType('tensorflow.contrib')
    _cfw = types.ModuleType('tensorflow.contrib.framework')
    _cfp = types.ModuleType('tensorflow.contrib.framework.python')
    _cfo = types.ModuleType('tensorflow.contrib.framework.python.ops')
    _cfo.arg_scope    = _arg_scope
    _cfo.add_arg_scope = lambda f: f
    class _FakeLayers:
        def l2_regularizer(self, wd): return tf.keras.regularizers.l2(wd)
        def flatten(self, x): return tf.reshape(x, [tf.shape(x)[0], -1])
        def xavier_initializer_conv2d(self): return tf.keras.initializers.GlorotUniform()
        def xavier_initializer(self):        return tf.keras.initializers.GlorotUniform()
    _c.layers = _FakeLayers(); _c.framework = _cfw
    _cfw.python = _cfp;        _cfp.ops = _cfo
    tf.contrib = _c
    sys.modules['tensorflow.contrib']                      = _c
    sys.modules['tensorflow.contrib.framework']            = _cfw
    sys.modules['tensorflow.contrib.framework.python']     = _cfp
    sys.modules['tensorflow.contrib.framework.python.ops'] = _cfo

# 9) Summary no-ops (no necesarios para inferencia)
for _sm in ['image', 'scalar', 'histogram', 'merge_all', 'text']:
    if not hasattr(tf.summary, _sm):
        setattr(tf.summary, _sm, lambda *a, **k: None)
# ══════════════════════════════════════════════════════════════


# ── TEGRASTATS MONITOR ──────────────────────────────────────
class TegraStatsMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.lines = []
        self.proc  = None
        self._stop = False

    def start(self):
        self.lines = []; self._stop = False
        self.proc = subprocess.Popen(
            ['tegrastats', '--interval', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if self._stop: break
            self.lines.append(line.decode('utf-8', errors='ignore'))

    def stop(self):
        self._stop = True
        if self.proc:
            self.proc.terminate(); self.proc.wait()

    def _parse_mw(self, key):
        vals = []
        for line in self.lines:
            m = re.search(rf'{key} (\d+)mW', line)
            if m: vals.append(float(m.group(1)) / 1000.0)
        return float(sum(vals)/len(vals)) if vals else None

    def total_power_w(self):      return self._parse_mw('VDD_IN')
    def gpu_cpu_power_w(self):    return self._parse_mw('VDD_CPU_GPU_CV')

    def ram_used_mb(self):
        for line in reversed(self.lines):
            m = re.search(r'RAM (\d+)/\d+MB', line)
            if m: return float(m.group(1))
        return None
# ────────────────────────────────────────────────────────────

PROJECT_ROOT = "/home/sigcomjetson/Desktop/CloudGAN"
CHECKPOINT_DIR = f"{PROJECT_ROOT}/weights/SN_PatchGAN"
CHECKPOINT_PREFIX = "snap-1132000"
OUT_DIR = f"{PROJECT_ROOT}/evaluation/results"

sys.path.insert(0, PROJECT_ROOT)
from cloud_removal.inpaint_model import InpaintCAModel

class FLAGS:
    guided = False; edge_threshold = 0.6; padding = "SAME"; viz_max_out = 1
    height = 256;   width = 256;          img_shapes = [256, 256, 3]
    max_delta_height = 32; max_delta_width = 32
    vertical_margin = 0;   horizontal_margin = 0; batch_size = 1
    l1_loss_alpha = 1.0;   gan_loss_alpha = 1.0
    ae_loss = True;        gan = "sngan"; gan_with_mask = True; random_seed = False

def count_params():
    total = 0
    for v in tf.compat.v1.trainable_variables():
        n = 1
        for d in v.get_shape().as_list():
            if d is not None: n *= d
        total += n
    return total

def get_checkpoint():
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    return latest if latest else os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

def get_gpu_power():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"])
        return float(out.decode().strip().split("\n")[0])
    except: return None

def get_gpu_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"])
        return float(out.decode().strip().split("\n")[0])
    except: return None

def evaluate():
    print("=" * 60)
    print("SN-PATCHGAN EVALUATION — Jetson Orin Nano")
    print("=" * 60)

    total_size = sum(os.path.getsize(os.path.join(CHECKPOINT_DIR, f))
                     for f in os.listdir(CHECKPOINT_DIR) if CHECKPOINT_PREFIX in f)
    size_mb = total_size / (1024**2)
    print(f"[1/6] Size: {size_mb:.2f} MB")

    print("[2/6] Building graph...")
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tf.compat.v1.placeholder(
            tf.float32, shape=[1, 256, 256, 3], name="input")
        model  = InpaintCAModel()
        output = model.build_static_infer_graph(FLAGS, input_ph, name="val")
        total_params = count_params()
        print(f"  Parameters: {total_params:,}")

        print("[3/6] Computing FLOPs...")
        try:
            from tensorflow.python.profiler import model_analyzer, option_builder
            opts   = option_builder.ProfileOptionBuilder.float_operation()
            flops  = model_analyzer.profile(graph, options=opts)
            total_flops = flops.total_float_ops if flops else 0
        except Exception as e:
            print(f"  FLOPs no disponibles: {e}"); total_flops = 0
        print(f"  FLOPs: {total_flops:,}")

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, get_checkpoint())
            print("[4/6] Checkpoint restored")

            dummy = np.random.randint(0, 256, (1,256,256,3)).astype(np.float32)
            feed  = {input_ph: dummy}
            for _ in range(10): sess.run(output, feed_dict=feed)

            print("[5/6] Measuring inference (100 runs) + tegrastats...")
            monitor = TegraStatsMonitor(interval_ms=100)
            monitor.start()
            time.sleep(0.5)  # baseline

            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(output, feed_dict=feed)
                times.append((time.perf_counter() - t0) * 1000)

            time.sleep(0.3)
            monitor.stop()

            mean_ms        = float(np.mean(times))
            fps            = float(1000.0 / mean_ms)
            total_power    = monitor.total_power_w()
            gpu_cpu_power  = monitor.gpu_cpu_power_w()
            ram_used       = monitor.ram_used_mb()
            avg_power      = gpu_cpu_power  # VDD_CPU_GPU_CV para el calculo de energia
            energy         = avg_power * (mean_ms / 1000.0) if avg_power else None

            print(f"  Latency: {mean_ms:.2f} ms  |  FPS: {fps:.2f}")
            print(f"  Total board power : {total_power:.2f} W"   if total_power   else "  Total power: N/A")
            print(f"  CPU+GPU+CV power  : {gpu_cpu_power:.2f} W" if gpu_cpu_power  else "  GPU power: N/A")
            print(f"  RAM used          : {ram_used:.0f} MB"      if ram_used      else "  RAM: N/A")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {
        "model": "SN-PatchGAN", "parameters": int(total_params),
        "size_mb": size_mb, "flops": int(total_flops),
        "latency_ms": mean_ms, "fps": fps,
        "ram_used_mb": ram_used,
        "total_board_power_w": total_power,
        "gpu_cpu_power_w": gpu_cpu_power,
        "energy_per_inference_j": energy,
    }
    out_path = os.path.join(OUT_DIR, "snpatchgan_jetson.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    print(f"[6/6] Guardado en {out_path}")
    print("=" * 60)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    evaluate()
