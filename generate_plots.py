import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Create assets directory
os.makedirs("assets", exist_ok=True)

# Set global style for professional look
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 14})

def add_labels(ax, bars, fmt='%.1f'):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt % height,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

# 1. Quantization Latency
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Jetson\n(RT FP16)', 'Qualcomm\n(W8A8)', 'Qualcomm\n(W8A16)', 'Radxa\n(RKNN W8A8)']
values = [18.5, 23.0, 41.5, 38.0]
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
bars = ax.bar(labels, values, color=colors, alpha=0.8)
ax.set_ylabel('Latency (ms) - Lower is Better')
ax.set_title('DINOv2 Latency by Quantization & Framework')
add_labels(ax, bars)
plt.tight_layout()
plt.savefig('assets/quantization_latency.png', dpi=300)
plt.close()

# 2. Throughput (FPS)
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Jetson\n(15W Throttled)', 'Qualcomm\n(Native)', 'Radxa\n(Native)']
values = [31.2, 22.4, 25.3]
colors = ['#2ca02c', '#1f77b4', '#d62728']
bars = ax.bar(labels, values, color=colors, alpha=0.8)
ax.set_ylabel('Frames per Second (Higher is Better)')
ax.set_title('Pipeline Throughput (FPS)')
ax.axhline(y=24, color='r', linestyle='--', alpha=0.5, label='Real-time Target (24 FPS)')
ax.legend()
add_labels(ax, bars)
plt.tight_layout()
plt.savefig('assets/throughput_fps.png', dpi=300)
plt.close()

# 3. Power Consumption (Watts)
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Jetson\n(15W Mode)', 'Qualcomm\nRB3 Gen2', 'Radxa\nCM5']
values = [15.0, 2.8, 6.0]
colors = ['#2ca02c', '#1f77b4', '#d62728']
bars = ax.bar(labels, values, color=colors, alpha=0.8)
ax.set_ylabel('Estimated Watts (Lower is Better)')
ax.set_title('System Power Consumption')
add_labels(ax, bars)
plt.tight_layout()
plt.savefig('assets/power_consumption.png', dpi=300)
plt.close()

# 4. Impact of TTA (QWK)
fig, ax = plt.subplots(figsize=(6, 6))
labels = ['Baseline', '+ Train-Time\nAugmentation']
values = [0.774, 0.849]
bars = ax.bar(labels, values, color=['#7f7f7f', '#9467bd'], alpha=0.8, width=0.5)
ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
ax.set_title('Impact of TTA on Accuracy')
ax.set_ylim(0.7, 0.9)
add_labels(ax, bars, fmt='%.3f')
plt.tight_layout()
plt.savefig('assets/tta_qwk.png', dpi=300)
plt.close()

# 4b. Comprehensive Framework & Quantization Latency (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 7))
labels = [
    'Radxa CPU\n(Native FP32)', 
    'Qualcomm CPU\n(Native FP32)', 
    'Qualcomm DSP\n(TFLite QNN W8A16)',
    'Radxa NPU\n(RKNN INT8 W8A8)',
    'Jetson GPU\n(Native FP32)',
    'Qualcomm DSP\n(TFLite QNN W8A8)',
    'Jetson GPU\n(TensorRT FP16)'
]
latencies = [450.0, 280.0, 41.5, 38.0, 37.0, 23.0, 18.5]
# Color code by hardware vendor: Radxa=Red, Qualcomm=Blue, NVIDIA=Green
colors = ['#d62728', '#1f77b4', '#1f77b4', '#d62728', '#2ca02c', '#1f77b4', '#2ca02c']

bars = ax.bar(labels, latencies, color=colors, alpha=0.8)

ax.set_ylabel('DINOv2 Latency (ms) - Log Scale')
ax.set_title('The Cost of FP32 vs Edge Quantization (DINOv2)')
ax.set_yscale('log')

def add_labels_log(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.1f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

add_labels_log(ax, bars)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('assets/fp32_vs_optimized.png', dpi=300)
plt.close()

# 5. Latency Breakdown (Stacked Bar)
fig, ax = plt.subplots(figsize=(10, 7))
labels = ['NVIDIA Jetson\n(15W)', 'Qualcomm RB3\nGen2', 'Radxa CM5\n(RK3588)']
decode = np.array([4.0, 11.2, 8.0])
resize = np.array([0.5, 1.1, 1.5])
yolo = np.array([11.0, 8.6, 12.5])
dino = np.array([18.5, 23.0, 38.0])
cpu = np.array([1.5, 1.5, 1.8])

width = 0.5
p1 = ax.bar(labels, decode, width, label='HW Decode', color='#1f77b4')
p2 = ax.bar(labels, resize, width, bottom=decode, label='Memory Resize', color='#ff7f0e')
p3 = ax.bar(labels, yolo, width, bottom=decode+resize, label='YOLOv8', color='#2ca02c')
p4 = ax.bar(labels, dino, width, bottom=decode+resize+yolo, label='DINOv2 Exec.', color='#d62728')
p5 = ax.bar(labels, cpu, width, bottom=decode+resize+yolo+dino, label='CPU Head', color='#9467bd')

ax.set_ylabel('Latency (ms) - Lower is Better')
ax.set_title('Hardware Latency Profiling (Per Frame Breakdown)')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Add total labels
totals = decode + resize + yolo + dino + cpu
for i, total in enumerate(totals):
    ax.annotate('%.1f ms total' % total,
                xy=(i, total),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/latency_breakdown.png', dpi=300)
plt.close()

# 6. Comprehensive YOLOv8 Object Detection Latency
fig, ax = plt.subplots(figsize=(12, 7))
labels = [
    'Radxa CPU\n(Native FP32)', 
    'Qualcomm CPU\n(Native FP32)', 
    'Jetson GPU\n(Native FP32)',
    'Radxa NPU\n(RKNN INT8 W8A8)',
    'Jetson GPU\n(TensorRT INT8)',
    'Qualcomm DSP\n(TFLite QNN W8A8)'
]
latencies = [120.0, 85.0, 18.0, 12.5, 11.0, 8.6]
# Color code by hardware vendor: Radxa=Red, Qualcomm=Blue, NVIDIA=Green
colors = ['#d62728', '#1f77b4', '#2ca02c', '#d62728', '#2ca02c', '#1f77b4']

bars = ax.bar(labels, latencies, color=colors, alpha=0.8)

ax.set_ylabel('YOLOv8 Latency (ms) - Log Scale')
ax.set_title('The Cost of FP32 vs Edge Quantization (YOLOv8)')
ax.set_yscale('log')

add_labels_log(ax, bars)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('assets/yolov8_latency.png', dpi=300)
plt.close()

print("All plots generated successfully in assets/ directory!")
