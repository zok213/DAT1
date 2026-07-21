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
values = [31.0, 22.0, 25.0]
colors = ['#2ca02c', '#1f77b4', '#d62728']
bars = ax.bar(labels, values, color=colors, alpha=0.8)
ax.set_ylabel('Frames per Second (Higher is Better)')
ax.set_title('Pipeline Throughput (FPS)')
ax.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Real-time Target (30 FPS)')
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

# 4b. FP32 vs INT8/FP16 Latency (Grouped Bar)
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['NVIDIA Jetson', 'Qualcomm RB3', 'Radxa CM5']
fp32_latencies = [37.0, 280.0, 450.0]  # FP32 GPU, FP32 CPU, FP32 CPU
opt_latencies = [18.5, 23.0, 38.0]     # FP16 TensorRT, INT8 DSP, INT8 NPU

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, fp32_latencies, width, label='FP32 (Unoptimized CPU/GPU)', color='#ff7f0e', alpha=0.8)
rects2 = ax.bar(x + width/2, opt_latencies, width, label='FP16/INT8 (TensorRT/DSP/NPU)', color='#2ca02c', alpha=0.8)

ax.set_ylabel('DINOv2 Latency (ms) - Log Scale')
ax.set_title('The Cost of Unoptimized FP32 vs Edge Quantization')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log') # Use log scale because 450ms is huge compared to 18.5ms
ax.legend()

def add_labels_log(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.1f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

add_labels_log(ax, rects1)
add_labels_log(ax, rects2)

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

print("All plots generated successfully in assets/ directory!")
