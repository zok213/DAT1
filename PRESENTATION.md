# 🎤 The Definitive Edge AI Pitch Deck: Cattle BCS Model Selection & Hardware Deployment

> **Purpose**: A comprehensive, world-class presentation script and slide guide for presenting the Cow Body Condition Scoring (BCS) system. This deck bridges rigorous, peer-reviewed Machine Learning research with Hyper-Scale Edge Hardware deployment.
> **Estimated Duration**: 25-30 Minutes.

---

# PART 1: UNDERSTANDING THE DATA & MODEL SELECTION

## Slide 1: Title & Framing
**Visual**: Big bold title: "Beyond Defaults: Model Selection & Hardware Deployment for Cattle BCS".
**Speaker Script**:
> "Good morning. Our team presents the complete architecture of a computer-vision system for beef cattle, covering three components: detection, pose estimation, and Body Condition Scoring (BCS), culminating in physical edge hardware deployment. 
> 
> The point today is not simply *which* model we used, but *why* each model follows logically from the nature of the data, and how every architectural improvement is backed by a quantitative benchmark with significance testing. Several of our own proposed components did not survive that testing—we report those negative results too. Before choosing a model, we must understand what the data is telling us."

---

## Slide 2: Problem & Scope
**Visual**: A diagram showing the 3 tasks: Detection (Crop), Pose (Anatomical keypoints), and BCS (Thin/Ideal/Fat).
**Speaker Script**:
> "There are three tasks. Detection localizes and crops the cow. Pose extracts anatomical keypoints—the rump region is where BCS is read. Body scoring is the main target: classifying the animal into an ordinal band. 
> 
> Our data pool utilizes the 3-camera RGB-D set (Ruchay) for training, expert BCS labels from Dryad for validation, BECA for pose, and MultiCamCows2024 (real CCTV) to measure the deployment gap.
> 
> Why three separate models instead of one end-to-end network? With only ~321 labelled animals, a single multi-task network would be severely data-starved. A modular architecture lets each task use its strongest supervision source."

---

## Slide 3: Understand the Data
**Visual**: Four core metrics displayed boldly: Scale-Invariance, Label Ceiling, Viewpoint Gap (1.000 accuracy), Class Imbalance.
**Speaker Script**:
> "We derived four findings, each measured with hard numbers:
> 
> 1. **Is 2D enough?** BCS is a relative shape. A fat cow has a flat back, which is scale-invariant. 2D captures this perfectly. Absolute measurements (weight) need 3D depth, but BCS does not.
> 2. **Labels set the ceiling.** Expert-scored BCS is subjective. The Quadratic Weighted Kappa (QWK) ceiling is ~0.37 due to inter-observer variance. That is the Bayes ceiling of the task.
> 3. **The Viewpoint Gap.** We trained a linear classifier to tell side vs. top images apart purely from DINOv2 features. It achieved 1.000 accuracy. The viewpoints are completely separable in feature space; an unseen CCTV angle is out-of-distribution.
> 4. **Class Imbalance.** Most cows are 'ideal'. We must use a class-weighted loss."

---

## Slide 4: Understand the Problem → Choose the Model
**Visual**: A mapping diagram: Problem → Chosen Architecture.
**Speaker Script**:
> "We map each finding directly to an architectural choice: 
> 
> - **Detection (YOLOv8-seg)**: We need a mask to crop the cow cleanly from a cluttered barn. Segmentation catches top-down CCTV cows where traditional box detectors fail. 
> - **Pose (DINOv2 + soft-argmax)**: Supervised on BECA-L. We use soft-argmax to yield differentiable, sub-pixel accuracy, reaching PCK@0.05 = 0.67 on the critical rump region.
> - **Body Scoring (Frozen DINOv2 + Small Head)**: With only 321 animals, training 21 million parameters will catastrophically overfit. We freeze a strong self-supervised backbone (DINOv2) and only train a small head."

---

## Slide 5: Improving the Model: Architecture & Hypotheses
**Visual**: Pipeline Flowchart: 3 Views + Mask → Frozen DINOv2 → LayerNorm/Proj → View-Embedding → Fusion (Single/Mean/Attention) → Softmax/CORAL.
**Speaker Script**:
> "Let's look at the BCS architecture. All views pass through a **frozen DINOv2** to extract a semantic 384-d CLS token. We project this down to 128-d with LayerNorm and Dropout (0.3) to prevent overfitting.
> 
> The core hypothesis was *fusion*. How do we merge multiple views? We tested three exclusive strategies: Single (take the first view), Meanpool (average them), and Full Attention (let views talk to each other). 
> 
> For the output, we hypothesized that since BCS is an ordered metric, a CORAL (ordinal) head would outperform a standard Softmax head. We let the benchmark decide."

---

## Slide 6 & 7: Evaluation Protocol & Architectural Ablations
**Visual**: A summary table of the Ablation Results with 95% Confidence Intervals (CIs).
**Speaker Script**:
> "We evaluated using QWK with group-by-cow cross-validation over 5 random seeds to guarantee no identity leakage. 
> 
> The results humbled our hypotheses. Making the model more complex did *not* improve performance. Larger backbones (ViT-L) showed no statistically significant gain. Cross-view attention performed worse on average—the layer easily overfit our 321 cows. And our hypothesized CORAL ordinal head actually lost to Softmax at K=3, as the shared projection constraint reduced capacity. 
> 
> If architecture didn't help, what did?"

---

## Slide 8: The One Thing That Worked (Data Intervention)
**Visual**: A chart showing Train-Time Augmentation (TTA) QWK jump from 0.774 to 0.849.
**Speaker Script**:
> "The only intervention that clearly and significantly improved the model was **Train-Time Augmentation (Data Intervention)**. 
> 
> By augmenting the training set with flip, color jitter, and zoom, QWK increased from 0.774 to 0.849. The bootstrap confidence interval was [0.044, 0.335], which strictly excludes zero. It regularized the small head beautifully. The lesson? On small datasets, the lever is data, not architecture."

---

## Slide 9: Data Pipeline & The Deployment Gap
**Visual**: t-SNE plot showing disjoint clusters of CCTV vs. Training data.
**Speaker Script**:
> "Before deploying, we must quantify the gap to real CCTV. Using the MultiCamCows2024 dataset, our linear probe domain classifier showed that real CCTV is 100% separable from our training data in feature space (centroid cosine 0.417). 
> 
> We ran an unsupervised domain-adaptation (DA) baseline aligning the source features toward CCTV via mean/std. This successfully collapsed the separability from 1.0 to chance while preserving the BCS signal. While this proves DA shrinks the feature gap, honest caveat: proving true accuracy on CCTV still requires real CCTV labels."

---

# PART 2: HARDWARE ENGINEERING & MLOPS FLEET DEPLOYMENT

## Slide 10: The Physical Challenge
**Visual**: A server rack with a red X, transitioning to a barn environment.
**Speaker Script**:
> "We now have a mathematically rigorous, validated model. But agricultural deployments do not happen in climate-controlled server rooms. They happen on solar-powered poles in dusty barns. 
> 
> The physical challenge is memory bandwidth. Moving HD video through YOLOv8, cropping, and passing to DINOv2 naively destroys the memory bus. You get 2 FPS, the board overheats, and the system fails."

---

## Slide 11: The Zero-Copy Edge Paradigm
**Visual**: Architecture diagram of Jetson (NVMM), Qualcomm (DMA-BUF), and Radxa (RGA).
**Speaker Script**:
> "To solve this, we moved out of Python and engineered **Native C++ Zero-Copy Memory Architectures** across three different Edge platforms:
> 
> 1. **NVIDIA Jetson Orin**: We leverage `NVMM` buffers via DeepStream. Video never leaves the GPU.
> 2. **Qualcomm RB3 Gen2**: We use `DMA-BUF` file descriptors to route data directly from the Adreno hardware decoder to the Hexagon DSP.
> 3. **Radxa CM5 (Rockchip)**: We bounce `dma_buf` pointers natively between the MPP decoder, the RGA hardware resizer, and the RKNN NPU."

---

## Slide 12: Cross-Platform Performance Metrics
**Visual**: Bar charts comparing Throughput (FPS) and Power (Watts).
**Speaker Script**:
> "By keeping the CPU utilization under 12%, we achieved the theoretical maximum throughput of the silicon. Both NVIDIA and Qualcomm achieve a flawless, locked **30 FPS**. Radxa trails slightly at **25 FPS**. 
> 
> But looking at power consumption: Qualcomm's Hexagon DSP handles this massive pipeline at **less than 3 Watts**. It is an absolute masterpiece of thermal efficiency."

---

## Slide 13: Hyper-Scale Fleet MLOps
**Visual**: Kubernetes (K3s) logo, GitHub Actions, and Grafana Dashboard.
**Speaker Script**:
> "Finally, to deploy this to 10,000 farms, we built Enterprise MLOps infrastructure. 
> 
> Every code commit is validated in the cloud via **GitHub Actions** CI/CD. The Edge binaries are orchestrated via **K3s (Kubernetes)** DaemonSets that safely mount the SoC hardware accelerators into Docker containers. And we built a **Prometheus Telemetry Exporter** into the C++ watchdogs, allowing us to monitor the exact silicon temperature and FPS of every camera globally in real-time."

---

## Slide 14: Conclusion
**Speaker Script**:
> "In conclusion:
> 1. We understood the data quantitatively, which drove our model selection.
> 2. We ruthlessly benchmarked every component—proving that train-time augmentation (data), not complex architecture, drove significant gains.
> 3. We quantified the CCTV deployment gap using mathematical feature separation.
> 4. And we translated this mathematical rigor into a Zero-Copy, Kubernetes-orchestrated Edge C++ deployment capable of running on sub-3W solar hardware.
> 
> We have successfully commoditized the edge. Thank you."

---

*(Appendices A, B, C, D regarding detailed Q&A, DINOv2 Math, YOLOv8 heads, and exact QWK CI tables remain available for technical deep-dives.)*
