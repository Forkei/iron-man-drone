# Outreach Target List — Iron Man Drone Project

_Last updated: 2026-05-08_

## Ranking Logic

Targets are ranked by a combination of three factors: (1) **technical overlap** — how directly the person's work maps to your current or near-future milestones; (2) **accessibility** — PhD students and postdocs over PIs, people with active public presence over silent ones; and (3) **mutual upside** — whether reaching out gives them a reason to engage (citing their paper, building on their simulator, asking a specific implementation question). The top entries are people you are already building on or whose techniques you intend to adapt directly in M1–M2; lower entries are "warm up for M3–M5" relationships.

---

## 1. Jiayu Chen — PhD Student, Tsinghua University (NICS-EFC)

**Why**: Lead author of SimpleFlight (arXiv 2412.11764, RA-L 2025), the exact sim-to-real RL framework your M1 milestone is reproducing — best possible technical context for a cold email.

**Papers (recent)**:
- _What Matters in Learning A Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study_ (2024, arXiv 2412.11764; accepted RA-L May 2025) — foundational paper for M1; PPO + Omnidrones + Crazyflie real-world validation
- _JuggleRL: Mastering Ball Juggling with a Quadrotor via Deep Reinforcement Learning_ (2025, arXiv 2509.24892) — follow-on agility work from the same group

**Contact**: Cold email — no public social presence found; email contact via corresponding author Chao Yu (zoeyuchao@gmail.com) or institutional route through nicsefc.ee.tsinghua.edu.cn/people/JiayuChen; ask a specific technical question about Omnidrones integration or domain randomisation choices.

**Public presence**: Twitter/X: not found | Bluesky: not found | Web: https://nicsefc.ee.tsinghua.edu.cn/people/JiayuChen | GitHub: https://github.com/thu-uav/SimpleFlight

---

## 2. Qianzhong Chen — PhD Student, Stanford Aero-Astro (Multi-Robot Systems Lab, advisor: Mac Schwager)

**Why**: Author of both GRaD-Nav (IROS 2025) and GRaD-Nav++ (RA-L 2025) — the 3DGS + differentiable RL drone navigation stack that is the foundation for your M5, and the closest existing work to your combined M4+M5 goals; actively posts about this work on Twitter and replied to paper discussions.

**Papers (recent)**:
- _GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics_ (2025, RA-L / arXiv 2506.14009) — VLA drone control trained inside a 3DGS scene; MoE action head; onboard deployment
- _GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics_ (2025, IROS 2025 / arXiv 2503.03984) — base version; differentiable RL in 3DGS sim

**Contact**: Social (Twitter/X) then email — he is active on Twitter (@QianzhongChen), publicly discussed visa issues preventing IROS attendance (showing engagement), and has a public email qchen23@stanford.edu; open with a tweet reply before emailing.

**Public presence**: Twitter/X: @QianzhongChen | Bluesky: not found | Web: https://qianzhong-chen.github.io/ | Email: qchen23@stanford.edu

---

## 3. Jiaxu Xing — PhD Student, University of Zurich (Robotics and Perception Group, advisor: Davide Scaramuzza)

**Why**: Most prolific current PhD student in UZH RPG on agile RL drone control — first or co-first author on CoRL 2024 (bootstrapping RL+IL for vision-based agile flight) and RA-L 2025 (multi-task RL for quadrotors); his multi-task RL setup is directly relevant to your M2 fault-tolerance policy design.

**Papers (recent)**:
- _Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight_ (CoRL 2024, arXiv 2403.12203) — teacher-student RL/IL pipeline for vision-based agile flight; strong real-world results
- _Multi-Task Reinforcement Learning for Quadrotors_ (RA-L 2025, arXiv 2412.12442) — multi-critic MTRL for a single policy across racing, stabilisation, and velocity tracking; shared encoders
- _Student-Informed Teacher Training_ (ICLR 2025 Spotlight, Top 5%) — joint teacher-student training; generalisation advances

**Contact**: Email (jixing@ifi.uzh.ch) — Twitter account (@jixing24) exists but shows zero posts; better to email with a specific technical question about multi-task policy design or their distillation approach.

**Public presence**: Twitter/X: @jixing24 (inactive) | Bluesky: not found | Web: https://jiaxux.ing/ | Email: jixing@ifi.uzh.ch

---

## 4. Artem Lykov — PhD Student, Skolkovo Institute of Science and Technology (ISR Lab, advisor: Dzmitry Tsetserukou)

**Why**: First author of CognitiveDrone (arXiv 2503.01378, March 2025) — the most thorough VLA benchmark for UAV cognitive tasks — and co-author of RaceVLA; his lab is the most active producer of drone VLA papers in 2025, directly relevant to M5; he has a Yandex ML Prize and is internationally connected.

**Papers (recent)**:
- _CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs_ (2025, arXiv 2503.01378) — VLA model + benchmark with 8000+ simulated trajectories; CognitiveDrone-R1 achieves 77.2% success
- _RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour_ (2025, arXiv 2503.02572) — first VLA system for FPV racing; outperforms RT-2 on semantic generalisation
- _UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation_ (HRI 2025) — mission generation via satellite imagery + GPT; 6.5x faster than human operator

**Contact**: Cold email (artem.lykov@skoltech.ru) — no Twitter handle verified; email is publicly listed; Skoltech lab is internationally oriented and the lab's PI actively presents at ICRA.

**Public presence**: Twitter/X: not found | Bluesky: not found | Web: Google Scholar (Y9ZtqH0AAAAJ) | Email: artem.lykov@skoltech.ru

---

## 5. Yunfan Ren — Postdoc, University of Zurich (Robotics and Perception Group, advisor: Davide Scaramuzza)

**Why**: Lead author of "Learning Agile Quadrotor Flight in the Real World" (arXiv 2026, UZH RPG) — introduces Adaptive Temporal Scaling and online residual learning that triple peak speed within 100 seconds of real-world flight, which is the most direct prior work for your M2 adaptive/fault-tolerant policy and future M3 sim-to-real transfer.

**Papers (recent)**:
- _Learning Agile Quadrotor Flight in the Real World_ (2026, arXiv, UZH RPG) — ATS + RASH-BPTT for rapid in-flight policy update; peak speeds triple from 1.9 to 7.3 m/s within 100 s
- Prior work includes 20 papers across Science Robotics, Nature Communications, T-RO (5 papers), RA-L, ICRA, IROS during his PhD

**Contact**: Cold email (yunfan@ifi.uzh.ch) — no social media presence found; postdoc stage typically means more bandwidth for collaborations than senior PIs; cite the real-world adaptation paper specifically.

**Public presence**: Twitter/X: not found | Bluesky: not found | Web: https://renyunfan.cn/ | Email: yunfan@ifi.uzh.ch | GitHub: https://github.com/RENyunfan

---

## 6. Jin Zhou — PhD Student / Researcher, Zhejiang University (College of Control Science and Engineering)

**Why**: Lead author of MAVEN (arXiv 2603.10714, ZJU, March 2026) — a meta-RL framework that adapts in real-time to mass variation of up to 66.7% and single-rotor thrust loss up to 70%, then zero-shot transfers to real hardware; this is exactly the adaptive fault-tolerance mechanism your M2 milestone targets.

**Papers (recent)**:
- _MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers_ (2026, arXiv 2603.10714) — predictive context encoder for latent dynamics inference; demonstrated on mass variation and single-rotor failure; zero-shot sim-to-real

**Contact**: Cold email (j.zhou2020@zju.edu.cn) — email is listed directly in paper; no social media presence found; open with a specific question about the context encoder architecture or the rotor-failure training curriculum.

**Public presence**: Twitter/X: not found | Bluesky: not found | Web: not found | Email: j.zhou2020@zju.edu.cn

---

## 7. Jiehao Chen — PhD Student, Harbin Institute of Technology Shenzhen (School of Intelligence Science and Engineering)

**Why**: Lead author of "Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor Failure" (IROS 2025, arXiv 2503.02649) — proposes a Selector-Controller network trained with hybrid RL+BC+SL that handles the full range from zero-fault to complete single-rotor loss without controller switching; the cleanest existing RL fault-tolerance architecture to study for your M2.

**Papers (recent)**:
- _Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor Failure_ (IROS 2025, arXiv 2503.02649) — unified PFTC policy covering fault-free to complete rotor failure; real-world validated; Selector-Controller network architecture

**Contact**: Cold email via corresponding author YanJie Li (autolyj@hit.edu.cn) — no personal social media found for Jiehao Chen; emailing the corresponding author and cc'ing Jiehao Chen is the best route; IROS acceptance signals the lab is internationally engaged.

**Public presence**: Twitter/X: not found | Bluesky: not found | Web: not found | Corresponding author email: autolyj@hit.edu.cn

---

## 8. Valerii Serpiva — PhD Student, Skolkovo Institute of Science and Technology (ISR Lab, advisor: Dzmitry Tsetserukou)

**Why**: First author of RaceVLA (arXiv 2503.02572, March 2025) — the first VLA-based FPV racing drone system, combining agile flight and language conditioning; serves as the agile-RL + VLA bridge that connects your M2 and M5 directions; the ISR Lab under Tsetserukou is the most prolific drone VLA group globally right now.

**Papers (recent)**:
- _RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour_ (2025, arXiv 2503.02572) — FPV VLA racing; processes first-person video + language commands; outperforms RT-2 and OpenVLA on semantic generalisation
- _CognitiveDrone_ (2025, arXiv 2503.01378) — co-author; VLA benchmark for UAV cognitive tasks

**Contact**: Cold email via Skoltech institutional address (researcher details on ResearchGate: researchgate.net/profile/Valerii-Serpiva-2) — no Twitter/X handle verified; the lab collectively presents at major robotics venues and the ISR Lab page lists contact info.

**Public presence**: Twitter/X: not found | Bluesky: not found | Web: ResearchGate (Valerii-Serpiva-2) | Google Scholar: r1lQTeUAAAAJ

---

## 9. Riku Murai — Postdoc, Imperial College London (Dyson Robotics Laboratory, advisor: Andrew Davison)

**Why**: Co-first author of MonoGS / Gaussian Splatting SLAM (CVPR 2024 Highlight + Best Demo Award) and sole first author of MASt3R-SLAM (CVPR 2025) — the two most impactful Gaussian SLAM papers; your M4 milestone needs a working 3DGS scene representation pipeline and Riku is the most accessible (postdoc, Twitter-active) expert in that exact space.

**Papers (recent)**:
- _Gaussian Splatting SLAM_ (CVPR 2024 Highlight + Best Demo Award, arXiv 2312.06741) — first monocular SLAM with 3D Gaussian Splatting; real-time capable
- _MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors_ (CVPR 2025, arXiv 2412.12392) — dense monocular SLAM at 15 FPS using MASt3R priors; globally consistent poses + dense geometry

**Contact**: Social (Twitter/X) — @rmurai0610 is active; he posts about new SLAM papers and engages with the community; a reply to a relevant tweet or a short public question about drone-facing 3DGS deployment is likely to get noticed; follow up by email (rm3115@ic.ac.uk) if needed.

**Public presence**: Twitter/X: @rmurai0610 | Bluesky: not found | Web: https://rmurai.co.uk/ | Email: rm3115@ic.ac.uk

---

## 10. Yunlong Song — Founding Research Scientist, Genesis AI (prev. PhD at UZH RPG / ETH)

**Why**: Delivered some of the fastest RL-trained drone policies in existence (108 km/h, 12g peak acceleration) as part of the UZH RPG; now at Genesis AI working on robot foundation models; his Actor-Critic MPC paper (T-RO 2025) and prior RL drone racing work are direct technical antecedents to M2; his move to industry makes him potentially more responsive to practical collaboration questions.

**Papers (recent)**:
- _Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning_ (IEEE T-RO 2025) — superhuman performance at up to 21 m/s; bridges MPC and RL for agile flight
- _Learning Quadrotor Control From Visual Features Using Differentiable Simulation_ (ICRA 2025) — differentiable sim for sample-efficient vision-based control
- _Multi-Task Reinforcement Learning for Quadrotors_ (RA-L 2025, co-author with Jiaxu Xing) — multi-task drone policy

**Contact**: Social (Twitter/X) — @realyunlong is the verified handle; he is now at a startup so posting about robotics publicly; engagement on Twitter is the right first step before LinkedIn or email.

**Public presence**: Twitter/X: @realyunlong | Bluesky: not found | Web: https://yunlong-song.com/ | GitHub: yun-long

---

## Notes on Excluded Candidates

- **Dzmitry Tsetserukou** (Skoltech, ISR Lab PI): Technically the highest-signal connection for M5, but as an Associate Professor and lab head he is far less accessible than his students (Lykov, Serpiva). Reach out after establishing contact with lab members.
- **Ashish Kumar / Jitendra Malik** (RMA lineage, UC Berkeley): The original RMA framework (legged robots) inspired much of the fault-tolerant drone work, but these researchers are no longer focused on quadrotors specifically. The drone-specific implementations (MAVEN, Jiehao Chen) are better targets.
- **Hidenobu Matsuki** (MonoGS co-author): Now at Google (GenXR team) — industry position makes cold outreach harder and less likely to yield collaboration; Riku Murai at Imperial is the better target.
- **Davide Scaramuzza** (UZH RPG PI): Prolific and connected but approached through group members first.

---

_Sources and arXiv IDs verified via web search, May 2026. Twitter handles verified via direct search or personal website inspection. Email addresses taken from paper author lists or personal websites — do not paste in public-facing documents._
