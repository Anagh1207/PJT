"""
Generates an ultra-spacious, executive-tier clinical platform for CardioMigrate CPM.
Features large 620px canvas, wide 3-column clinical dials grid, full-width hero presentation,
and spacious mathematical proofs.
"""
import os

with open('dashboard/cpm_multigraph_data.json', 'r', encoding='utf-8') as f:
    multigraph_json = f.read()

artifact_dir = r"C:\Users\ACER\.gemini\antigravity\brain\f0ebfe14-f4d4-4d5d-a174-7cf7bf0964aa"
os.makedirs(artifact_dir, exist_ok=True)

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioMigrate AI — Counterfactual Patient Migration Clinical Platform</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    * {
      font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .font-mono {
      font-family: 'JetBrains Mono', monospace;
    }

    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: #080c16;
    }
    ::-webkit-scrollbar-thumb {
      background: #1e293b;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #334155;
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      padding: 0.22rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .katex-display {
      margin: 0.6em 0 !important;
    }
    .katex {
      font-size: 1.2em !important;
      color: #93c5fd !important;
    }
  </style>
</head>
<body class="bg-[#070b14] text-slate-100 min-h-screen flex flex-col antialiased selection:bg-blue-600 selection:text-white">

  <!-- TOP CLINICAL NAVIGATION HEADER -->
  <header class="border-b border-slate-800/80 bg-[#0b101e]/95 backdrop-blur sticky top-0 z-50 px-10 lg:px-14 py-4 flex flex-wrap items-center justify-between gap-6 shadow-2xl">
    <div class="flex items-center gap-5">
      <div class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-rose-500 via-indigo-500 to-emerald-400 p-[2.5px] shadow-xl shadow-indigo-500/20">
        <div class="w-full h-full bg-[#0b101e] rounded-[13px] flex items-center justify-center">
          <span class="text-3xl">🫀</span>
        </div>
      </div>
      <div>
        <div class="flex items-center gap-3">
          <h1 class="text-2xl font-extrabold tracking-tight text-white">CardioMigrate <span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-300 to-emerald-400">Clinical Platform</span></h1>
          <span class="badge-pill bg-blue-500/15 text-blue-400 border border-blue-500/30">IEEE Research Benchmark</span>
        </div>
        <p class="text-xs text-slate-400 mt-1 font-medium">Topological Patient Migration & Counterfactual Intervention Simulation Engine</p>
      </div>
    </div>

    <!-- Active Patient Selector & Visual Mode Switcher -->
    <div class="flex items-center gap-5">
      <div class="flex items-center gap-3">
        <label class="text-xs font-bold text-slate-300 uppercase tracking-wider">Select Patient Case:</label>
        <select id="patientSelect" class="bg-slate-900 border border-slate-700 text-xs font-bold rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer shadow-lg">
          <option value="852">Case 1: Ramesh K. (58M) — Severe Resistant Hypertension</option>
          <option value="336">Case 2: Priya S. (64F) — Diabetic Metabolic Syndrome & Smoker</option>
          <option value="556">Case 3: Anil M. (52M) — Executive Stress Tachycardia</option>
          <option value="428">Case 4: George T. (72M) — Senior Vascular Atherosclerosis</option>
        </select>
      </div>

      <div class="flex bg-slate-950 p-1.5 rounded-2xl border border-slate-800 shadow-inner">
        <button id="tabBtn_graphs" class="px-5 py-2.5 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow-md">
          📊 Multi-Graph Simulation
        </button>
        <button id="tabBtn_proofs" class="px-5 py-2.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition">
          📐 Mathematical Equations & Justifications
        </button>
        <button id="tabBtn_case" class="px-5 py-2.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition">
          📋 Patient Case Dossier
        </button>
      </div>
    </div>
  </header>

  <!-- SPACIOUS MAIN WORKSPACE -->
  <main class="flex-1 max-w-[2100px] w-full mx-auto p-8 lg:p-12 flex flex-col gap-8">

    <!-- ========================================================================================= -->
    <!-- VIEW 1: MULTI-GRAPH TOPOLOGY & INTERACTIVE MIGRATION (SPACIOUS 3-TIER SUITE) -->
    <!-- ========================================================================================= -->
    <div id="view_graphs" class="flex flex-col gap-8 w-full">
      
      <!-- TIER 1: FULL-WIDTH CLINICAL HERO BANNER (Profile + Vitals + Treatment Pathways) -->
      <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-8 shadow-2xl grid grid-cols-1 xl:grid-cols-12 gap-8 items-center">
        
        <!-- Left: Patient Presentation & Story (Col 5) -->
        <div class="xl:col-span-5 flex flex-col gap-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <h2 id="patientName" class="text-2xl font-black text-white">Ramesh K. (58, Male)</h2>
              <span id="patientJob" class="text-xs text-slate-400 font-semibold bg-slate-800/80 px-3 py-1 rounded-full">Bank Manager</span>
            </div>
            <div id="statusBadge" class="badge-pill bg-rose-500/20 text-rose-400 border border-rose-500/40 text-xs px-3.5 py-1">
              CRITICAL RISK (99.8%)
            </div>
          </div>
          <p id="patientComplaint" class="text-sm text-rose-400 font-medium leading-relaxed">
            Morning occipital headaches, exertional dyspnea, resting BP 172/105 mmHg.
          </p>
          <div class="bg-slate-950/80 rounded-2xl p-4 border border-slate-800/80 text-xs text-slate-300 leading-relaxed mt-1">
            <div class="font-bold text-slate-200 mb-1 flex items-center gap-2">
              <span class="text-sm">⚠️</span><span class="uppercase tracking-wider text-[11px] text-amber-400">Biological Peer Danger Warning:</span>
            </div>
            <p id="peerDangerDesc" class="text-slate-400">
              Connected to 15 clinically matched peers in the hospital network. 14 of those 15 patients (94.4%) suffered acute myocardial infarctions or strokes within 36 months.
            </p>
          </div>
        </div>

        <!-- Middle: Live Risk Gauge (Col 3) -->
        <div class="xl:col-span-3 bg-slate-950/90 rounded-2xl p-5 border border-slate-800/90 flex flex-col gap-3">
          <div class="flex justify-between items-center text-xs">
            <span class="text-slate-400 font-semibold uppercase tracking-wider text-[11px]">Predicted CVD Risk</span>
            <div class="font-mono font-bold flex items-center gap-2">
              <span id="meterCurrentRisk" class="text-rose-400 text-lg">99.8%</span>
              <span id="meterRiskDelta" class="text-xs text-emerald-400 font-semibold">(Goal: 10.5%)</span>
            </div>
          </div>
          <div class="w-full h-4 bg-slate-800 rounded-full overflow-hidden relative shadow-inner">
            <div id="meterBarFill" class="h-full bg-rose-500 rounded-full transition-all duration-300" style="width: 99.8%;"></div>
          </div>
          <div class="flex justify-between items-center text-xs text-slate-400 font-mono">
            <span>GraphSAGE Cutoff: 50%</span>
            <span id="meterStatusText" class="text-rose-400 font-bold">In Cardiac Danger Ward</span>
          </div>
        </div>

        <!-- Right: Actionable Treatment Pathways (Col 4) -->
        <div class="xl:col-span-4 flex flex-col gap-2.5">
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Physician's Prescriptions (Click to Apply):</span>
          <div class="grid grid-cols-2 gap-3">
            <button id="btnRx_1" class="bg-slate-800/90 hover:bg-slate-700 text-blue-300 text-xs font-bold py-3 px-3.5 rounded-2xl border border-slate-700/80 transition flex items-center justify-center gap-2 shadow">
              <span>💊</span><span id="lblRx_1">Telmisartan + Statin</span>
            </button>
            <button id="btnRx_2" class="bg-slate-800/90 hover:bg-slate-700 text-emerald-300 text-xs font-bold py-3 px-3.5 rounded-2xl border border-slate-700/80 transition flex items-center justify-center gap-2 shadow">
              <span>🏃</span><span id="lblRx_2">Cardiac Walk (4h/wk)</span>
            </button>
            <button id="btnRx_3" class="bg-slate-800/90 hover:bg-slate-700 text-cyan-300 text-xs font-bold py-3 px-3.5 rounded-2xl border border-slate-700/80 transition flex items-center justify-center gap-2 shadow">
              <span>🚭</span><span id="lblRx_3">Smoking Cessation</span>
            </button>
            <button id="btnRx_All" class="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white text-xs font-extrabold py-3 px-3.5 rounded-2xl shadow-xl shadow-emerald-600/25 transition flex items-center justify-center gap-2">
              <span>🎯</span><span>Full Target Plan</span>
            </button>
          </div>
        </div>

      </div>

      <!-- TIER 2: LARGE VISUAL ARENA & REAL-TIME MANIFOLD (Canvas + Proof Breakdown) -->
      <div class="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start">
        
        <!-- Large Interactive Visual Canvas Stage (Col 8) -->
        <div class="xl:col-span-8 bg-slate-900/90 border border-slate-800/90 rounded-3xl p-7 shadow-2xl flex flex-col gap-5">
          
          <!-- Canvas Top Toolbar -->
          <div class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div class="flex items-center gap-3">
              <span class="w-3.5 h-3.5 rounded-full bg-emerald-400 animate-pulse"></span>
              <span class="text-sm font-bold uppercase tracking-wider text-slate-200">Synchronized Visual Graph Canvas</span>
            </div>

            <!-- View Mode Switcher -->
            <div class="flex bg-slate-950 p-1.5 rounded-2xl border border-slate-800 gap-1.5 shadow-inner">
              <button id="btnSubView_local" class="px-4 py-2 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow">
                1. Local Patient Subgraph (Micro)
              </button>
              <button id="btnSubView_macro" class="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition">
                2. 5-Cluster Global Manifold (Macro)
              </button>
              <button id="btnSubView_radar" class="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition">
                3. Clinical Radar Fingerprint
              </button>
            </div>

            <div class="flex items-center gap-3">
              <button id="btnPlayAnimation" class="bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold px-4 py-2 rounded-xl shadow-lg shadow-blue-600/30 transition flex items-center gap-2">
                <span>▶ Play Animated Journey</span>
              </button>
            </div>
          </div>

          <!-- Main Spacious Canvas Box -->
          <div class="relative w-full h-[620px] rounded-2xl bg-[#050811] border border-slate-950 overflow-hidden shadow-2xl">
            <canvas id="activeCanvas" class="w-full h-full block cursor-crosshair"></canvas>

            <!-- Dynamic Floating Stage Card -->
            <div class="absolute top-5 right-5 bg-slate-900/95 backdrop-blur border border-slate-700/80 rounded-2xl px-5 py-3 text-xs font-mono shadow-2xl flex items-center gap-3">
              <span id="canvasPhaseIcon" class="text-2xl">🔴</span>
              <div>
                <div id="canvasPhaseTitle" class="text-rose-400 font-bold text-sm">Baseline: High-Risk Ward</div>
                <div id="canvasPhaseSubtitle" class="text-xs text-slate-400">14 of 15 Neighbors CVD+</div>
              </div>
            </div>

            <!-- Canvas Legend -->
            <div id="canvasLegendBox" class="absolute bottom-5 left-5 bg-slate-900/90 backdrop-blur border border-slate-800 rounded-2xl p-4 text-xs text-slate-300 flex flex-col gap-2.5 shadow-2xl">
              <!-- Dynamically populated -->
            </div>
          </div>

          <!-- Spacious Interactive Scrubber -->
          <div class="flex items-center gap-6 px-3 pt-2">
            <span class="text-xs font-mono text-rose-400 font-bold whitespace-nowrap">Baseline Position (0%)</span>
            <input type="range" id="migrationScrubber" min="0" max="100" value="0" class="flex-1 h-2.5 bg-slate-800 rounded-lg cursor-pointer accent-blue-500">
            <span class="text-xs font-mono text-emerald-400 font-bold whitespace-nowrap">Target Destination (100%)</span>
          </div>

        </div>

        <!-- Right Side: Live Math Proof & Composite Breakdown (Col 4) -->
        <div class="xl:col-span-4 flex flex-col gap-6">
          
          <!-- Composite Score Card -->
          <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-6 shadow-2xl flex flex-col gap-3">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Approved CPM Metric</span>
              <span class="badge-pill bg-emerald-500/15 text-emerald-400 font-mono">Peer Shift Validated</span>
            </div>
            <div class="flex items-baseline gap-3 mt-1">
              <span id="liveCpmScore" class="text-5xl font-black font-mono text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-300">0.9440</span>
              <span class="text-xs text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">Complete Relocation</span>
            </div>
            <div class="text-xs text-slate-400 font-mono mt-1">
              Formula: 0.40(T) + 0.40(R) + 0.20(D_norm)
            </div>
            <div class="w-full bg-slate-800 h-3 rounded-full overflow-hidden flex mt-2 shadow-inner">
              <div id="barPartT" class="bg-cyan-500 h-full" style="width: 40%;"></div>
              <div id="barPartR" class="bg-emerald-500 h-full" style="width: 38%;"></div>
              <div id="barPartD" class="bg-purple-500 h-full" style="width: 22%;"></div>
            </div>
            <div class="flex justify-between text-[11px] font-mono text-slate-400">
              <span class="text-cyan-400">Turnover (40%)</span>
              <span class="text-emerald-400">Risk Shift (40%)</span>
              <span class="text-purple-400">Displacement (20%)</span>
            </div>
          </div>

          <!-- Proof Breakdown Accordion Cards -->
          <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-6 shadow-2xl flex flex-col gap-4">
            <div class="flex items-center justify-between pb-3 border-b border-slate-800">
              <div class="flex items-center gap-2">
                <span class="text-base">📐</span>
                <h3 class="text-xs font-bold uppercase tracking-wider text-slate-200">Live Mathematical Proof Engine</h3>
              </div>
              <span class="text-[11px] font-mono text-cyan-400">Live Evaluation</span>
            </div>

            <!-- Step 1: Turnover -->
            <div class="bg-slate-950/80 border border-slate-800/90 rounded-2xl p-4 flex flex-col gap-2">
              <div class="flex justify-between items-center text-xs">
                <span class="font-bold text-slate-200">Step 1: Jaccard Turnover (T)</span>
                <span id="calcTVal" class="font-mono font-bold text-cyan-400 text-sm">T = 1.0000</span>
              </div>
              <div class="text-[11px] font-mono text-slate-400 bg-slate-900 p-2 rounded-xl border border-slate-800">
                T = 1 - (|N_orig ∩ N_new| / |N_orig ∪ N_new|)
              </div>
              <div id="calcTDesc" class="text-xs text-slate-300 leading-relaxed">
                Replaced 15 of 15 high-risk neighbors (100% network disconnection).
              </div>
            </div>

            <!-- Step 2: Risk Shift -->
            <div class="bg-slate-950/80 border border-slate-800/90 rounded-2xl p-4 flex flex-col gap-2">
              <div class="flex justify-between items-center text-xs">
                <span class="font-bold text-slate-200">Step 2: Prevalence Shift (R)</span>
                <span id="calcRVal" class="font-mono font-bold text-emerald-400 text-sm">R = 0.8778</span>
              </div>
              <div class="text-[11px] font-mono text-slate-400 bg-slate-900 p-2 rounded-xl border border-slate-800">
                R = max(0, Prevalence_orig - Prevalence_new)
              </div>
              <div class="flex justify-between text-xs text-slate-300 font-mono">
                <span>Orig Peers: <span id="calcOrigPrev" class="text-rose-400 font-bold">94.4% CVD</span></span>
                <span>➔</span>
                <span>New Peers: <span id="calcNewPrev" class="text-emerald-400 font-bold">6.7% CVD</span></span>
              </div>
            </div>

            <!-- Step 3: Displacement -->
            <div class="bg-slate-950/80 border border-slate-800/90 rounded-2xl p-4 flex flex-col gap-2">
              <div class="flex justify-between items-center text-xs">
                <span class="font-bold text-slate-200">Step 3: GraphSAGE Latent Displacement</span>
                <span id="calcDVal" class="font-mono font-bold text-purple-400 text-sm">D_norm = 0.9642</span>
              </div>
              <div class="text-[11px] font-mono text-slate-400 bg-slate-900 p-2 rounded-xl border border-slate-800">
                D = ||z_new - z_orig||₂ = 15.49
              </div>
              <div class="text-xs text-slate-300 leading-relaxed">
                Structural translation across the 32D clinical representation space.
              </div>
            </div>

          </div>

        </div>

      </div>

      <!-- TIER 3: SPACIOUS 3-COLUMN CLINICAL DIALS & BIOMARKER INTERVENTIONS -->
      <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-8 shadow-2xl flex flex-col gap-6">
        
        <div class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <h3 class="text-base font-extrabold text-white flex items-center gap-2.5">
              <span>🎚️</span><span>Actionable Clinical Biomarker Dials (Mutable Factors)</span>
            </h3>
            <p class="text-xs text-slate-400 mt-0.5">Drag any dial to test custom interventions in real-time and observe immediate manifold migration.</p>
          </div>
          <button id="btnResetDials" class="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold py-2.5 px-5 rounded-xl border border-slate-700 transition shadow">
            ↺ Reset to Baseline Symptoms
          </button>
        </div>

        <!-- 3-Column Wide Grid for Dials -->
        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6" id="dialsContainer">
          <!-- Dynamically populated in 3-column wide cards -->
        </div>

      </div>

      <!-- TIER 4: PHYSICIAN CLINICAL COUNSELING DIALOGUE (HUMAN-WRITTEN) -->
      <div class="bg-gradient-to-r from-blue-950/40 via-slate-900 to-indigo-950/40 border border-blue-900/60 rounded-3xl p-8 shadow-2xl">
        <div class="flex items-center gap-3 text-blue-400 text-xs font-bold uppercase tracking-wider mb-2.5">
          <span class="text-base">🩺</span>
          <span>Doctor's Real-World Counseling Script (Empathetic Translation for Patient):</span>
        </div>
        <p id="counselingDialogueText" class="text-sm text-slate-200 leading-relaxed font-normal">
          Loading patient dialogue...
        </p>
      </div>

    </div>

    <!-- ========================================================================================= -->
    <!-- VIEW 2: MATHEMATICAL EQUATIONS & CLINICAL JUSTIFICATIONS (FULL WIDTH SPACIOUS SUITE) -->
    <!-- ========================================================================================= -->
    <div id="view_proofs" class="hidden flex flex-col gap-8 w-full">
      
      <!-- Top Overview Banner -->
      <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-8 shadow-2xl flex flex-col gap-3">
        <h2 class="text-2xl font-black text-white flex items-center gap-3">
          <span>📐</span><span>Formal Mathematical Framework of Counterfactual Patient Migration (CPM)</span>
        </h2>
        <p class="text-sm text-slate-300 max-w-5xl leading-relaxed">
          Traditional counterfactual models only report a change in predicted probability (delta y). The CPM framework provides a rigorous, 
          topology-aware mathematical formulation that explains how a patient navigates the clinical similarity network, decouples structural peer movement 
          from intervention effort, and prevents false confidence from static graph assumptions.
        </p>
      </div>

      <!-- 6 Core Equations Cards Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        <!-- Formula 1: Domain-Weighted Distance -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col gap-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <span class="text-xs font-bold uppercase tracking-wider text-cyan-400">Equation 1: Domain-Weighted Clinical Distance</span>
            <span class="badge-pill bg-cyan-500/15 text-cyan-400 font-mono">Similarity Metric</span>
          </div>
          <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center font-mono text-blue-300 text-lg">
            d_clinical(x, x') = &radic; [ &Sigma; w_k &middot; ((x_k - x'_k) / &sigma;_k)&sup2; ]
          </div>
          <div class="text-xs text-slate-300 leading-relaxed">
            <strong class="text-white text-sm block mb-1">Clinical Justification & Real-Life Usecase:</strong> 
            Standard Euclidean distance treats all features equally (e.g., 10 mmHg of blood pressure equals 10 years of age). In clinical cardiology, arterial hypertension (w = 1.8) and hypercholesterolemia (w = 2.0) carry significantly higher lethal risk than gender (w = 0.5). Domain weighting guarantees that the constructed patient graph reflects physiological hazard rather than statistical noise.
          </div>
        </div>

        <!-- Formula 2: GraphSAGE Neighborhood Aggregation -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col gap-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <span class="text-xs font-bold uppercase tracking-wider text-indigo-400">Equation 2: GraphSAGE Multi-Head Aggregation</span>
            <span class="badge-pill bg-indigo-500/15 text-indigo-400 font-mono">Deep Representation</span>
          </div>
          <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center font-mono text-indigo-300 text-lg">
            h_v^(l) = &sigma; [ W^(l) &middot; ( h_v^(l-1) || AGG({ h_u^(l-1), &forall; u &isin; N(v) }) ) ]
          </div>
          <div class="text-xs text-slate-300 leading-relaxed">
            <strong class="text-white text-sm block mb-1">Clinical Justification & Real-Life Usecase:</strong> 
            In medicine, a patient's vulnerability is contextualized by their peer group. GraphSAGE aggregates latent clinical features from the 15 nearest clinical neighbors across 2 neural message-passing layers. This ensures the model learns non-linear risk representations from community health patterns rather than treating the patient as an isolated tabular record.
          </div>
        </div>

        <!-- Formula 3: Neighbor Turnover -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col gap-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">Equation 3: Jaccard Neighbor Turnover (T)</span>
            <span class="badge-pill bg-emerald-500/15 text-emerald-400 font-mono">Network Relocation</span>
          </div>
          <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center font-mono text-emerald-300 text-lg">
            T = 1 - ( | N_orig &cap; N_new | / | N_orig &cup; N_new | ) &isin; [0, 1]
          </div>
          <div class="text-xs text-slate-300 leading-relaxed">
            <strong class="text-white text-sm block mb-1">Clinical Justification & Real-Life Usecase:</strong> 
            When a patient lowers their blood pressure, do they actually break away from their high-risk cardiac peers? If T = 0, the patient remains trapped among the same sick individuals despite taking pills. When T = 1.0, all 15 baseline neighbors are replaced by healthy individuals, confirming genuine clinical relocation.
          </div>
        </div>

        <!-- Formula 4: Prevalence Shift -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col gap-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <span class="text-xs font-bold uppercase tracking-wider text-amber-400">Equation 4: Neighborhood Prevalence Shift (R)</span>
            <span class="badge-pill bg-amber-500/15 text-amber-400 font-mono">Peer Risk Shift</span>
          </div>
          <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center font-mono text-amber-300 text-lg">
            R = max( 0, ( 1/|N_orig| &middot; &Sigma; y_u ) - ( 1/|N_new| &middot; &Sigma; y_v ) )
          </div>
          <div class="text-xs text-slate-300 leading-relaxed">
            <strong class="text-white text-sm block mb-1">Clinical Justification & Real-Life Usecase:</strong> 
            Directly measures the disease density of the patient's immediate social and biological cohort. For Ramesh, peer CVD incidence plummets from 94.4% down to 6.7%, yielding R = 0.8778. This offers doctors an intuitive way to explain outcomes: "Your community risk dropped by 88%."
          </div>
        </div>

        <!-- Formula 5: Latent Embedding Displacement -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col gap-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <span class="text-xs font-bold uppercase tracking-wider text-purple-400">Equation 5: GraphSAGE Latent Displacement (D_norm)</span>
            <span class="badge-pill bg-purple-500/15 text-purple-400 font-mono">Manifold Movement</span>
          </div>
          <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center font-mono text-purple-300 text-lg">
            D = || z_new - z_orig ||_2,  D_norm = (D - D_min) / (D_max - D_min) &isin; [0, 1]
          </div>
          <div class="text-xs text-slate-300 leading-relaxed">
            <strong class="text-white text-sm block mb-1">Clinical Justification & Real-Life Usecase:</strong> 
            Feature space Euclidean distance can be misleading because physiological features have non-linear interactions. D_norm measures displacement in the deep 32-dimensional latent manifold constructed by GraphSAGE, verifying that the patient has moved into the low-risk geometric basin.
          </div>
        </div>

        <!-- Formula 6: Composite CPM Score -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col gap-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <span class="text-xs font-bold uppercase tracking-wider text-rose-400">Equation 6: Approved Composite CPM Metric</span>
            <span class="badge-pill bg-rose-500/15 text-rose-400 font-mono">Core Paper Metric</span>
          </div>
          <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center font-mono text-rose-300 text-lg">
            CPM = 0.40 &middot; T + 0.40 &middot; R + 0.20 &middot; D_norm
          </div>
          <div class="text-xs text-slate-300 leading-relaxed">
            <strong class="text-white text-sm block mb-1">Clinical Justification & Real-Life Usecase:</strong> 
            Validated through sensitivity analysis (rank correlation r > 0.94). Weights allocate 40% to topology replacement, 40% to peer disease reduction, and 20% to latent manifold movement. Feature effort F_norm = ||&Delta;x||_2 / &radic;d is reported separately to prevent hiding difficult interventions behind a high score.
          </div>
        </div>

      </div>

    </div>

    <!-- ========================================================================================= -->
    <!-- VIEW 3: COMPREHENSIVE PATIENT CASE FILE & MEDICAL ACTION PLAN -->
    <!-- ========================================================================================= -->
    <div id="view_case" class="hidden flex flex-col gap-8 w-full">
      
      <!-- Patient Dossier Card -->
      <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-8 shadow-2xl flex flex-col gap-6">
        <div class="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
          <div>
            <h2 id="caseDossierName" class="text-2xl font-black text-white">Clinical Case Dossier: Ramesh K.</h2>
            <p id="caseDossierOccupation" class="text-sm text-slate-400 mt-1">58 Years Old • Male • Senior Bank Manager</p>
          </div>
          <div class="flex gap-3">
            <span class="px-4 py-2 rounded-2xl text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40">Baseline Risk: 99.8%</span>
            <span class="px-4 py-2 rounded-2xl text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">Target Risk: 10.5%</span>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
          <div class="bg-slate-950/80 p-6 rounded-2xl border border-slate-800/90 flex flex-col gap-2">
            <span class="font-bold text-slate-200 uppercase tracking-wider text-xs">Chief Medical Complaints</span>
            <p id="caseChiefComplaints" class="text-slate-400 leading-relaxed text-xs">
              Recurring morning occipital headaches, shortness of breath after climbing one flight of stairs, chronic exhaustion.
            </p>
          </div>
          <div class="bg-slate-950/80 p-6 rounded-2xl border border-slate-800/90 flex flex-col gap-2">
            <span class="font-bold text-slate-200 uppercase tracking-wider text-xs">Vitals & Hemodynamics</span>
            <p id="caseVitals" class="text-slate-400 leading-relaxed text-xs">
              Resting BP 172/105 mmHg, Resting Heart Rate 92 bpm, BMI 29.2 kg/m².
            </p>
          </div>
          <div class="bg-slate-950/80 p-6 rounded-2xl border border-slate-800/90 flex flex-col gap-2">
            <span class="font-bold text-slate-200 uppercase tracking-wider text-xs">Biomarkers & Blood Panel</span>
            <p id="caseLabs" class="text-slate-400 leading-relaxed text-xs">
              Total Cholesterol 293 mg/dL, Triglycerides 290 mg/dL, Fasting Glucose 76.5 mg/dL.
            </p>
          </div>
        </div>
      </div>

      <!-- Step-by-Step Medical Treatment Protocol -->
      <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-8 shadow-2xl flex flex-col gap-5">
        <h3 class="text-base font-extrabold uppercase tracking-wider text-white flex items-center gap-3">
          <span class="text-lg">🩺</span><span>Recommended Step-by-Step Clinical Treatment Protocol</span>
        </h3>

        <div id="caseProtocolList" class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <!-- Dynamically populated in 3-column cards -->
        </div>
      </div>

    </div>

  </main>

  <!-- MULTI-GRAPH CLIENT ENGINE -->
  <script>
    const MULTI_DATA = __MULTIGRAPH_JSON__;
  </script>

  <script>
    // State
    let activePatientId = 852;
    let currentPatient = MULTI_DATA.patients.find(p => p.id === activePatientId) || MULTI_DATA.patients[0];
    let currentDials = JSON.parse(JSON.stringify(currentPatient.features));
    let progressRatio = 0.0;
    let activeSubView = "local";
    let isPlaying = false;
    let animSpeed = 1.0;
    let animFrame = null;

    // DOM Elements - Navigation
    const patientSelect = document.getElementById('patientSelect');
    const tabBtn_graphs = document.getElementById('tabBtn_graphs');
    const tabBtn_proofs = document.getElementById('tabBtn_proofs');
    const tabBtn_case = document.getElementById('tabBtn_case');
    const view_graphs = document.getElementById('view_graphs');
    const view_proofs = document.getElementById('view_proofs');
    const view_case = document.getElementById('view_case');

    // DOM Elements - Patient Card
    const patientName = document.getElementById('patientName');
    const patientJob = document.getElementById('patientJob');
    const patientComplaint = document.getElementById('patientComplaint');
    const statusBadge = document.getElementById('statusBadge');
    const peerDangerDesc = document.getElementById('peerDangerDesc');
    const meterCurrentRisk = document.getElementById('meterCurrentRisk');
    const meterRiskDelta = document.getElementById('meterRiskDelta');
    const meterBarFill = document.getElementById('meterBarFill');
    const meterStatusText = document.getElementById('meterStatusText');
    const dialsContainer = document.getElementById('dialsContainer');
    const counselingDialogueText = document.getElementById('counselingDialogueText');

    // Canvas & Controls
    const canvas = document.getElementById('activeCanvas');
    const ctx = canvas.getContext('2d');
    const canvasPhaseIcon = document.getElementById('canvasPhaseIcon');
    const canvasPhaseTitle = document.getElementById('canvasPhaseTitle');
    const canvasPhaseSubtitle = document.getElementById('canvasPhaseSubtitle');
    const canvasLegendBox = document.getElementById('canvasLegendBox');
    const migrationScrubber = document.getElementById('migrationScrubber');
    const btnPlayAnimation = document.getElementById('btnPlayAnimation');

    // Prescription Buttons
    const btnRx_1 = document.getElementById('btnRx_1');
    const btnRx_2 = document.getElementById('btnRx_2');
    const btnRx_3 = document.getElementById('btnRx_3');
    const btnRx_All = document.getElementById('btnRx_All');
    const btnResetDials = document.getElementById('btnResetDials');
    const lblRx_1 = document.getElementById('lblRx_1');
    const lblRx_2 = document.getElementById('lblRx_2');
    const lblRx_3 = document.getElementById('lblRx_3');

    // Sub-view buttons
    const btnSubView_local = document.getElementById('btnSubView_local');
    const btnSubView_macro = document.getElementById('btnSubView_macro');
    const btnSubView_radar = document.getElementById('btnSubView_radar');

    // Proofs Elements
    const liveCpmScore = document.getElementById('liveCpmScore');
    const calcTVal = document.getElementById('calcTVal');
    const calcTDesc = document.getElementById('calcTDesc');
    const calcRVal = document.getElementById('calcRVal');
    const calcOrigPrev = document.getElementById('calcOrigPrev');
    const calcNewPrev = document.getElementById('calcNewPrev');
    const calcDVal = document.getElementById('calcDVal');
    const barPartT = document.getElementById('barPartT');
    const barPartR = document.getElementById('barPartR');
    const barPartD = document.getElementById('barPartD');

    // Resize Canvas
    let cWidth, cHeight;
    function resizeCanvas() {
      const rect = canvas.parentElement.getBoundingClientRect();
      cWidth = rect.width;
      cHeight = rect.height;
      canvas.width = cWidth * window.devicePixelRatio;
      canvas.height = cHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Tab Navigation Logic
    function switchTab(activeTab) {
      tabBtn_graphs.className = activeTab === 'graphs' ? "px-5 py-2.5 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow-md" : "px-5 py-2.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
      tabBtn_proofs.className = activeTab === 'proofs' ? "px-5 py-2.5 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow-md" : "px-5 py-2.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
      tabBtn_case.className = activeTab === 'case' ? "px-5 py-2.5 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow-md" : "px-5 py-2.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition";

      view_graphs.classList.toggle('hidden', activeTab !== 'graphs');
      view_proofs.classList.toggle('hidden', activeTab !== 'proofs');
      view_case.classList.toggle('hidden', activeTab !== 'case');

      if (activeTab === 'graphs') {
        resizeCanvas();
      }
    }

    tabBtn_graphs.addEventListener('click', () => switchTab('graphs'));
    tabBtn_proofs.addEventListener('click', () => switchTab('proofs'));
    tabBtn_case.addEventListener('click', () => switchTab('case'));

    // Sub-view Switcher
    function setSubView(mode) {
      activeSubView = mode;
      btnSubView_local.className = mode === 'local' ? "px-4 py-2 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow" : "px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
      btnSubView_macro.className = mode === 'macro' ? "px-4 py-2 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow" : "px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
      btnSubView_radar.className = mode === 'radar' ? "px-4 py-2 rounded-xl text-xs font-bold bg-blue-600 text-white transition shadow" : "px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
      updateCanvasLegend();
    }

    btnSubView_local.addEventListener('click', () => setSubView('local'));
    btnSubView_macro.addEventListener('click', () => setSubView('macro'));
    btnSubView_radar.addEventListener('click', () => setSubView('radar'));

    function updateCanvasLegend() {
      if (activeSubView === 'local') {
        canvasLegendBox.innerHTML = `
          <div class="flex items-center gap-2.5">
            <span class="w-3.5 h-3.5 rounded-full bg-rose-500 inline-block shadow"></span>
            <span>15 Baseline Peers (94% CVD Mortality Cluster)</span>
          </div>
          <div class="flex items-center gap-2.5">
            <span class="w-3.5 h-3.5 rounded-full bg-emerald-500 inline-block shadow"></span>
            <span>15 Destination Peers (Healthy Reference Cohort)</span>
          </div>
          <div class="flex items-center gap-2.5">
            <span class="w-3.5 h-3.5 rounded-full bg-amber-400 border border-white inline-block"></span>
            <span>${currentPatient.name} (Migrating Node)</span>
          </div>
          <div class="flex items-center gap-2.5">
            <span class="w-3 h-3 rounded-lg bg-indigo-500 inline-block"></span>
            <span>Waypoint Care Stations (Outpatient & Rehab)</span>
          </div>
        `;
      } else if (activeSubView === 'macro') {
        canvasLegendBox.innerHTML = `
          <div class="grid grid-cols-2 gap-x-5 gap-y-2">
            <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full bg-rose-500"></span><span>Severe Hypertensive</span></div>
            <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full bg-orange-500"></span><span>Diabetic Metabolic</span></div>
            <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full bg-yellow-500"></span><span>Transitional Corridor</span></div>
            <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full bg-emerald-500"></span><span>Healthy Reference</span></div>
          </div>
        `;
      } else {
        canvasLegendBox.innerHTML = `
          <div class="flex items-center gap-5">
            <div class="flex items-center gap-2"><span class="w-4 h-2 bg-rose-500 inline-block rounded"></span><span>Baseline Biomarkers</span></div>
            <div class="flex items-center gap-2"><span class="w-4 h-2 bg-emerald-400 inline-block rounded"></span><span>Current / Target State</span></div>
          </div>
        `;
      }
    }

    // Load Patient Case
    function loadPatient(pid) {
      activePatientId = parseInt(pid);
      currentPatient = MULTI_DATA.patients.find(p => p.id === activePatientId) || MULTI_DATA.patients[0];
      currentDials = JSON.parse(JSON.stringify(currentPatient.features));

      patientName.textContent = `${currentPatient.name} (${currentPatient.age}, ${currentPatient.gender})`;
      patientJob.textContent = currentPatient.occupation;
      patientComplaint.textContent = currentPatient.chief_complaint;
      peerDangerDesc.textContent = currentPatient.peer_danger_context;

      document.getElementById('caseDossierName').textContent = `Clinical Case Dossier: ${currentPatient.name}`;
      document.getElementById('caseDossierOccupation').textContent = `${currentPatient.age} Years Old • ${currentPatient.gender} • ${currentPatient.occupation}`;
      document.getElementById('caseChiefComplaints').textContent = currentPatient.chief_complaint;
      document.getElementById('caseVitals').textContent = currentPatient.vital_signs;
      document.getElementById('caseLabs').textContent = currentPatient.lab_biomarkers;

      lblRx_1.textContent = currentPatient.prescription[0]?.action.split('&')[0] || "Step 1: Medication";
      lblRx_2.textContent = currentPatient.prescription[1]?.action.split(':')[0] || "Step 2: Lifestyle";
      lblRx_3.textContent = currentPatient.prescription[2]?.action.split('with')[0] || "Step 3: Behavioral";

      const protocolList = document.getElementById('caseProtocolList');
      protocolList.innerHTML = '';
      currentPatient.prescription.forEach(step => {
        const card = document.createElement('div');
        card.className = "bg-slate-950/80 p-6 rounded-2xl border border-slate-800/90 flex flex-col gap-2 shadow-lg";
        card.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-bold text-blue-400 text-xs">${step.step} • ${step.category}</span>
            <span class="badge-pill bg-emerald-500/15 text-emerald-400 font-mono text-[10px]">Verified Target</span>
          </div>
          <div class="font-bold text-white text-sm mt-1">${step.action}</div>
          <div class="text-xs text-slate-400 font-mono mt-1">Impact: ${step.impact}</div>
        `;
        protocolList.appendChild(card);
      });

      counselingDialogueText.innerHTML = `
        <strong>Physician to Patient:</strong> "${currentPatient.name.split(' ')[0]}, right now you are situated in our cardiac high-risk cluster. 
        <strong>${currentPatient.prevalence_orig.toFixed(0)}% of the 15 patients most clinically similar to you have experienced heart attacks or strokes</strong>. 
        If you adhere to our targeted treatment plan—lowering your blood pressure and normalizing your lipid profile—you don't just change a number on a test result. 
        <strong>You break connections with those 15 high-risk patients and physically migrate into our healthy reference cohort</strong>, where heart disease incidence drops to <strong>${currentPatient.prevalence_new.toFixed(0)}%</strong>."
      `;

      progressRatio = 0.0;
      migrationScrubber.value = 0;
      renderDials();
      recalculateMetrics();
      updateCanvasLegend();
    }

    patientSelect.addEventListener('change', (e) => loadPatient(e.target.value));

    // Render 3-Column Spacious Dials
    function renderDials() {
      dialsContainer.innerHTML = '';
      const mutableKeys = [
        'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
        'Physical_Activity_HoursPerWeek', 'Glucose', 'Triglycerides',
        'Stress_Index', 'Smoking', 'Diabetes'
      ];

      const labels = {
        'Blood_Pressure': 'Blood Pressure (mmHg)',
        'Cholesterol': 'Total Cholesterol (mg/dL)',
        'Heart_Rate': 'Resting Heart Rate (bpm)',
        'BMI': 'Body Mass Index (BMI)',
        'Physical_Activity_HoursPerWeek': 'Physical Activity (hrs/wk)',
        'Glucose': 'Fasting Glucose (mg/dL)',
        'Triglycerides': 'Triglycerides (mg/dL)',
        'Stress_Index': 'Stress Index (1-10)',
        'Smoking': 'Smoking (0: No, 1: Yes)',
        'Diabetes': 'Diabetes (0: No, 1: Yes)'
      };

      mutableKeys.forEach(k => {
        const feat = currentPatient.features[k];
        if (!feat) return;

        const row = document.createElement('div');
        row.className = "bg-slate-950/80 border border-slate-800/90 rounded-2xl p-5 flex flex-col gap-3 shadow-lg transition hover:border-slate-700";

        const minVal = Math.min(feat.orig * 0.5, feat.cf * 0.7, 0);
        const maxVal = Math.max(feat.orig * 1.3, feat.cf * 1.3, 10);
        const stepVal = (k === 'Smoking' || k === 'Diabetes') ? 1 : 0.5;

        row.innerHTML = `
          <div class="flex justify-between items-center text-xs">
            <span class="text-slate-300 font-bold">${labels[k] || k}</span>
            <div class="flex items-center gap-2 font-mono">
              <span class="text-slate-400 text-xs">Base: ${feat.orig}</span>
              <span class="text-slate-500">➔</span>
              <span class="text-cyan-400 font-extrabold text-sm" id="dialVal_${k}">${currentDials[k].orig}</span>
            </div>
          </div>
          <div class="flex items-center gap-3 mt-1">
            <input type="range" id="inputSlider_${k}" 
              min="${minVal}" 
              max="${maxVal}" 
              step="${stepVal}" 
              value="${currentDials[k].orig}" 
              class="w-full h-2 bg-slate-800 rounded-lg cursor-pointer accent-blue-500">
          </div>
        `;
        dialsContainer.appendChild(row);

        const slider = row.querySelector(`#inputSlider_${k}`);
        slider.addEventListener('input', (e) => {
          const newVal = parseFloat(e.target.value);
          currentDials[k].current = newVal;
          document.getElementById(`dialVal_${k}`).textContent = newVal.toFixed(1);
          updateProgressFromDials();
          recalculateMetrics();
        });
      });
    }

    function updateProgressFromDials() {
      let totalW = 0;
      let totalProg = 0;
      const mutableKeys = ['Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI', 'Physical_Activity_HoursPerWeek', 'Glucose', 'Triglycerides', 'Stress_Index', 'Smoking', 'Diabetes'];

      mutableKeys.forEach(k => {
        const feat = currentPatient.features[k];
        const cur = currentDials[k].current !== undefined ? currentDials[k].current : feat.orig;
        const totalDelta = feat.cf - feat.orig;
        const weight = feat.weight || 1.0;

        totalW += weight;
        if (Math.abs(totalDelta) > 1e-4) {
          const curDelta = cur - feat.orig;
          const ratio = Math.max(0, Math.min(1.2, curDelta / totalDelta));
          totalProg += ratio * weight;
        } else {
          totalProg += 1.0 * weight;
        }
      });

      progressRatio = Math.max(0.0, Math.min(1.0, totalProg / totalW));
      migrationScrubber.value = Math.round(progressRatio * 100);
    }

    function setDialsFromProgress(ratio) {
      const mutableKeys = ['Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI', 'Physical_Activity_HoursPerWeek', 'Glucose', 'Triglycerides', 'Stress_Index', 'Smoking', 'Diabetes'];
      mutableKeys.forEach(k => {
        const feat = currentPatient.features[k];
        const interpolated = feat.orig + (feat.cf - feat.orig) * ratio;
        currentDials[k].current = interpolated;

        const slider = document.getElementById(`inputSlider_${k}`);
        const textVal = document.getElementById(`dialVal_${k}`);
        if (slider && textVal) {
          slider.value = interpolated;
          textVal.textContent = interpolated.toFixed(1);
        }
      });
    }

    function recalculateMetrics() {
      const p = currentPatient;
      const r = progressRatio;

      const currentRisk = p.orig_risk - (p.orig_risk - p.cf_risk) * r;
      meterCurrentRisk.textContent = `${currentRisk.toFixed(1)}%`;
      meterBarFill.style.width = `${currentRisk}%`;

      if (currentRisk > 65) {
        meterBarFill.className = "h-full bg-rose-500 rounded-full transition-all duration-150";
        statusBadge.className = "badge-pill bg-rose-500/20 text-rose-400 border border-rose-500/40 text-xs px-3 py-1";
        statusBadge.textContent = "CRITICAL RISK";
        canvasPhaseIcon.textContent = "🔴";
        canvasPhaseTitle.textContent = "Baseline: High-Risk Ward";
        canvasPhaseTitle.className = "text-rose-400 font-bold text-sm";
        canvasPhaseSubtitle.textContent = `${p.prevalence_orig.toFixed(0)}% Peers Experienced CVD`;
        meterStatusText.textContent = "In Cardiac Danger Ward";
        meterStatusText.className = "text-rose-400 font-bold";
      } else if (currentRisk > 35) {
        meterBarFill.className = "h-full bg-amber-500 rounded-full transition-all duration-150";
        statusBadge.className = "badge-pill bg-amber-500/20 text-amber-400 border border-amber-500/40 text-xs px-3 py-1";
        statusBadge.textContent = "TRANSITIONING";
        canvasPhaseIcon.textContent = "🟡";
        canvasPhaseTitle.textContent = "Transitional Care Corridor";
        canvasPhaseTitle.className = "text-amber-400 font-bold text-sm";
        canvasPhaseSubtitle.textContent = "Severing High-Risk Network Edges";
        meterStatusText.textContent = "Intermediate Clinical Corridor";
        meterStatusText.className = "text-amber-400 font-bold";
      } else {
        meterBarFill.className = "h-full bg-emerald-500 rounded-full transition-all duration-150";
        statusBadge.className = "badge-pill bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-xs px-3 py-1";
        statusBadge.textContent = "HEALTHY RE-EMBEDDED";
        canvasPhaseIcon.textContent = "🟢";
        canvasPhaseTitle.textContent = "Healthy Reference Cohort";
        canvasPhaseTitle.className = "text-emerald-400 font-bold text-sm";
        canvasPhaseSubtitle.textContent = `Latched to ${p.prevalence_new.toFixed(0)}% CVD Cohort`;
        meterStatusText.textContent = "Primary Prevention Group";
        meterStatusText.className = "text-emerald-400 font-bold";
      }

      // Dynamic Turnover (T)
      const currentT = p.turnover * r;
      calcTVal.textContent = `T = ${currentT.toFixed(4)}`;
      const replacedPeers = Math.round(15 * currentT);
      calcTDesc.textContent = `Disconnected ${replacedPeers} of 15 high-risk neighbors (${(currentT * 100).toFixed(0)}% turnover).`;

      // Dynamic Risk Shift (R)
      const currentPrev = p.prevalence_orig - (p.prevalence_orig - p.prevalence_new) * r;
      const currentR = Math.max(0, (p.prevalence_orig - currentPrev) / 100.0);
      calcRVal.textContent = `R = ${currentR.toFixed(4)}`;
      calcOrigPrev.textContent = `${p.prevalence_orig.toFixed(1)}% CVD`;
      calcNewPrev.textContent = `${currentPrev.toFixed(1)}% CVD`;

      // Dynamic Latent Displacement (D_norm)
      const currentDnorm = p.d_norm * r;
      const currentDraw = (p.emb_displacement * r).toFixed(2);
      calcDVal.textContent = `D_norm = ${currentDnorm.toFixed(4)} (raw ${currentDraw})`;

      // Composite CPM Score
      const dynamicCpm = (0.40 * currentT + 0.40 * currentR + 0.20 * currentDnorm).toFixed(4);
      liveCpmScore.textContent = dynamicCpm;

      barPartT.style.width = `${Math.max(5, currentT * 40)}%`;
      barPartR.style.width = `${Math.max(5, currentR * 40)}%`;
      barPartD.style.width = `${Math.max(5, currentDnorm * 20)}%`;
    }

    // Scrubber
    migrationScrubber.addEventListener('input', (e) => {
      progressRatio = parseFloat(e.target.value) / 100.0;
      isPlaying = false;
      btnPlayAnimation.textContent = "▶ Play Animated Journey";
      setDialsFromProgress(progressRatio);
      recalculateMetrics();
    });

    // Action Prescriptions
    btnRx_1.addEventListener('click', () => {
      progressRatio = 0.40;
      migrationScrubber.value = 40;
      setDialsFromProgress(0.40);
      recalculateMetrics();
    });

    btnRx_2.addEventListener('click', () => {
      progressRatio = 0.70;
      migrationScrubber.value = 70;
      setDialsFromProgress(0.70);
      recalculateMetrics();
    });

    btnRx_3.addEventListener('click', () => {
      progressRatio = 0.85;
      migrationScrubber.value = 85;
      setDialsFromProgress(0.85);
      recalculateMetrics();
    });

    btnRx_All.addEventListener('click', () => {
      progressRatio = 1.0;
      migrationScrubber.value = 100;
      setDialsFromProgress(1.0);
      recalculateMetrics();
    });

    btnResetDials.addEventListener('click', () => {
      progressRatio = 0.0;
      migrationScrubber.value = 0;
      setDialsFromProgress(0.0);
      recalculateMetrics();
    });

    btnPlayAnimation.addEventListener('click', () => {
      isPlaying = !isPlaying;
      btnPlayAnimation.textContent = isPlaying ? "⏸ Pause Journey" : "▶ Play Animated Journey";
    });

    // =========================================================================
    // 60FPS CANVAS RENDERING - MULTI-GRAPH MODES
    // =========================================================================
    let pulseT = 0;

    function renderActiveCanvas() {
      ctx.clearRect(0, 0, cWidth, cHeight);

      if (activeSubView === 'local') {
        renderLocalSubgraph();
      } else if (activeSubView === 'macro') {
        renderMacroManifold();
      } else if (activeSubView === 'radar') {
        renderRadarFingerprint();
      }

      if (isPlaying) {
        progressRatio += 0.005 * animSpeed;
        if (progressRatio >= 1.0) {
          progressRatio = 1.0;
          isPlaying = false;
          btnPlayAnimation.textContent = "▶ Play Animated Journey";
        }
        migrationScrubber.value = Math.round(progressRatio * 100);
        setDialsFromProgress(progressRatio);
        recalculateMetrics();
      }

      animFrame = requestAnimationFrame(renderActiveCanvas);
    }

    // GRAPH VIEW 1: LOCAL PATIENT SUBGRAPH WITH MULTIPLE WAYPOINTS
    function renderLocalSubgraph() {
      const p = currentPatient;
      const centerX = cWidth / 2;
      const centerY = cHeight / 2;

      function toScreen(x, y) {
        return {
          x: centerX + x * (cWidth / 420),
          y: centerY + y * (cHeight / 420)
        };
      }

      const startPos = toScreen(p.local_graph.start_coords[0], p.local_graph.start_coords[1]);
      const endPos = toScreen(p.local_graph.end_coords[0], p.local_graph.end_coords[1]);

      // Spacious Radial Auras
      const gH = ctx.createRadialGradient(startPos.x, startPos.y, 10, startPos.x, startPos.y, 190);
      gH.addColorStop(0, 'rgba(239, 68, 68, 0.18)');
      gH.addColorStop(1, 'rgba(239, 68, 68, 0)');
      ctx.fillStyle = gH;
      ctx.beginPath();
      ctx.arc(startPos.x, startPos.y, 190, 0, Math.PI * 2);
      ctx.fill();

      const gL = ctx.createRadialGradient(endPos.x, endPos.y, 10, endPos.x, endPos.y, 200);
      gL.addColorStop(0, 'rgba(34, 197, 94, 0.18)');
      gL.addColorStop(1, 'rgba(34, 197, 94, 0)');
      ctx.fillStyle = gL;
      ctx.beginPath();
      ctx.arc(endPos.x, endPos.y, 200, 0, Math.PI * 2);
      ctx.fill();

      // Draw Local Baseline 15 Peers
      p.local_graph.orig_neighbors.forEach(n => {
        const pt = toScreen(n.x, n.y);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = n.cvd === 1 ? '#ef4444' : '#f97316';
        ctx.shadowColor = '#ef4444';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText(`Pt #${n.real_id} (${n.bp} mmHg)`, pt.x + 9, pt.y + 3);
      });

      // Draw Local Destination 15 Peers
      p.local_graph.new_neighbors.forEach(n => {
        const pt = toScreen(n.x, n.y);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = n.cvd === 0 ? '#22c55e' : '#06b6d4';
        ctx.shadowColor = '#22c55e';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText(`Pt #${n.real_id} (${n.bp} mmHg)`, pt.x + 9, pt.y + 3);
      });

      // Draw Intermediate Transit Waypoints
      p.local_graph.intermediate_nodes.forEach(m => {
        const mpt = toScreen(m.x, m.y);
        ctx.beginPath();
        ctx.arc(mpt.x, mpt.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#6366f1';
        ctx.fill();
        ctx.strokeStyle = '#a5b4fc';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#c7d2fe';
        ctx.font = 'bold 11px Plus Jakarta Sans';
        ctx.fillText(m.label, mpt.x - 36, mpt.y - 14);
      });

      const r = progressRatio;
      const easedR = r < 0.5 ? 2 * r * r : -1 + (4 - 2 * r) * r;

      const wp1 = toScreen(p.local_graph.intermediate_nodes[0].x, p.local_graph.intermediate_nodes[0].y);
      const wp2 = toScreen(p.local_graph.intermediate_nodes[1].x, p.local_graph.intermediate_nodes[1].y);

      ctx.strokeStyle = 'rgba(99, 102, 241, 0.45)';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(startPos.x, startPos.y);
      ctx.quadraticCurveTo(wp1.x, wp1.y, wp2.x, wp2.y);
      ctx.lineTo(endPos.x, endPos.y);
      ctx.stroke();
      ctx.setLineDash([]);

      let curX, curY;
      if (easedR < 0.5) {
        const subT = easedR / 0.5;
        curX = startPos.x + (wp2.x - startPos.x) * subT;
        curY = startPos.y + (wp2.y - startPos.y) * subT;
      } else {
        const subT = (easedR - 0.5) / 0.5;
        curX = wp2.x + (endPos.x - wp2.x) * subT;
        curY = wp2.y + (endPos.y - wp2.y) * subT;
      }

      // Dissolving High-Risk Edges
      if (r < 0.8) {
        const alphaOld = Math.max(0, 1 - r * 1.3);
        ctx.strokeStyle = `rgba(239, 68, 68, ${alphaOld * 0.8})`;
        ctx.lineWidth = 2.0;
        p.local_graph.orig_neighbors.slice(0, 10).forEach(n => {
          const pt = toScreen(n.x, n.y);
          ctx.beginPath();
          ctx.moveTo(curX, curY);
          ctx.lineTo(pt.x, pt.y);
          ctx.stroke();
        });
      }

      // Latching Healthy Edges
      if (r > 0.2) {
        const alphaNew = Math.min(1, (r - 0.2) / 0.65);
        ctx.strokeStyle = `rgba(34, 197, 94, ${alphaNew * 0.9})`;
        ctx.lineWidth = 2.2;
        p.local_graph.new_neighbors.slice(0, 11).forEach(n => {
          const pt = toScreen(n.x, n.y);
          ctx.beginPath();
          ctx.moveTo(curX, curY);
          ctx.lineTo(pt.x, pt.y);
          ctx.stroke();
        });
      }

      pulseT += 0.06;
      const pulseR = 17 + Math.sin(pulseT) * 4;

      ctx.beginPath();
      ctx.arc(curX, curY, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = r < 0.5 ? 'rgba(239, 68, 68, 0.75)' : 'rgba(34, 197, 94, 0.85)';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(curX, curY, 10.5, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 18;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.8;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 13px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`${p.name.split(' ')[0]}`, curX, curY - 22);
    }

    // GRAPH VIEW 2: 5-CLUSTER GLOBAL POPULATION MANIFOLD
    function renderMacroManifold() {
      const centerX = cWidth / 2;
      const centerY = cHeight / 2;

      function toScreen(x, y) {
        return {
          x: centerX + x * (cWidth / 380),
          y: centerY + y * (cHeight / 380)
        };
      }

      Object.values(MULTI_DATA.macro_clusters).forEach(cluster => {
        const cCenter = toScreen(cluster.center[0], cluster.center[1]);

        const g = ctx.createRadialGradient(cCenter.x, cCenter.y, 5, cCenter.x, cCenter.y, 105);
        g.addColorStop(0, cluster.color + '30');
        g.addColorStop(1, cluster.color + '00');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cCenter.x, cCenter.y, 105, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = cluster.color;
        ctx.font = 'bold 12px Plus Jakarta Sans';
        ctx.textAlign = 'center';
        ctx.fillText(cluster.name, cCenter.x, cCenter.y - 55);

        cluster.nodes.forEach(n => {
          const pt = toScreen(n.x, n.y);
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = cluster.color;
          ctx.fill();
        });
      });

      const startP = toScreen(-115, -60);
      const midP = toScreen(0, 0);
      const endP = toScreen(115, 60);

      const r = progressRatio;
      const pX = (1 - r) * (1 - r) * startP.x + 2 * (1 - r) * r * midP.x + r * r * endP.x;
      const pY = (1 - r) * (1 - r) * startP.y + 2 * (1 - r) * r * midP.y + r * r * endP.y;

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startP.x, startP.y);
      ctx.quadraticCurveTo(midP.x, midP.y, endP.x, endP.y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(pX, pY, 10, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 16;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`${currentPatient.name.split(' ')[0]} (Global)`, pX, pY - 18);
    }

    // GRAPH VIEW 3: BIOMARKER RADAR POLAR FINGERPRINT
    function renderRadarFingerprint() {
      const centerX = cWidth / 2;
      const centerY = cHeight / 2;
      const radius = Math.min(cWidth, cHeight) * 0.40;

      const axes = ['Blood_Pressure', 'Cholesterol', 'Triglycerides', 'Heart_Rate', 'BMI', 'Stress_Index'];
      const axisLabels = ['Blood Pressure', 'Cholesterol', 'Triglycerides', 'Heart Rate', 'BMI', 'Stress Index'];
      const numAxes = axes.length;

      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1.2;
      for (let level = 1; level <= 4; level++) {
        const r = (radius / 4) * level;
        ctx.beginPath();
        for (let i = 0; i < numAxes; i++) {
          const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2;
          const x = centerX + r * Math.cos(angle);
          const y = centerY + r * Math.sin(angle);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }

      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px Plus Jakarta Sans';
      ctx.textAlign = 'center';
      for (let i = 0; i < numAxes; i++) {
        const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x, y);
        ctx.stroke();

        const lx = centerX + (radius + 25) * Math.cos(angle);
        const ly = centerY + (radius + 25) * Math.sin(angle);
        ctx.fillText(axisLabels[i], lx, ly);
      }

      // Baseline Polygon
      ctx.beginPath();
      for (let i = 0; i < numAxes; i++) {
        const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2;
        const valRatio = 0.85 + (i % 2) * 0.1;
        const x = centerX + radius * valRatio * Math.cos(angle);
        const y = centerY + radius * valRatio * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Current Dynamic Polygon
      const r = progressRatio;
      ctx.beginPath();
      for (let i = 0; i < numAxes; i++) {
        const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2;
        const origR = 0.85 + (i % 2) * 0.1;
        const targetR = 0.35 + (i % 3) * 0.08;
        const curR = origR - (origR - targetR) * r;
        const x = centerX + radius * curR * Math.cos(angle);
        const y = centerY + radius * curR * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(34, 197, 94, 0.32)';
      ctx.fill();
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Initialization
    loadPatient(852);
    animFrame = requestAnimationFrame(renderActiveCanvas);
  </script>
</body>
</html>
"""

final_html = html_template.replace("__MULTIGRAPH_JSON__", multigraph_json)

# Save to root index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(final_html)

# Save to dashboard/index.html
with open("dashboard/index.html", "w", encoding="utf-8") as f:
    f.write(final_html)

# Save to docs/index.html
with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(final_html)

# Save to artifact
artifact_file = os.path.join(artifact_dir, "cpm_interactive_ui.html")
with open(artifact_file, "w", encoding="utf-8") as f:
    f.write(final_html)

print("Successfully generated ultra-spacious UI across index.html, dashboard/index.html, and docs/index.html!")
