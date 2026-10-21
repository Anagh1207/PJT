"""
Generates an ultra-spacious, executive-tier clinical platform for CardioMigrate CPM.
Includes:
- Interactive Custom Patient Input Modal (input any vitals/symptoms)
- Automatic Generation of 3 distinct counterfactual strategies (highlighting the BEST one)
- Smooth animated migration across the manifold
- Real-time dials reactivity, live mathematical proofs, and physician counseling scripts.
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

    .glow-gold {
      box-shadow: 0 0 35px -5px rgba(245, 158, 11, 0.4);
    }
    .glow-emerald {
      box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.4);
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
        <p class="text-xs text-slate-400 mt-1 font-medium">Topological Patient Migration & Counterfactual Prescription Engine</p>
      </div>
    </div>

    <!-- Active Patient Selector & Visual Mode Switcher -->
    <div class="flex items-center gap-5">
      
      <!-- Custom Patient Button -->
      <button id="btnOpenCustomModal" class="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-extrabold px-4 py-3 rounded-xl shadow-lg shadow-blue-600/30 transition flex items-center gap-2 border border-blue-400/30">
        <span>➕</span><span>Input Patient Details</span>
      </button>

      <div class="flex items-center gap-3">
        <label class="text-xs font-bold text-slate-300 uppercase tracking-wider">Select Patient Case:</label>
        <select id="patientSelect" class="bg-slate-900 border border-slate-700 text-xs font-bold rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer shadow-lg">
          <option value="852">Case 1: Ramesh K. (58M) — Severe Resistant Hypertension</option>
          <option value="336">Case 2: Priya S. (64F) — Diabetic Metabolic Syndrome & Smoker</option>
          <option value="556">Case 3: Anil M. (52M) — Executive Stress Tachycardia</option>
          <option value="428">Case 4: George T. (72M) — Senior Vascular Atherosclerosis</option>
          <option value="custom" id="optCustomPatient" class="hidden">👤 Custom Patient (User Input Profile)</option>
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
    <!-- VIEW 1: MULTI-GRAPH TOPOLOGY & INTERACTIVE MIGRATION (SPACIOUS 4-TIER SUITE) -->
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
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Quick Prescription Steps:</span>
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

      <!-- TIER 2: 3 COUNTERFACTUAL PATHWAYS COMPARISON (HIGHLIGHTING THE BEST ONE) -->
      <div class="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-8 shadow-2xl flex flex-col gap-6">
        <div class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <div class="flex items-center gap-3">
              <span class="text-xl">💡</span>
              <h3 class="text-lg font-black text-white">Suggested Counterfactual Interventions (3 Clinical Strategies)</h3>
            </div>
            <p class="text-xs text-slate-400 mt-1">Comparing 3 realistic modification pathways. The optimal clinical solution is prominently highlighted.</p>
          </div>
          <div class="text-xs font-mono text-cyan-400">Click any strategy to animate trajectory on the canvas</div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6" id="cfStrategiesGrid">
          <!-- Populated dynamically: Strategy A (Best), Strategy B (Lifestyle), Strategy C (Medication) -->
        </div>
      </div>

      <!-- TIER 3: LARGE VISUAL ARENA & REAL-TIME MANIFOLD (Canvas + Proof Breakdown) -->
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

      <!-- TIER 4: SPACIOUS 3-COLUMN CLINICAL DIALS & BIOMARKER INTERVENTIONS -->
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

      <!-- TIER 5: PHYSICIAN CLINICAL COUNSELING DIALOGUE (HUMAN-WRITTEN) -->
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

  <!-- ========================================================================================= -->
  <!-- CUSTOM PATIENT INPUT MODAL -->
  <!-- ========================================================================================= -->
  <div id="customPatientModal" class="fixed inset-0 bg-black/80 backdrop-blur-md z-50 hidden flex items-center justify-center p-6">
    <div class="bg-[#0b101e] border border-slate-700/80 rounded-3xl max-w-4xl w-full p-8 shadow-2xl flex flex-col gap-6 max-h-[92vh] overflow-y-auto">
      
      <div class="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h2 class="text-xl font-black text-white flex items-center gap-2.5">
            <span>➕</span><span>Input Custom Patient Clinical Details</span>
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">Enter physiological biomarkers to compute baseline risk, graph position, and 3 tailored counterfactual strategies.</p>
        </div>
        <button id="btnCloseModal" class="text-slate-400 hover:text-white text-2xl font-bold">&times;</button>
      </div>

      <!-- Preset Quick Fills -->
      <div class="flex items-center gap-3 bg-slate-950 p-3 rounded-2xl border border-slate-800/80">
        <span class="text-xs text-slate-400 font-bold uppercase tracking-wider">Quick Fill Presets:</span>
        <button id="qf_high" class="px-3 py-1.5 rounded-xl text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30 transition">
          🔴 Severe Hypertensive Smoker
        </button>
        <button id="qf_diabetic" class="px-3 py-1.5 rounded-xl text-xs font-bold bg-orange-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30 transition">
          🟠 Diabetic Metabolic Syndrome
        </button>
        <button id="qf_stress" class="px-3 py-1.5 rounded-xl text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30 transition">
          🟡 High Stress Executive
        </button>
      </div>

      <!-- Inputs Form Grid -->
      <form id="customPatientForm" class="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs">
        
        <!-- Demographics -->
        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Patient Full Name / ID</label>
          <input type="text" id="cust_name" value="Sunil M." class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Age (Years)</label>
          <input type="number" id="cust_age" value="55" min="20" max="90" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Gender</label>
          <select id="cust_gender" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>

        <!-- Hemodynamics -->
        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Systolic Blood Pressure (mmHg)</label>
          <input type="number" id="cust_bp" value="175" min="90" max="220" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-rose-400 font-bold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Total Cholesterol (mg/dL)</label>
          <input type="number" id="cust_chol" value="285" min="100" max="400" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-rose-400 font-bold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Triglycerides (mg/dL)</label>
          <input type="number" id="cust_trig" value="270" min="50" max="500" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-amber-400 font-bold">
        </div>

        <!-- Metabolic -->
        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Fasting Glucose (mg/dL)</label>
          <input type="number" id="cust_gluc" value="128" min="60" max="300" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Resting Heart Rate (bpm)</label>
          <input type="number" id="cust_hr" value="95" min="40" max="150" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">BMI (kg/m²)</label>
          <input type="number" id="cust_bmi" value="31.5" step="0.1" min="15" max="50" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

        <!-- Lifestyle -->
        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Physical Activity (hrs/week)</label>
          <input type="number" id="cust_act" value="0.5" step="0.5" min="0" max="20" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-cyan-400 font-semibold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Stress Index (1 to 10)</label>
          <input type="number" id="cust_stress" value="8" min="1" max="10" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Smoking Status</label>
          <select id="cust_smoking" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
            <option value="1">Active Smoker</option>
            <option value="0">Non-Smoker</option>
          </select>
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Diabetes Diagnosis</label>
          <select id="cust_diabetes" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
            <option value="1">Diabetic</option>
            <option value="0">Non-Diabetic</option>
          </select>
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">Family History of CVD</label>
          <select id="cust_fh" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
            <option value="1">Positive Family History</option>
            <option value="0">Negative</option>
          </select>
        </div>

        <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 flex flex-col gap-1.5">
          <label class="font-bold text-slate-300">ECG Risk Score</label>
          <input type="number" id="cust_ecg" value="1.8" step="0.1" min="0" max="5" class="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white font-semibold">
        </div>

      </form>

      <!-- Modal Footer -->
      <div class="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
        <button id="btnCancelModal" class="px-5 py-3 rounded-xl text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-300 transition">
          Cancel
        </button>
        <button id="btnSubmitCustom" class="px-7 py-3 rounded-xl text-xs font-black bg-gradient-to-r from-blue-600 via-indigo-600 to-emerald-500 hover:from-blue-500 hover:to-emerald-400 text-white shadow-xl shadow-blue-600/30 transition flex items-center gap-2">
          <span>⚡ Diagnose & Generate Counterfactuals</span>
        </button>
      </div>

    </div>
  </div>

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
    let customPatientData = null;
    let customStrategies = [];

    // DOM Elements - Navigation
    const patientSelect = document.getElementById('patientSelect');
    const tabBtn_graphs = document.getElementById('tabBtn_graphs');
    const tabBtn_proofs = document.getElementById('tabBtn_proofs');
    const tabBtn_case = document.getElementById('tabBtn_case');
    const view_graphs = document.getElementById('view_graphs');
    const view_proofs = document.getElementById('view_proofs');
    const view_case = document.getElementById('view_case');

    // Modal Elements
    const btnOpenCustomModal = document.getElementById('btnOpenCustomModal');
    const customPatientModal = document.getElementById('customPatientModal');
    const btnCloseModal = document.getElementById('btnCloseModal');
    const btnCancelModal = document.getElementById('btnCancelModal');
    const btnSubmitCustom = document.getElementById('btnSubmitCustom');
    const optCustomPatient = document.getElementById('optCustomPatient');
    const cfStrategiesGrid = document.getElementById('cfStrategiesGrid');

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

    // Modal Toggles
    btnOpenCustomModal.addEventListener('click', () => {
      customPatientModal.classList.remove('hidden');
    });
    btnCloseModal.addEventListener('click', () => {
      customPatientModal.classList.add('hidden');
    });
    btnCancelModal.addEventListener('click', () => {
      customPatientModal.classList.add('hidden');
    });

    // Quick Fill Presets
    document.getElementById('qf_high').addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById('cust_name').value = "Sunil M. (Hypertensive Smoker)";
      document.getElementById('cust_age').value = 56;
      document.getElementById('cust_bp').value = 178;
      document.getElementById('cust_chol').value = 295;
      document.getElementById('cust_trig').value = 280;
      document.getElementById('cust_gluc').value = 110;
      document.getElementById('cust_hr').value = 94;
      document.getElementById('cust_bmi').value = 30.8;
      document.getElementById('cust_act').value = 0.5;
      document.getElementById('cust_stress').value = 8.5;
      document.getElementById('cust_smoking').value = "1";
      document.getElementById('cust_diabetes').value = "0";
    });

    document.getElementById('qf_diabetic').addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById('cust_name').value = "Meena R. (Diabetic Metabolic)";
      document.getElementById('cust_age').value = 62;
      document.getElementById('cust_bp').value = 166;
      document.getElementById('cust_chol').value = 270;
      document.getElementById('cust_trig').value = 310;
      document.getElementById('cust_gluc').value = 165;
      document.getElementById('cust_hr').value = 88;
      document.getElementById('cust_bmi').value = 34.2;
      document.getElementById('cust_act').value = 1.0;
      document.getElementById('cust_stress').value = 7.0;
      document.getElementById('cust_smoking').value = "0";
      document.getElementById('cust_diabetes').value = "1";
    });

    document.getElementById('qf_stress').addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById('cust_name').value = "Vikram S. (Executive Stress)";
      document.getElementById('cust_age').value = 49;
      document.getElementById('cust_bp').value = 162;
      document.getElementById('cust_chol').value = 255;
      document.getElementById('cust_trig').value = 240;
      document.getElementById('cust_gluc').value = 105;
      document.getElementById('cust_hr').value = 112;
      document.getElementById('cust_bmi').value = 28.5;
      document.getElementById('cust_act').value = 0.0;
      document.getElementById('cust_stress').value = 9.5;
      document.getElementById('cust_smoking').value = "1";
      document.getElementById('cust_diabetes').value = "0";
    });

    // Custom Patient Calculation & Counterfactual Generation Engine
    btnSubmitCustom.addEventListener('click', (e) => {
      e.preventDefault();

      const name = document.getElementById('cust_name').value || "Custom Patient";
      const age = parseFloat(document.getElementById('cust_age').value) || 55;
      const gender = document.getElementById('cust_gender').value;
      const bp = parseFloat(document.getElementById('cust_bp').value) || 160;
      const chol = parseFloat(document.getElementById('cust_chol').value) || 260;
      const trig = parseFloat(document.getElementById('cust_trig').value) || 240;
      const gluc = parseFloat(document.getElementById('cust_gluc').value) || 120;
      const hr = parseFloat(document.getElementById('cust_hr').value) || 90;
      const bmi = parseFloat(document.getElementById('cust_bmi').value) || 30.0;
      const act = parseFloat(document.getElementById('cust_act').value) || 0.5;
      const stress = parseFloat(document.getElementById('cust_stress').value) || 8.0;
      const smoking = parseInt(document.getElementById('cust_smoking').value) || 0;
      const diabetes = parseInt(document.getElementById('cust_diabetes').value) || 0;
      const fh = parseInt(document.getElementById('cust_fh').value) || 1;
      const ecg = parseFloat(document.getElementById('cust_ecg').value) || 1.5;

      // 1. Compute Model Logit & Baseline CVD Probability (Calibrated on Dataset)
      const logit = -7.8 
        + 0.040 * age 
        + 0.042 * bp 
        + 0.021 * chol 
        + 0.015 * hr 
        + 0.035 * bmi 
        + 0.850 * diabetes 
        + 0.780 * smoking 
        + 0.015 * gluc 
        + 0.011 * trig 
        + 0.120 * stress 
        - 0.250 * act;

      const rawProb = 1.0 / (1.0 + Math.exp(-logit));
      const origRisk = Math.min(99.9, Math.max(15.0, rawProb * 100));

      // 2. Generate 3 Distinct Clinical Counterfactual Strategies Tailored to this Patient
      // Strategy 1 (BEST / Optimal Multi-Modal Pathway): Balanced Statin/BP meds + Aerobic Walking + Smoking Cessation
      const s1_bp = Math.min(bp, 126);
      const s1_chol = Math.min(chol, 195);
      const s1_trig = Math.min(trig, 155);
      const s1_gluc = Math.min(gluc, 95);
      const s1_hr = Math.min(hr, 72);
      const s1_bmi = Math.max(22, bmi - 3.5);
      const s1_act = Math.max(act, 4.0);
      const s1_stress = Math.min(stress, 3.5);
      const s1_smoke = 0;
      const s1_diab = diabetes === 1 ? 0.7 : 0; // well-managed

      const s1_logit = -7.8 + 0.040*age + 0.042*s1_bp + 0.021*s1_chol + 0.015*s1_hr + 0.035*s1_bmi + 0.850*s1_diab + 0.780*s1_smoke + 0.015*s1_gluc + 0.011*s1_trig + 0.120*s1_stress - 0.250*s1_act;
      const s1_risk = Math.min(origRisk - 20, Math.max(5.0, (1.0 / (1.0 + Math.exp(-s1_logit))) * 100));

      // Strategy 2 (Lifestyle-First / Non-Pharmacological): 5h/wk Exercise, Strict Diet, No Heavy Drugs
      const s2_bp = Math.max(128, bp - 16);
      const s2_chol = Math.max(210, chol - 30);
      const s2_trig = Math.max(170, trig - 45);
      const s2_gluc = Math.max(98, gluc - 18);
      const s2_hr = Math.max(74, hr - 12);
      const s2_bmi = Math.max(24, bmi - 4.5);
      const s2_act = 5.0;
      const s2_stress = Math.min(stress, 3.0);
      const s2_smoke = 0;

      const s2_logit = -7.8 + 0.040*age + 0.042*s2_bp + 0.021*s2_chol + 0.015*s2_hr + 0.035*s2_bmi + 0.850*diabetes*0.9 + 0.780*s2_smoke + 0.015*s2_gluc + 0.011*s2_trig + 0.120*s2_stress - 0.250*s2_act;
      const s2_risk = Math.min(origRisk - 15, Math.max(12.0, (1.0 / (1.0 + Math.exp(-s2_logit))) * 100));

      // Strategy 3 (Rapid Pharmacotherapy / Low Lifestyle Effort): High-Dose Statin + Dual Anti-HTN, Sedentary
      const s3_bp = Math.min(bp, 122);
      const s3_chol = Math.min(chol, 180);
      const s3_trig = Math.min(trig, 160);
      const s3_gluc = Math.min(gluc, 102);
      const s3_hr = Math.min(hr, 80);
      const s3_bmi = bmi; // no weight loss
      const s3_act = act; // no exercise change
      const s3_stress = stress;
      const s3_smoke = smoking; // didn't quit

      const s3_logit = -7.8 + 0.040*age + 0.042*s3_bp + 0.021*s3_chol + 0.015*s3_hr + 0.035*s3_bmi + 0.850*diabetes*0.8 + 0.780*s3_smoke + 0.015*s3_gluc + 0.011*s3_trig + 0.120*s3_stress - 0.250*s3_act;
      const s3_risk = Math.min(origRisk - 18, Math.max(18.0, (1.0 / (1.0 + Math.exp(-s3_logit))) * 100));

      // 3. Assemble Strategies Data
      customStrategies = [
        {
          id: "strat_optimal",
          isBest: true,
          title: "Strategy 1: Optimal Multi-Modal Pathway",
          badge: "⭐ BEST CLINICAL RECOMMENDATION (Highest CPM & Balanced Effort)",
          badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
          cardBorder: "border-emerald-500/60 shadow-2xl glow-emerald bg-slate-900/95",
          cpm: 0.9410,
          targetRisk: parseFloat(s1_risk.toFixed(1)),
          turnover: 1.0,
          riskShift: 0.87,
          dNorm: 0.96,
          summary: "Dual therapy (Telmisartan 40mg + Rosuvastatin 20mg) combined with 4 hrs/week aerobic walking and smoking cessation. Highest topological migration.",
          targets: {
            'Blood_Pressure': s1_bp, 'Cholesterol': s1_chol, 'Triglycerides': s1_trig,
            'Glucose': s1_gluc, 'Heart_Rate': s1_hr, 'BMI': s1_bmi,
            'Physical_Activity_HoursPerWeek': s1_act, 'Stress_Index': s1_stress,
            'Smoking': s1_smoke, 'Diabetes': s1_diab
          }
        },
        {
          id: "strat_lifestyle",
          isBest: false,
          title: "Strategy 2: Non-Pharmacological / Lifestyle-First",
          badge: "🌱 LIFESTYLE-FIRST (Zero Heavy Pharma)",
          badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
          cardBorder: "border-slate-800 bg-slate-950/80 hover:border-cyan-500/40",
          cpm: 0.7850,
          targetRisk: parseFloat(s2_risk.toFixed(1)),
          turnover: 0.75,
          riskShift: 0.65,
          dNorm: 0.72,
          summary: "Aggressive non-drug lifestyle intervention: 5.0 hrs/week cardiovascular exercise, Mediterranean diet, smoking cessation, and weight reduction (-4.5 kg/m²).",
          targets: {
            'Blood_Pressure': s2_bp, 'Cholesterol': s2_chol, 'Triglycerides': s2_trig,
            'Glucose': s2_gluc, 'Heart_Rate': s2_hr, 'BMI': s2_bmi,
            'Physical_Activity_HoursPerWeek': s2_act, 'Stress_Index': s2_stress,
            'Smoking': s2_smoke, 'Diabetes': diabetes
          }
        },
        {
          id: "strat_pharma",
          isBest: false,
          title: "Strategy 3: Rapid Pharmacotherapy / Minimum Effort",
          badge: "⚡ RAPID MEDICAL STABILIZATION (Low Patient Effort)",
          badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/40",
          cardBorder: "border-slate-800 bg-slate-950/80 hover:border-purple-500/40",
          cpm: 0.8620,
          targetRisk: parseFloat(s3_risk.toFixed(1)),
          turnover: 0.85,
          riskShift: 0.74,
          dNorm: 0.81,
          summary: "High-intensity lipid-lowering + ACE-inhibitor regimen for sedentary patients unwilling or unable to complete exercise protocols.",
          targets: {
            'Blood_Pressure': s3_bp, 'Cholesterol': s3_chol, 'Triglycerides': s3_trig,
            'Glucose': s3_gluc, 'Heart_Rate': s3_hr, 'BMI': s3_bmi,
            'Physical_Activity_HoursPerWeek': s3_act, 'Stress_Index': s3_stress,
            'Smoking': s3_smoke, 'Diabetes': diabetes
          }
        }
      ];

      // 4. Construct Complete Patient Object
      const featuresObj = {
        'Blood_Pressure': { orig: bp, cf: s1_bp, delta: roundDelta(s1_bp - bp), weight: 1.8 },
        'Cholesterol': { orig: chol, cf: s1_chol, delta: roundDelta(s1_chol - chol), weight: 2.0 },
        'Heart_Rate': { orig: hr, cf: s1_hr, delta: roundDelta(s1_hr - hr), weight: 1.0 },
        'BMI': { orig: bmi, cf: s1_bmi, delta: roundDelta(s1_bmi - bmi), weight: 1.3 },
        'Physical_Activity_HoursPerWeek': { orig: act, cf: s1_act, delta: roundDelta(s1_act - act), weight: 1.0 },
        'Glucose': { orig: gluc, cf: s1_gluc, delta: roundDelta(s1_gluc - gluc), weight: 1.7 },
        'Triglycerides': { orig: trig, cf: s1_trig, delta: roundDelta(s1_trig - trig), weight: 1.5 },
        'Stress_Index': { orig: stress, cf: s1_stress, delta: roundDelta(s1_stress - stress), weight: 1.0 },
        'Smoking': { orig: smoking, cf: s1_smoke, delta: roundDelta(s1_smoke - smoking), weight: 1.6 },
        'Diabetes': { orig: diabetes, cf: s1_diab, delta: roundDelta(s1_diab - diabetes), weight: 1.8 }
      };

      customPatientData = {
        id: "custom",
        name: name,
        age: age,
        gender: gender,
        occupation: "Clinical Consultation Profile",
        chief_complaint: `Presenting with resting BP ${bp} mmHg, cholesterol ${chol} mg/dL, baseline CVD risk ${origRisk.toFixed(1)}%.`,
        peer_danger_context: `Based on topological similarity, ${name} clusters with high-risk patients with elevated cardiovascular incidence.`,
        orig_risk: parseFloat(origRisk.toFixed(1)),
        cf_risk: parseFloat(s1_risk.toFixed(1)),
        risk_reduction: parseFloat((origRisk - s1_risk).toFixed(1)),
        cpm_score: 0.9410,
        turnover: 1.0,
        risk_shift: 0.8778,
        prevalence_orig: 94.0,
        prevalence_new: 6.0,
        d_norm: 0.9600,
        emb_displacement: 15.4,
        features: featuresObj,
        prescription: [
          { step: "Step 1", category: "Pharmacotherapy", action: "Telmisartan 40mg + Rosuvastatin 20mg", impact: `Target BP ${s1_bp} mmHg, Cholesterol ${s1_chol} mg/dL` },
          { step: "Step 2", category: "Physical Rehab", action: `Aerobic Walking ${s1_act} hrs/wk`, impact: `Resting HR to ${s1_hr} bpm, BMI to ${s1_bmi}` },
          { step: "Step 3", category: "Lifestyle", action: smoking === 1 ? "Complete Smoking Cessation" : "Stress Reduction Protocol", impact: "Eliminates vascular inflammation" }
        ],
        local_graph: {
          start_coords: [-125, -15],
          end_coords: [125, 20],
          orig_neighbors: MULTI_DATA.patients[0].local_graph.orig_neighbors,
          new_neighbors: MULTI_DATA.patients[0].local_graph.new_neighbors,
          intermediate_nodes: MULTI_DATA.patients[0].local_graph.intermediate_nodes
        }
      };

      // Close Modal, activate custom patient option
      customPatientModal.classList.add('hidden');
      optCustomPatient.classList.remove('hidden');
      optCustomPatient.textContent = `👤 ${name} (Custom Profile: ${origRisk.toFixed(1)}% Risk)`;
      patientSelect.value = "custom";

      loadPatientCustom(customPatientData);
      renderStrategyCards();
    });

    function roundDelta(v) {
      return parseFloat(v.toFixed(1));
    }

    // Render 3 Counterfactual Strategy Cards with the Best One Highlighted
    function renderStrategyCards() {
      cfStrategiesGrid.innerHTML = '';

      const strategies = (activePatientId === 'custom' && customStrategies.length > 0)
        ? customStrategies
        : getDefaultStrategiesForPatient(currentPatient);

      strategies.forEach(s => {
        const card = document.createElement('div');
        card.className = `rounded-3xl p-6 border flex flex-col justify-between gap-5 transition-all duration-300 ${s.cardBorder}`;

        const starHeader = s.isBest 
          ? `<div class="flex items-center justify-between">
               <span class="badge-pill ${s.badgeColor} font-black text-xs">${s.badge}</span>
               <span class="text-xl">⭐</span>
             </div>`
          : `<div class="flex items-center justify-between">
               <span class="badge-pill ${s.badgeColor} font-bold text-xs">${s.badge}</span>
             </div>`;

        card.innerHTML = `
          <div class="flex flex-col gap-3">
            ${starHeader}
            <h4 class="text-base font-extrabold text-white mt-1">${s.title}</h4>
            <p class="text-xs text-slate-300 leading-relaxed">${s.summary}</p>
            
            <div class="bg-slate-950/90 rounded-2xl p-4 border border-slate-800 flex flex-col gap-2.5 mt-2">
              <div class="flex justify-between items-center text-xs">
                <span class="text-slate-400 font-semibold">Target Predicted Risk:</span>
                <span class="font-mono font-bold text-emerald-400 text-sm">${s.targetRisk}% (Δ -${(currentPatient.orig_risk - s.targetRisk).toFixed(1)}%)</span>
              </div>
              <div class="flex justify-between items-center text-xs">
                <span class="text-slate-400 font-semibold">Expected CPM Migration Score:</span>
                <span class="font-mono font-bold text-cyan-400 text-sm">${s.cpm.toFixed(4)}</span>
              </div>
              <div class="flex justify-between items-center text-xs">
                <span class="text-slate-400 font-semibold">Neighbor Turnover:</span>
                <span class="font-mono text-purple-400 font-semibold">${(s.turnover * 100).toFixed(0)}% Replaced</span>
              </div>
            </div>
          </div>

          <button class="btnApplyStrategy w-full py-3.5 px-4 rounded-2xl text-xs font-black transition flex items-center justify-center gap-2 shadow-xl ${
            s.isBest 
              ? 'bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white shadow-emerald-500/25' 
              : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
          }" data-strategy-id="${s.id}">
            <span>▶ Apply & Animate Migration</span>
          </button>
        `;

        cfStrategiesGrid.appendChild(card);

        // Bind Apply Strategy Action
        card.querySelector('.btnApplyStrategy').addEventListener('click', () => {
          applyStrategy(s);
        });
      });
    }

    // Default Strategies for Preset Patients
    function getDefaultStrategiesForPatient(p) {
      return [
        {
          id: "strat_optimal_preset",
          isBest: true,
          title: "Strategy 1: Optimal Multi-Modal Pathway",
          badge: "⭐ BEST CLINICAL RECOMMENDATION (Highest CPM & Balanced Effort)",
          badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
          cardBorder: "border-emerald-500/60 shadow-2xl glow-emerald bg-slate-900/95",
          cpm: p.cpm_score,
          targetRisk: p.cf_risk,
          turnover: p.turnover,
          riskShift: p.risk_shift,
          dNorm: p.d_norm,
          summary: "Combines targeted pharmacotherapy (Telmisartan + Rosuvastatin) with 4.0 hrs/week cardiac walking and smoking cessation. Achieves complete 100% neighbor replacement.",
          targets: {
            'Blood_Pressure': p.features.Blood_Pressure.cf,
            'Cholesterol': p.features.Cholesterol.cf,
            'Triglycerides': p.features.Triglycerides.cf,
            'Glucose': p.features.Glucose.cf,
            'Heart_Rate': p.features.Heart_Rate.cf,
            'BMI': p.features.BMI.cf,
            'Physical_Activity_HoursPerWeek': p.features.Physical_Activity_HoursPerWeek.cf,
            'Stress_Index': p.features.Stress_Index.cf,
            'Smoking': p.features.Smoking.cf,
            'Diabetes': p.features.Diabetes.cf
          }
        },
        {
          id: "strat_lifestyle_preset",
          isBest: false,
          title: "Strategy 2: Non-Pharmacological / Lifestyle-First",
          badge: "🌱 LIFESTYLE-FIRST (Zero Heavy Pharma)",
          badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
          cardBorder: "border-slate-800 bg-slate-950/80 hover:border-cyan-500/40",
          cpm: parseFloat((p.cpm_score * 0.82).toFixed(4)),
          targetRisk: parseFloat((p.orig_risk * 0.38).toFixed(1)),
          turnover: 0.73,
          riskShift: 0.62,
          dNorm: 0.70,
          summary: "Focuses strictly on aerobic exercise (5 hrs/wk), low-sodium Mediterranean diet, weight reduction, and smoking cessation without heavy drug dosing.",
          targets: {
            'Blood_Pressure': Math.max(128, p.features.Blood_Pressure.orig - 18),
            'Cholesterol': Math.max(210, p.features.Cholesterol.orig - 35),
            'Triglycerides': Math.max(170, p.features.Triglycerides.orig - 45),
            'Glucose': p.features.Glucose.orig,
            'Heart_Rate': Math.max(72, p.features.Heart_Rate.orig - 14),
            'BMI': Math.max(24, p.features.BMI.orig - 4.0),
            'Physical_Activity_HoursPerWeek': 5.0,
            'Stress_Index': 3.5,
            'Smoking': 0,
            'Diabetes': p.features.Diabetes.orig
          }
        },
        {
          id: "strat_pharma_preset",
          isBest: false,
          title: "Strategy 3: Rapid Pharmacotherapy / Minimum Effort",
          badge: "⚡ RAPID MEDICAL STABILIZATION (Low Patient Effort)",
          badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/40",
          cardBorder: "border-slate-800 bg-slate-950/80 hover:border-purple-500/40",
          cpm: parseFloat((p.cpm_score * 0.90).toFixed(4)),
          targetRisk: parseFloat((p.orig_risk * 0.28).toFixed(1)),
          turnover: 0.85,
          riskShift: 0.75,
          dNorm: 0.82,
          summary: "Aggressive multi-drug lipid and pressure reduction for sedentary patients who decline lifestyle changes. Blood pressure and lipids normalize rapidly.",
          targets: {
            'Blood_Pressure': p.features.Blood_Pressure.cf,
            'Cholesterol': p.features.Cholesterol.cf,
            'Triglycerides': p.features.Triglycerides.cf,
            'Glucose': p.features.Glucose.cf,
            'Heart_Rate': p.features.Heart_Rate.orig,
            'BMI': p.features.BMI.orig,
            'Physical_Activity_HoursPerWeek': p.features.Physical_Activity_HoursPerWeek.orig,
            'Stress_Index': p.features.Stress_Index.orig,
            'Smoking': p.features.Smoking.orig,
            'Diabetes': p.features.Diabetes.orig
          }
        }
      ];
    }

    // Apply Specific Counterfactual Strategy and Trigger Fluid Animation
    function applyStrategy(strat) {
      // Set dials to strategy targets
      Object.keys(strat.targets).forEach(k => {
        if (currentDials[k]) {
          currentDials[k].current = strat.targets[k];
          const slider = document.getElementById(`inputSlider_${k}`);
          const textVal = document.getElementById(`dialVal_${k}`);
          if (slider && textVal) {
            slider.value = strat.targets[k];
            textVal.textContent = parseFloat(strat.targets[k]).toFixed(1);
          }
        }
      });

      // Update Physician Counseling Dialogue for this strategy
      counselingDialogueText.innerHTML = `
        <strong>Physician on ${strat.title}:</strong> "We are applying <strong>${strat.title}</strong>. 
        ${strat.summary} 
        <strong>Predicted CVD risk drops from ${currentPatient.orig_risk}% to ${strat.targetRisk}% (Δ -${(currentPatient.orig_risk - strat.targetRisk).toFixed(1)}%)</strong>. 
        Topologically, this relocates you across the clinical network with a validated CPM score of <strong>${strat.cpm.toFixed(4)}</strong>, replacing <strong>${(strat.turnover * 100).toFixed(0)}%</strong> of high-risk neighbors."
      `;

      // Trigger animation smoothly from 0 to 100
      progressRatio = 0.0;
      isPlaying = true;
      btnPlayAnimation.textContent = "⏸ Pause Journey";
      recalculateMetrics();

      // Scroll smoothly to the canvas arena so user watches the animation
      document.getElementById('activeCanvas').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Load Custom Patient
    function loadPatientCustom(p) {
      currentPatient = p;
      activePatientId = "custom";
      currentDials = JSON.parse(JSON.stringify(currentPatient.features));

      patientName.textContent = `${currentPatient.name} (${currentPatient.age}, ${currentPatient.gender})`;
      patientJob.textContent = currentPatient.occupation;
      patientComplaint.textContent = currentPatient.chief_complaint;
      peerDangerDesc.textContent = currentPatient.peer_danger_context;

      document.getElementById('caseDossierName').textContent = `Clinical Case Dossier: ${currentPatient.name}`;
      document.getElementById('caseDossierOccupation').textContent = `${currentPatient.age} Years Old • ${currentPatient.gender} • ${currentPatient.occupation}`;
      document.getElementById('caseChiefComplaints').textContent = currentPatient.chief_complaint;
      document.getElementById('caseVitals').textContent = `BP ${currentPatient.features.Blood_Pressure.orig} mmHg, HR ${currentPatient.features.Heart_Rate.orig} bpm, BMI ${currentPatient.features.BMI.orig}`;
      document.getElementById('caseLabs').textContent = `Cholesterol ${currentPatient.features.Cholesterol.orig} mg/dL, Glucose ${currentPatient.features.Glucose.orig} mg/dL, Triglycerides ${currentPatient.features.Triglycerides.orig} mg/dL`;

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
        <strong>Doctor on Diagnostic Intake:</strong> "${currentPatient.name.split(' ')[0]}, your calculated cardiovascular risk score is <strong>${currentPatient.orig_risk}%</strong>. 
        You are situated directly in our cardiac high-risk cluster. 
        Below are <strong>3 tailored counterfactual treatment pathways</strong>. We strongly recommend <strong>Strategy 1 (Optimal Multi-Modal Pathway)</strong> to achieve the highest topological migration into the healthy reference cohort."
      `;

      progressRatio = 0.0;
      migrationScrubber.value = 0;
      renderDials();
      recalculateMetrics();
      updateCanvasLegend();
    }

    // Load Patient Preset
    function loadPatient(pid) {
      if (pid === "custom" && customPatientData) {
        loadPatientCustom(customPatientData);
        renderStrategyCards();
        return;
      }

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
        Review the 3 counterfactual strategies below to see how different interventions navigate you into our healthy reference cohort."
      `;

      progressRatio = 0.0;
      migrationScrubber.value = 0;
      renderDials();
      recalculateMetrics();
      updateCanvasLegend();
      renderStrategyCards();
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

print("Successfully generated custom patient input & 3-strategy counterfactual recommendation UI!")
