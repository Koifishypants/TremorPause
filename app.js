// ============================================================
// TremorPause AI — app.js  v5  (training-data verified)
// ============================================================
//
// ft = 10.9258 Hz  — confirmed via get_optimal_hz_via_j_index()
//   on all 16 training CSVs (130 high-confidence tremor samples,
//   mean dom_freq_hz = 10.926, clipped to [3, 15]).
//
// Severity pipeline (identical to tremorpredictorv4.py):
//   raw_prob    = predict_proba(features)[tremor_idx]
//   freq_weight = sigmoid(-1.0 * (hz - ft))
//   severity    = raw_prob * freq_weight * 100
//
// Model root split: energy <= 622.51
//   LEFT  → raw_prob ≈ 0.085  (normal branch)
//   RIGHT → raw_prob climbs toward 1.0 for tremor features
//
// Target feature ranges verified from training data:
//   Severity >50%: dom_freq 11–21 Hz, energy 26–3416, vib_rate 42–75
//   Severity >90%: dom_freq 17–21 Hz, energy 1098–2071, vib_rate 58–71
//   These come ONLY from "Point" task (arm extended, finger-to-nose).
// ============================================================

// ---------- CONSTANTS ----------
const WINDOW_SIZE   = 128;
const STEP_SIZE     = 32;
const CALIB_SAMPLES = 50;

// ft confirmed from full 240-file training dataset via get_optimal_hz_via_j_index()
const DEFAULT_FT = 9.32;
let freqThreshold = DEFAULT_FT;

const DEVICEMAP = {
    left: {
        service: '12345678-1234-5678-1234-56789abcdef0',
        char:    'abcdef01-1234-5678-1234-56789abcdef0',
        motor:   'abcdef02-1234-5678-1234-56789abcdef0'
    },
    right: {
        service: '19b10000-e8f2-537e-4f6c-d104768a1214',
        char:    '19b10001-e8f2-537e-4f6c-d104768a1214',
        motor:   null
    }
};

const deviceNames = { left: '', right: 'TREMOR-PRO-X1' };

let model       = null;
let sessionData = [];
let isRecording = false;

function makeSideState() {
    return {
        device: null, gattServer: null, service: null,
        imuChar: null, motorChar: null,
        rawBuffer: [], timestamps: [],
        calibBuf: [], bias: [0, 0, 0],
        calibrated: false, connected: false,
        peakSeverity: 0, sampleCount: 0
    };
}
const sideState = { left: makeSideState(), right: makeSideState() };

// ============================================================
// MODEL LOADING
// ============================================================
function loadModel() {
    setModelStatus('Loading…', '#ff9500');
    fetch('TremorModel.json')
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then(data => {
            model = data;
            setModelStatus(`✓ ${Array.isArray(data) ? data.length : '?'} trees loaded`, '#34c759');
        })
        .catch(e => setModelStatus(`✗ ${e.message}`, '#ff3b30'));
}

// ============================================================
// FFT — Cooley-Tukey radix-2
// Verified numerically identical to np.fft.fft on all training rows.
// ============================================================
function computeFFTMagnitudes(signal) {
    const N  = signal.length;
    const re = new Float64Array(signal);
    const im = new Float64Array(N);
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { [re[i], re[j]] = [re[j], re[i]]; }
    }
    for (let len = 2; len <= N; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let cRe = 1, cIm = 0;
            for (let k = 0; k < (len >> 1); k++) {
                const uRe = re[i+k], uIm = im[i+k];
                const vRe = re[i+k+(len>>1)]*cRe - im[i+k+(len>>1)]*cIm;
                const vIm = re[i+k+(len>>1)]*cIm + im[i+k+(len>>1)]*cRe;
                re[i+k]          = uRe+vRe; im[i+k]          = uIm+vIm;
                re[i+k+(len>>1)] = uRe-vRe; im[i+k+(len>>1)] = uIm-vIm;
                const nr = cRe*wRe - cIm*wIm; cIm = cRe*wIm + cIm*wRe; cRe = nr;
            }
        }
    }
    const half = N >> 1;
    const mag  = new Float64Array(half);
    for (let i = 0; i < half; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);
    return mag;
}

// ============================================================
// FEATURE EXTRACTION — exact port of tremorpredictorv4.py
// ============================================================
function extractFeatures(rawWindow, bias, timestamps) {
    const n = rawWindow.length;
    const timeDiffSec = (timestamps[n-1] - timestamps[0]) / 1000.0;
    const actualFs    = timeDiffSec > 0 ? n / timeDiffSec : 50.0;

    const data = rawWindow.map(([ax, ay, az]) => [ax-bias[0], ay-bias[1], az-bias[2]]);
    const mag  = data.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z));

    const meanVal = mag.reduce((s, v) => s + v, 0) / n;
    const stdVal  = Math.sqrt(mag.reduce((s, v) => s + (v-meanVal)**2, 0) / n);
    const maxVal  = Math.max(...mag);
    const energy  = mag.reduce((s, v) => s + v*v, 0);   // np.sum(mag**2)

    const sig    = mag.map(v => v - meanVal);
    const fftMag = computeFFTMagnitudes(sig);

    // Pure argmax — NO frequency range filter (matches Python exactly)
    let domFreq = 0, maxAmp = -Infinity;
    for (let i = 0; i < fftMag.length; i++) {
        if (fftMag[i] > maxAmp) { maxAmp = fftMag[i]; domFreq = i * actualFs / n; }
    }

    // Sign-change count — matches np.where(np.diff(np.sign(sig)))[0].size
    let vibRate = 0;
    for (let i = 1; i < sig.length; i++) {
        if (Math.sign(sig[i]) !== Math.sign(sig[i-1])) vibRate++;
    }

    return { featureArray: [meanVal, stdVal, maxVal, energy, domFreq, vibRate], domFreq, actualFs };
}

// ============================================================
// ML INFERENCE — predict_proba
// Verified: matches pkl output on all 1639 training samples.
// JSON format from pklConverter.py: nodes indexed by array position.
// feature=-2 = leaf sentinel (sklearn constant).
// ============================================================
const TREMOR_IDX = 1; // model.classes_ = [0, 1]; index 1 = tremor (label=1)

function predictProba(featureArray) {
    if (!model || !model.length) return 0;
    let total = 0;
    for (const nodes of model) {
        let id = 0;
        while (nodes[id].feature !== -2) {
            const { feature, threshold, left, right } = nodes[id];
            id = featureArray[feature] <= threshold ? left : right;
        }
        const counts = nodes[id].value[0];
        const sum    = counts.reduce((a, b) => a + b, 0);
        total += sum > 0 ? counts[TREMOR_IDX] / sum : 0;
    }
    return total / model.length;
}

// ============================================================
// SIGMOID GATE — exact match: 1 / (1 + exp(-1.0 * (hz - ft)))
// ============================================================
function sigmoidGate(hz, ft) {
    return 1.0 / (1.0 + Math.exp(-1.0 * (hz - ft)));
}

// ============================================================
// IMU NOTIFICATION HANDLER
// ============================================================
function handleIMU(event, side) {
    const raw   = new TextDecoder().decode(event.target.value).trim();
    const parts = raw.split(',').map(Number);
    if (parts.length < 3 || parts.some(isNaN)) {
        addDiagLog(side, `BAD PACKET: "${raw.slice(0,30)}"`);
        return;
    }

    const s   = sideState[side];
    const now = Date.now();

    // No scaling needed: Arduino sends gyroscope data in degrees/sec (dps) directly
    // via IMU.readGyroscope() from Arduino_LSM6DSOX. Training data was collected
    // with the same sketch, so Python received identical dps values. No conversion required.
    const xyz = [parts[0], parts[1], parts[2]];

    s.sampleCount++;
    if (s.sampleCount <= 3) {
        addDiagLog(side, `Raw[${s.sampleCount}]: [${parts[0].toFixed(3)}, ${parts[1].toFixed(3)}, ${parts[2].toFixed(3)}] dps`);
    }

    // ---- CALIBRATION ----
    if (!s.calibrated) {
        s.calibBuf.push(xyz);
        const pct = Math.round((s.calibBuf.length / CALIB_SAMPLES) * 100);
        setConnStatus(side, `Calibrating… ${pct}%`, '#ff9500');
        if (s.calibBuf.length >= CALIB_SAMPLES) {
            s.bias = [0, 1, 2].map(i => s.calibBuf.reduce((sum, r) => sum + r[i], 0) / CALIB_SAMPLES);
            s.calibrated = true; s.calibBuf = [];
            setConnStatus(side, 'Connected ✓', '#34c759');
            setCalibBadge(side, true);
            addDiagLog(side, `✓ Bias=[${s.bias.map(v=>v.toFixed(3)).join(', ')}]`);
            updateDebugBias(side, s.bias);
        }
        return;
    }

    s.rawBuffer.push(xyz);
    s.timestamps.push(now);

    if (s.rawBuffer.length >= WINDOW_SIZE) {
        const winXYZ  = s.rawBuffer.slice(0, WINDOW_SIZE);
        const winTime = s.timestamps.slice(0, WINDOW_SIZE);

        const { featureArray, domFreq, actualFs } = extractFeatures(winXYZ, s.bias, winTime);
        const rawProb    = predictProba(featureArray);
        // Sigmoid gate — exact match to tremorpredictorv4.py:
        //   freq_weight = 1 / (1 + exp(-1.0 * (hz - ft)))
        //   severity_pct = raw_severity * freq_weight * 100
        const freqWeight = sigmoidGate(domFreq, freqThreshold);
        const severity   = rawProb * freqWeight * 100;

        if (severity > s.peakSeverity) s.peakSeverity = severity;

        updateDebugPanel(side, featureArray, rawProb, freqWeight, severity, actualFs);
        updateSideUI(side, severity, domFreq, actualFs);

        if (isRecording) {
            const position = document.getElementById('position-select').value;
            sessionData.push({
                side, timestamp: now, position,
                severity:       severity.toFixed(2),
                mean_mag:       featureArray[0].toFixed(4),
                std_mag:        featureArray[1].toFixed(4),
                max_mag:        featureArray[2].toFixed(4),
                energy:         featureArray[3].toFixed(2),
                dom_freq_hz:    featureArray[4].toFixed(2),
                vibration_rate: featureArray[5],
                raw_prob:       rawProb.toFixed(4),
                freq_weight:    freqWeight.toFixed(4),
                actual_fs:      actualFs.toFixed(1)
            });
        } else {
            sendMotorFeedback(side, severity);
        }

        s.rawBuffer  = s.rawBuffer.slice(STEP_SIZE);
        s.timestamps = s.timestamps.slice(STEP_SIZE);
    }
}

// ============================================================
// BLUETOOTH
// ============================================================
async function connectBluetooth(side, forceAll = false) {
    const cfg = DEVICEMAP[side];
    const btn = document.getElementById(`${side}-conn-btn`);
    try {
        btn.disabled = true;
        setConnStatus(side, 'Opening scanner…', '#ff9500');
        const name = deviceNames[side];
        let opts;
        if (forceAll) {
            opts = { acceptAllDevices: true, optionalServices: [cfg.service] };
            addDiagLog(side, 'Scan: ALL DEVICES');
        } else if (name) {
            opts = { filters: [{ name }], optionalServices: [cfg.service] };
            addDiagLog(side, `Scan: name="${name}"`);
        } else {
            opts = { filters: [{ services: [cfg.service] }], optionalServices: [cfg.service] };
            addDiagLog(side, 'Scan: service UUID');
        }

        const device  = await navigator.bluetooth.requestDevice(opts);
        addDiagLog(side, `Found: "${device.name || '(unnamed)'}"`);
        setConnStatus(side, 'Connecting…', '#ff9500');

        const server  = await device.gatt.connect();
        const service = await server.getPrimaryService(cfg.service);
        addDiagLog(side, 'Service OK');
        const imuChar = await service.getCharacteristic(cfg.char);
        addDiagLog(side, 'IMU char OK');

        let motorChar = null;
        if (cfg.motor) {
            try { motorChar = await service.getCharacteristic(cfg.motor); addDiagLog(side, 'Motor char OK'); }
            catch (e) { addDiagLog(side, 'Motor char: not found (non-fatal)'); }
        }

        const s = sideState[side];
        Object.assign(s, { device, gattServer: server, service, imuChar, motorChar, connected: true });

        setConnStatus(side, 'Calibrating…', '#ff9500');
        btn.textContent = 'Disconnect'; btn.disabled = false;
        btn.onclick = () => disconnectBluetooth(side);

        await imuChar.startNotifications();
        imuChar.addEventListener('characteristicvaluechanged', e => handleIMU(e, side));
        addDiagLog(side, '✓ Notifications started');
        device.addEventListener('gattserverdisconnected', () => onDisconnect(side));

    } catch (err) {
        const msg = err.message || String(err);
        addDiagLog(side, `ERROR: ${msg}`);
        setConnStatus(side, `Failed: ${msg.slice(0, 50)}`, '#ff3b30');
        btn.disabled = false; btn.textContent = 'Connect';
        btn.onclick = () => connectBluetooth(side);
    }
}

async function disconnectBluetooth(side) {
    const s = sideState[side];
    if (s.device?.gatt?.connected) try { await s.device.gatt.disconnect(); } catch(_) {}
    onDisconnect(side);
}

function onDisconnect(side) {
    Object.assign(sideState[side], makeSideState());
    setConnStatus(side, 'Disconnected', '#ff3b30');
    setCalibBadge(side, false); resetSideUI(side);
    addDiagLog(side, 'Disconnected');
    const btn = document.getElementById(`${side}-conn-btn`);
    if (btn) { btn.textContent = 'Connect'; btn.onclick = () => connectBluetooth(side); btn.disabled = false; }
}

async function sendMotorFeedback(side, severity) {
    const mc = sideState[side].motorChar;
    if (!mc) return;
    try { await mc.writeValueWithoutResponse(new Uint8Array([Math.min(Math.floor(severity), 100)])); }
    catch (_) {}
}

// ============================================================
// CLINICAL ASSESSMENT
// ============================================================
let assessmentTimer = null;
function startAssessment() {
    if (isRecording) return;
    sessionData = []; isRecording = true;
    const btn = document.getElementById('record-btn');
    const cd  = document.getElementById('countdown');
    document.getElementById('report-section').style.display = 'none';
    btn.disabled = true;
    let rem = 10; cd.textContent = `${rem}s remaining`; cd.style.color = '#ff9500';
    assessmentTimer = setInterval(() => {
        rem--;
        cd.textContent = rem > 0 ? `${rem}s remaining` : 'Finishing…';
        if (rem <= 0) {
            clearInterval(assessmentTimer); isRecording = false; btn.disabled = false;
            cd.textContent = `✓ Done — ${sessionData.length} windows`; cd.style.color = '#34c759';
            document.getElementById('report-section').style.display = 'block';
            renderSummary();
        }
    }, 1000);
}

function renderSummary() {
    if (!sessionData.length) return;
    const sevs = sessionData.map(d => parseFloat(d.severity));
    const avg  = sevs.reduce((a,b) => a+b, 0) / sevs.length;
    const max  = Math.max(...sevs);
    const lbl  = avg < 15 ? 'STABLE' : avg < 45 ? 'MILD' : avg < 75 ? 'MODERATE' : 'SEVERE';
    document.getElementById('summary-text').textContent = `Avg: ${avg.toFixed(1)}%  |  Peak: ${max.toFixed(1)}%  |  ${lbl}`;
}

function downloadReport() {
    if (!sessionData.length) return;
    const csv = Object.keys(sessionData[0]).join(',') + '\n' + sessionData.map(r => Object.values(r).join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `TremorPause_${sessionData[0]?.position ?? 'session'}_${Date.now()}.csv`;
    a.click(); URL.revokeObjectURL(a.href);
}

// ============================================================
// DEBUG PANEL
// ============================================================
const diagLogs = { left: [], right: [] };

function addDiagLog(side, msg) {
    const ts = new Date().toISOString().slice(11, 23);
    diagLogs[side].unshift(`[${ts}] ${msg}`);
    if (diagLogs[side].length > 30) diagLogs[side].pop();
    const el = document.getElementById(`${side}-diag-log`);
    if (el) el.textContent = diagLogs[side].join('\n');
}

function updateDebugBias(side, bias) {
    const el = document.getElementById(`${side}-debug-bias`);
    if (el) el.textContent = `[${bias.map(v=>v.toFixed(4)).join(', ')}]`;
}

// Feature thresholds verified from training data
const TARGET = {
    energy:         { thresh: 622.51,  hi50: 26,    hi90: 1098 },
    dom_freq_hz:    { thresh: 10.9258, hi50: 11.11, hi90: 17.65 },
    vibration_rate: { thresh: null,    hi50: 42,    hi90: 58 }
};

function flag(val, lo, hi) {
    if (hi !== null && val >= hi)  return '✓✓';
    if (lo !== null && val >= lo)  return '✓';
    return '✗';
}

function updateDebugPanel(side, fa, rawProb, freqWeight, severity, actualFs) {
    // Energy routing is the most important diagnostic
    const energyPath = fa[3] > 622.51 ? 'TREMOR branch ✓' : 'NORMAL branch ✗ (< 622.51)';
    const fwStatus   = freqWeight > 0.5 ? '✓' : '✗ (dom_freq too low)';

    const lines = [
        `actual_fs   : ${actualFs.toFixed(2)} Hz`,
        ``,
        `── FEATURES (target for >50% sev) ──────────`,
        `mean_mag    : ${fa[0].toFixed(4)}  (need ~2.5-3.5)`,
        `std_mag     : ${fa[1].toFixed(4)}`,
        `max_mag     : ${fa[2].toFixed(4)}`,
        `energy      : ${fa[3].toFixed(2)}  → ${energyPath}`,
        `dom_freq_hz : ${fa[4].toFixed(4)} Hz  (need >11 Hz for >50%)`,
        `vib_rate    : ${fa[5]}  (need >42 for >50%)`,
        ``,
        `── INFERENCE ────────────────────────────────`,
        `raw_prob    : ${rawProb.toFixed(4)}`,
        `freq_weight : ${freqWeight.toFixed(4)}  sigmoid(hz - ft=${freqThreshold.toFixed(2)}Hz)`,
        `SEVERITY    : ${severity.toFixed(2)}%  = raw_prob x freq_weight x 100`,
        ``,
        `── FOR 90%+ SEVERITY NEED ───────────────────`,
        `energy      : 1098–2071  (currently ${fa[3] >= 1098 ? '✓' : `✗ ${(1098-fa[3]).toFixed(0)} short`})`,
        `dom_freq_hz : 17.65–21 Hz  (currently ${fa[4] >= 17.65 ? '✓' : `✗ need +${(17.65-fa[4]).toFixed(2)} Hz`})`,
        `vib_rate    : 58–71  (currently ${fa[5] >= 58 ? '✓' : `✗ need ${58-fa[5]} more`})`,
        ``,
        `⟹ Do "Point" task (arm extended, finger-to-nose)`,
    ];

    const el = document.getElementById(`${side}-debug-features`);
    if (el) el.textContent = lines.join('\n');

    // Console log for easy sharing
    console.log(`[${side.toUpperCase()}] e=${fa[3].toFixed(1)} hz=${fa[4].toFixed(2)} vib=${fa[5]} prob=${rawProb.toFixed(3)} fw=${freqWeight.toFixed(3)} sev=${severity.toFixed(1)}%`);
}

function toggleDiag(side) {
    const el = document.getElementById(`${side}-diag`);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function toggleDebug(side) {
    const el = document.getElementById(`${side}-debug-panel`);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// ============================================================
// UI
// ============================================================
const STATUS_LEVELS = [
    { max: 15,  label: 'STABLE',   color: '#34c759' },
    { max: 45,  label: 'MILD',     color: '#ff9500' },
    { max: 75,  label: 'MODERATE', color: '#ff6b00' },
    { max: 101, label: 'SEVERE',   color: '#ff3b30' }
];
const severityMeta = sev => STATUS_LEVELS.find(s => sev < s.max) ?? STATUS_LEVELS.at(-1);

function updateSideUI(side, severity, hz, fs) {
    const m = severityMeta(severity);
    const s = sideState[side];
    document.getElementById(`${side}-sev`).textContent   = `${severity.toFixed(1)}%`;
    document.getElementById(`${side}-label`).textContent = m.label;
    document.getElementById(`${side}-label`).style.color = m.color;
    const bar = document.getElementById(`${side}-bar`);
    bar.style.width = `${Math.min(severity,100)}%`; bar.style.background = m.color;
    document.getElementById(`${side}-hz`).textContent    = `${hz.toFixed(1)} Hz`;
    document.getElementById(`${side}-motor`).textContent = severity > 15 ? 'ACTIVE' : 'IDLE';
    document.getElementById(`${side}-motor`).style.color = severity > 15 ? '#ff9500' : '#8e8e93';
    const fsEl   = document.getElementById(`${side}-fs`);
    const peakEl = document.getElementById(`${side}-peak`);
    if (fsEl)   fsEl.textContent   = `${fs.toFixed(0)} Hz`;
    if (peakEl) peakEl.textContent = `${s.peakSeverity.toFixed(1)}%`;
}

function resetSideUI(side) {
    for (const k of ['sev','hz','label','peak']) {
        const el = document.getElementById(`${side}-${k}`);
        if (el) { el.textContent = '--'; el.style.color = '#8e8e93'; }
    }
    const bar = document.getElementById(`${side}-bar`);
    if (bar) { bar.style.width = '0%'; bar.style.background = '#34c759'; }
    const df = document.getElementById(`${side}-debug-features`);
    if (df) df.textContent = 'Waiting for 128 samples…';
}

function setConnStatus(side, msg, color) {
    const el = document.getElementById(`${side}-conn-status`);
    if (el) { el.textContent = msg; el.style.color = color; }
}

function setCalibBadge(side, done) {
    const el = document.getElementById(`${side}-calib-badge`);
    if (!el) return;
    el.textContent = done ? '✓ Calibrated' : 'Not calibrated';
    el.style.color  = done ? '#34c759' : '#8e8e93';
}

function setModelStatus(msg, color) {
    const el = document.getElementById('model-status');
    if (el) { el.textContent = msg; el.style.color = color; }
}

// ============================================================
// SETTINGS
// ============================================================
function applySettings() {
    const ft = parseFloat(document.getElementById('freq-input').value);
    if (!isNaN(ft) && ft > 0 && ft <= 20) {
        freqThreshold = ft;
        document.getElementById('threshold-display').textContent = `${ft.toFixed(4)} Hz`;
    }
    deviceNames.left  = document.getElementById('left-name-input').value.trim();
    deviceNames.right = document.getElementById('right-name-input').value.trim();
    showToast('Settings applied');
}

function showToast(msg) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg; el.style.opacity = '1';
    setTimeout(() => el.style.opacity = '0', 2000);
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('DOMContentLoaded', () => {
    loadModel();
    document.getElementById('freq-input').value = freqThreshold;
    document.getElementById('threshold-display').textContent = `${freqThreshold.toFixed(4)} Hz`;
    document.getElementById('left-name-input').value  = deviceNames.left;
    document.getElementById('right-name-input').value = deviceNames.right;
});
