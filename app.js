// ============================================================
// TremorPause AI — app.js  v3 (corrected)
// ============================================================
// Faithful port of tremorpredictorv4.py
//
// KEY CORRECTIONS vs previous version:
//  1. BLE: acceptAllDevices fallback + cached characteristics + diag log
//  2. FFT: pure argmax over [0..n/2-1] — NO frequency range filter (matches Python)
//  3. Vibration rate: Math.sign() comparison — catches zero-crossings (matches Python)
//  4. Stillness guard REMOVED from predictor path (only in data collector)
//  5. Raw buffer stores un-corrected XYZ; bias applied inside extractFeatures (matches Python)
// ============================================================

// ---------- CONSTANTS ----------
const WINDOW_SIZE   = 128;
const STEP_SIZE     = 32;
const CALIB_SAMPLES = 50;

// Sigmoid center (ft).
// Python computes this from training CSVs via get_optimal_hz_via_j_index().
// Paste the printed value into the Settings panel.
let freqThreshold = 4.0;

// UUIDs — taken verbatim from original app.js / Python config
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

// Runtime name overrides — editable in Settings
// Right device defaults to 'TREMOR-PRO-X1' from original app.js
const deviceNames = { left: '', right: 'TREMOR-PRO-X1' };

// ---------- GLOBAL STATE ----------
let model       = null;
let sessionData = [];
let isRecording = false;

function makeSideState() {
    return {
        device:       null,   // BluetoothDevice
        gattServer:   null,   // BluetoothRemoteGATTServer
        service:      null,   // BluetoothRemoteGATTService
        imuChar:      null,   // cached IMU characteristic
        motorChar:    null,   // cached motor characteristic
        rawBuffer:    [],     // raw (pre-bias) xyz — matches Python
        timestamps:   [],     // ms
        calibBuf:     [],     // xyz during calibration
        bias:         [0,0,0],
        calibrated:   false,
        connected:    false,
        peakSeverity: 0
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
            const n = Array.isArray(data) ? data.length : '?';
            setModelStatus(`✓ Ready — ${n} trees`, '#34c759');
        })
        .catch(e => setModelStatus(`✗ ${e.message}`, '#ff3b30'));
}

// ============================================================
// FFT — Cooley-Tukey radix-2 in-place
// Returns one-sided magnitude spectrum, indices 0..n/2-1
// Matches np.fft.fft for real input, n must be power of 2
// ============================================================
function computeFFTMagnitudes(signal) {
    const N  = signal.length;
    const re = new Float64Array(signal);
    const im = new Float64Array(N);

    // Bit-reversal permutation
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { [re[i], re[j]] = [re[j], re[i]]; }
    }

    // Butterfly stages
    for (let len = 2; len <= N; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let cRe = 1, cIm = 0;
            for (let k = 0; k < (len >> 1); k++) {
                const uRe = re[i+k],           uIm = im[i+k];
                const vRe = re[i+k+(len>>1)] * cRe - im[i+k+(len>>1)] * cIm;
                const vIm = re[i+k+(len>>1)] * cIm + im[i+k+(len>>1)] * cRe;
                re[i+k]          = uRe + vRe;  im[i+k]          = uIm + vIm;
                re[i+k+(len>>1)] = uRe - vRe;  im[i+k+(len>>1)] = uIm - vIm;
                const nr = cRe*wRe - cIm*wIm;  cIm = cRe*wIm + cIm*wRe;  cRe = nr;
            }
        }
    }

    const half = N >> 1;
    const mag  = new Float64Array(half);
    for (let i = 0; i < half; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);
    return mag;
}

// ============================================================
// FEATURE EXTRACTION
// Exact port of extract_features() in tremorpredictorv4.py
//
// rawWindow  : [[ax,ay,az], ...] — RAW, not bias-corrected
// bias       : [bx, by, bz]
// timestamps : [ms, ms, ...]
//
// Python:
//   data      = window - bias
//   mag       = sqrt(sum(data**2, axis=1))
//   sig       = mag - mean(mag)
//   fft_vals  = |fft(sig)|
//   idx       = argmax(fft_vals[:n//2])   ← NO frequency range filter
//   dom_freq  = |freqs[idx]|              ← freqs[k] = k * fs / n
//   vib_rate  = count sign-changes in sig (np.diff(np.sign))
//
// NOTE: tremorpredictorv4.py has NO stillness guard — always computes all features.
//       (The guard is only in tremordatav3.py, the data collector.)
// ============================================================
function extractFeatures(rawWindow, bias, timestamps) {
    const n = rawWindow.length;

    // Actual sampling frequency: actual_fs = WINDOW_SIZE / time_diff
    const timeDiffSec = (timestamps[n-1] - timestamps[0]) / 1000.0;
    const actualFs    = timeDiffSec > 0 ? n / timeDiffSec : 50.0;

    // Bias correction: data = np.array(window) - self.bias
    const data = rawWindow.map(([ax, ay, az]) => [ax - bias[0], ay - bias[1], az - bias[2]]);

    // Magnitude: mag = sqrt(sum(data**2, axis=1))
    const mag = data.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z));

    // Scalar features
    const meanVal = mag.reduce((s, v) => s + v, 0) / n;
    const stdVal  = Math.sqrt(mag.reduce((s, v) => s + (v-meanVal)**2, 0) / n);
    const maxVal  = Math.max(...mag);
    const energy  = mag.reduce((s, v) => s + v*v, 0);   // np.sum(mag**2)

    // Mean-centred signal: sig = mag - mean_val
    const sig = mag.map(v => v - meanVal);

    // FFT
    const fftMag = computeFFTMagnitudes(sig);

    // Dominant frequency:
    //   Python: idx = np.argmax(fft_vals[:n//2])  — includes DC (index 0 → 0 Hz)
    //           dom_freq = abs(freqs[idx]) = idx * actual_fs / n
    //   NO frequency range filter — pure argmax of the whole one-sided spectrum
    let domFreq = 0, maxAmp = -Infinity;
    for (let i = 0; i < fftMag.length; i++) {
        if (fftMag[i] > maxAmp) {
            maxAmp  = fftMag[i];
            domFreq = i * actualFs / n;
        }
    }

    // Vibration rate:
    //   Python: np.where(np.diff(np.sign(sig)))[0].size
    //   Counts every index where np.sign changes — includes transitions through 0
    //   np.sign(x): -1 if x<0, 0 if x==0, +1 if x>0
    let vibRate = 0;
    for (let i = 1; i < sig.length; i++) {
        if (Math.sign(sig[i]) !== Math.sign(sig[i-1])) vibRate++;
    }

    return {
        featureArray: [meanVal, stdVal, maxVal, energy, domFreq, vibRate],
        domFreq,
        actualFs
    };
}

// ============================================================
// ML INFERENCE — predict_proba
//
// pklConverter.py JSON format:
//   model = [ tree, ... ]
//   tree  = [ { id, feature, threshold, left, right, value }, ... ]
//   value = [[n_class0, n_class1]]  (shape: 1 × n_classes)
//   feature == -2 → leaf node (sklearn sentinel)
//
// sklearn predict_proba for RandomForest:
//   per tree: traverse to leaf → value[0] / sum(value[0]) → prob per class
//   final: average across all trees
//
// tremor_index = 1  (labels: 0=normal, 1=tremor; sklearn sorts → classes_=[0,1])
// ============================================================
const TREMOR_IDX = 1;

function predictProba(featureArray) {
    if (!model || !model.length) return 0;
    let totalTremorProb = 0;

    for (const nodes of model) {
        let id = 0;
        while (nodes[id].feature !== -2) {
            const { feature, threshold, left, right } = nodes[id];
            id = featureArray[feature] <= threshold ? left : right;
        }
        const counts = nodes[id].value[0];
        const total  = counts.reduce((a, b) => a + b, 0);
        totalTremorProb += total > 0 ? counts[TREMOR_IDX] / total : 0;
    }

    return totalTremorProb / model.length;
}

// ============================================================
// SIGMOID GATE
// Python: freq_weight = 1 / (1 + np.exp(-1.0 * (hz - ft)))
//         severity_pct = raw_severity * freq_weight * 100
// ============================================================
function sigmoidGate(hz, ft) {
    return 1.0 / (1.0 + Math.exp(-1.0 * (hz - ft)));
}

// ============================================================
// IMU NOTIFICATION HANDLER
// Mirrors notification_handler() in tremorpredictorv4.py
// ============================================================
function handleIMU(event, side) {
    const raw   = new TextDecoder().decode(event.target.value).trim();
    const parts = raw.split(',').map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return;

    const s   = sideState[side];
    const now = Date.now();
    const xyz = [parts[0], parts[1], parts[2]];

    // ---- CALIBRATION PHASE ----
    // Python: self.bias = np.mean([row[:3] for row in self.buffer], axis=0)
    if (!s.calibrated) {
        s.calibBuf.push(xyz);
        const pct = Math.round((s.calibBuf.length / CALIB_SAMPLES) * 100);
        setConnStatus(side, `Calibrating… ${pct}%`, '#ff9500');

        if (s.calibBuf.length >= CALIB_SAMPLES) {
            s.bias = [0, 1, 2].map(
                i => s.calibBuf.reduce((sum, r) => sum + r[i], 0) / CALIB_SAMPLES
            );
            s.calibrated = true;
            s.calibBuf   = [];
            setConnStatus(side, 'Connected ✓', '#34c759');
            setCalibBadge(side, true);
            addDiagLog(side, `Calibrated. Bias=[${s.bias.map(v=>v.toFixed(3)).join(',')}]`);
        }
        return;
    }

    // ---- BUFFER RAW (un-corrected) VALUES ----
    // Python: self.buffer.append(raw_vals + [t])  ← raw, bias applied in extract_features
    s.rawBuffer.push(xyz);
    s.timestamps.push(now);

    // ---- WINDOWED INFERENCE ----
    if (s.rawBuffer.length >= WINDOW_SIZE) {
        const winXYZ  = s.rawBuffer.slice(0, WINDOW_SIZE);
        const winTime = s.timestamps.slice(0, WINDOW_SIZE);

        const { featureArray, domFreq, actualFs } = extractFeatures(winXYZ, s.bias, winTime);

        // raw probability (0-1) from Random Forest predict_proba
        const rawProb    = predictProba(featureArray);

        // sigmoid frequency gate
        const freqWeight = sigmoidGate(domFreq, freqThreshold);

        // Python: severity_pct = raw_severity * freq_weight * 100
        const severity   = rawProb * freqWeight * 100;

        if (severity > s.peakSeverity) s.peakSeverity = severity;
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

        // Slide window: self.buffer = self.buffer[STEP_SIZE:]
        s.rawBuffer  = s.rawBuffer.slice(STEP_SIZE);
        s.timestamps = s.timestamps.slice(STEP_SIZE);
    }
}

// ============================================================
// BLUETOOTH CONNECTION
//
// Python uses BleakClient(MAC_ADDRESS) — Web BLE cannot connect by MAC.
// Strategy (in order):
//   1. If device name is set in Settings → filter by name (most reliable)
//   2. Else → filter by service UUID (Arduino must advertise it in packet)
//   3. Fallback "Scan All" button → acceptAllDevices=true (shows all BLE nearby)
// ============================================================
async function connectBluetooth(side, forceAll = false) {
    const cfg = DEVICEMAP[side];
    const btn = document.getElementById(`${side}-conn-btn`);

    try {
        btn.disabled = true;
        setConnStatus(side, 'Opening scanner…', '#ff9500');

        let opts;
        if (forceAll) {
            opts = { acceptAllDevices: true, optionalServices: [cfg.service] };
            addDiagLog(side, 'Scan mode: ALL DEVICES');
        } else if (deviceNames[side]) {
            opts = { filters: [{ name: deviceNames[side] }], optionalServices: [cfg.service] };
            addDiagLog(side, `Scan mode: name="${deviceNames[side]}"`);
        } else {
            opts = { filters: [{ services: [cfg.service] }], optionalServices: [cfg.service] };
            addDiagLog(side, `Scan mode: service UUID filter`);
        }

        const device = await navigator.bluetooth.requestDevice(opts);
        addDiagLog(side, `Found: "${device.name || '(unnamed)'}"`);
        setConnStatus(side, 'Connecting…', '#ff9500');

        const server  = await device.gatt.connect();
        addDiagLog(side, 'GATT connected');

        const service = await server.getPrimaryService(cfg.service);
        addDiagLog(side, 'Service OK');

        // Cache IMU characteristic
        const imuChar = await service.getCharacteristic(cfg.char);
        addDiagLog(side, 'IMU char OK');

        // Cache motor characteristic (non-fatal if missing)
        let motorChar = null;
        if (cfg.motor) {
            try {
                motorChar = await service.getCharacteristic(cfg.motor);
                addDiagLog(side, 'Motor char OK');
            } catch (e) {
                addDiagLog(side, `Motor char not found (non-fatal)`);
            }
        }

        const s      = sideState[side];
        s.device     = device;
        s.gattServer = server;
        s.service    = service;
        s.imuChar    = imuChar;
        s.motorChar  = motorChar;
        s.connected  = true;

        setConnStatus(side, 'Calibrating…', '#ff9500');
        btn.textContent = 'Disconnect';
        btn.disabled    = false;
        btn.onclick     = () => disconnectBluetooth(side);

        await imuChar.startNotifications();
        imuChar.addEventListener('characteristicvaluechanged', e => handleIMU(e, side));
        addDiagLog(side, 'Notifications started ✓');

        device.addEventListener('gattserverdisconnected', () => onDisconnect(side));

    } catch (err) {
        const msg = err.message || String(err);
        addDiagLog(side, `ERROR: ${msg}`);
        setConnStatus(side, `Failed — ${msg.slice(0, 50)}`, '#ff3b30');
        btn.disabled    = false;
        btn.textContent = 'Connect';
        btn.onclick     = () => connectBluetooth(side);
    }
}

async function disconnectBluetooth(side) {
    const s = sideState[side];
    if (s.device?.gatt?.connected) {
        try { await s.device.gatt.disconnect(); } catch(_) {}
    }
    onDisconnect(side);
}

function onDisconnect(side) {
    const s = sideState[side];
    Object.assign(s, makeSideState());   // reset all fields

    setConnStatus(side, 'Disconnected', '#ff3b30');
    setCalibBadge(side, false);
    resetSideUI(side);
    addDiagLog(side, 'Disconnected');

    const btn = document.getElementById(`${side}-conn-btn`);
    if (btn) { btn.textContent = 'Connect'; btn.onclick = () => connectBluetooth(side); btn.disabled = false; }
}

// ============================================================
// MOTOR FEEDBACK — uses cached characteristic (no re-fetch)
// ============================================================
async function sendMotorFeedback(side, severity) {
    const s = sideState[side];
    if (!s.motorChar) return;
    try {
        await s.motorChar.writeValueWithoutResponse(new Uint8Array([Math.min(Math.floor(severity), 100)]));
    } catch (_) {}
}

// ============================================================
// CLINICAL ASSESSMENT
// ============================================================
let assessmentTimer = null;

function startAssessment() {
    if (isRecording) return;
    sessionData = [];
    isRecording = true;
    const btn  = document.getElementById('record-btn');
    const cd   = document.getElementById('countdown');
    document.getElementById('report-section').style.display = 'none';
    btn.disabled = true;
    let remaining = 10;
    cd.textContent = `${remaining}s remaining`;
    cd.style.color = '#ff9500';
    assessmentTimer = setInterval(() => {
        remaining--;
        cd.textContent = remaining > 0 ? `${remaining}s remaining` : 'Finishing…';
        if (remaining <= 0) {
            clearInterval(assessmentTimer);
            isRecording  = false;
            btn.disabled = false;
            cd.textContent = `✓ Done — ${sessionData.length} windows captured`;
            cd.style.color = '#34c759';
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
    document.getElementById('summary-text').textContent =
        `Avg: ${avg.toFixed(1)}%  |  Peak: ${max.toFixed(1)}%  |  ${lbl}`;
}

function downloadReport() {
    if (!sessionData.length) return;
    const csv = Object.keys(sessionData[0]).join(',') + '\n'
              + sessionData.map(r => Object.values(r).join(',')).join('\n');
    const a = document.createElement('a');
    a.href     = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `TremorPause_${sessionData[0]?.position ?? 'session'}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ============================================================
// DIAGNOSTICS LOG
// ============================================================
const diagLogs = { left: [], right: [] };

function addDiagLog(side, msg) {
    const ts = new Date().toISOString().slice(11, 23);
    diagLogs[side].unshift(`[${ts}] ${msg}`);
    if (diagLogs[side].length > 25) diagLogs[side].pop();
    const el = document.getElementById(`${side}-diag-log`);
    if (el) el.textContent = diagLogs[side].join('\n');
}

function toggleDiag(side) {
    const el = document.getElementById(`${side}-diag`);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// ============================================================
// UI HELPERS
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
    bar.style.width = `${Math.min(severity,100)}%`;
    bar.style.background = m.color;
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
        document.getElementById('threshold-display').textContent = `${ft.toFixed(2)} Hz`;
    }
    const ln = document.getElementById('left-name-input').value.trim();
    const rn = document.getElementById('right-name-input').value.trim();
    deviceNames.left  = ln;
    deviceNames.right = rn;
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
    document.getElementById('freq-input').value     = freqThreshold;
    document.getElementById('threshold-display').textContent = `${freqThreshold.toFixed(2)} Hz`;
    document.getElementById('left-name-input').value  = deviceNames.left;
    document.getElementById('right-name-input').value = deviceNames.right;
});
