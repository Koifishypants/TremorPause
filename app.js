let model = null;
let assessmentMode = false;
let recordings = { Resting: [], Postural: [], Kinetic: [], Intention: [] };

// CONFIG FOR THE TWO DEVICES
const configs = {
    left: { name: null, service: '12345678-1234-5678-1234-56789abcdef0', char: 'abcdef01-1234-5678-1234-56789abcdef0', motor: 'abcdef02-1234-5678-1234-56789abcdef0' },
    right: { name: 'TREMOR-PRO-X1', service: '19b10000-e8f2-537e-4f6c-d104768a1214', char: '19b10001-e8f2-537e-4f6c-d104768a1214', motor: null }
};

// 1. LOAD MODEL
fetch('model.json').then(r => r.json()).then(data => { model = data; console.log("Model Loaded"); });

// 2. BLUETOOTH CONNECTION
async function connect(hand) {
    const config = configs[hand];
    try {
        const device = await navigator.bluetooth.requestDevice({
            filters: config.name ? [{ name: config.name }] : [{ services: [config.service] }],
            optionalServices: [config.service]
        });
        const server = await device.gatt.connect();
        const service = await server.getPrimaryService(config.service);
        const char = await service.getCharacteristic(config.char);
        
        char.startNotifications();
        char.addEventListener('characteristicvaluechanged', (e) => handleData(e, hand, service));
        document.getElementById(`${hand}-status`).innerText = "Connected";
    } catch (err) { alert("Connect failed: " + err); }
}

// 3. SIGNAL PROCESSING (Simplified features)
let buffers = { left: [], right: [] };
async function handleData(event, hand, service) {
    let raw = new TextDecoder().decode(event.target.value);
    let vals = raw.split(',').map(Number);
    if(vals.length < 3) return;

    let mag = Math.sqrt(vals[0]**2 + vals[1]**2 + vals[2]**2);
    buffers[hand].push(mag);

    if (buffers[hand].length >= 128) {
        let window = buffers[hand].slice(-128);
        let features = extractFeatures(window);
        let severity = runInference(features);

        updateUI(hand, severity);

        if (assessmentMode) {
            let pos = document.getElementById('position-select').value;
            recordings[pos].push({ s: severity, f: features[4] });
        } else {
            // SWAPPABLE MOTOR LOGIC
            sendMotorCommand(hand, service, severity);
        }
        buffers[hand] = buffers[hand].slice(32); // Step size
    }
}

function extractFeatures(w) {
    let mean = w.reduce((a,b)=>a+b)/w.length;
    let max = Math.max(...w);
    let energy = w.reduce((a,b)=>a + b*b, 0);
    let std = Math.sqrt(w.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / w.length);
    // Note: FFT implementation omitted for brevity, using mock dom_freq (5.5Hz)
    return [mean, std, max, energy, 5.5, 0.1]; 
}

// 4. ML INFERENCE (Random Forest Interpreter)
function runInference(features) {
    let totalProb = 0;
    model.forEach(tree => {
        let node = 0;
        while (tree.feature[node] !== -2) {
            let fIdx = tree.feature[node];
            node = (features[fIdx] <= tree.threshold[node]) ? tree.children_left[node] : tree.children_right[node];
        }
        // Prob of Tremor (class 1)
        let val = tree.values[node][0];
        totalProb += (val[1] > val[0] ? 1 : 0);
    });
    return (totalProb / model.length) * 100;
}

// 5. SWAPPABLE MOTOR LOGIC (Future Drone Motor logic goes here)
async function sendMotorCommand(hand, service, severity) {
    const config = configs[hand];
    if (!config.motor) return;
    try {
        const motorChar = await service.getCharacteristic(config.motor);
        let val = Math.floor(Math.min(severity, 100));
        await motorChar.writeValueWithoutResponse(new Uint8Array([val]));
    } catch(e) {}
}

function updateUI(hand, sev) {
    let fill = document.getElementById(`${hand}-fill`);
    fill.style.width = sev + "%";
    fill.style.background = sev > 70 ? "#ff3b30" : (sev > 30 ? "#ff9500" : "#4cd964");
}

function toggleAssessment() {
    assessmentMode = !assessmentMode;
    let btn = document.getElementById('record-btn');
    btn.innerText = assessmentMode ? "STOP RECORDING" : "START RECORDING";
    btn.className = assessmentMode ? "btn-red" : "";
}

function generateReport() {
    let html = "<h3>Patient Severity Summary</h3>";
    for (let pos in recordings) {
        let data = recordings[pos];
        if (data.length === 0) continue;
        let avg = data.reduce((a,b)=>a+b.s,0)/data.length;
        let peak = Math.max(...data.map(d=>d.s));
        html += `<div class='card'><b>${pos}</b><br>Avg: ${avg.toFixed(1)}% | Peak: ${peak.toFixed(1)}%</div>`;
    }
    document.getElementById('report-content').innerHTML = html;
    document.getElementById('report-modal').style.display = 'block';
}