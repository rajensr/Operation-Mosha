/**
 * Operation-Mosha | Public Health Surveillance Client
 * Zero Emojis Enforced | Public Health Administrative Copy | Interactive Plotly Charts
 */

// Plotly Dark Theme Configuration
const PLOTLY_THEME = {
  paper_bgcolor: 'rgb(19, 17, 17)',
  plot_bgcolor: 'rgb(19, 17, 17)',
  font: { family: 'Inter', color: 'rgb(247, 255, 229)', size: 12 },
  xaxis: {
    gridcolor: 'rgba(223, 255, 82, 0.08)',
    zerolinecolor: 'rgba(223, 255, 82, 0.15)',
    tickfont: { color: 'rgb(247, 255, 229)' },
  },
  yaxis: {
    gridcolor: 'rgba(223, 255, 82, 0.08)',
    zerolinecolor: 'rgba(223, 255, 82, 0.15)',
    tickfont: { color: 'rgb(247, 255, 229)' },
  },
  colorway: ['#DFFF52', '#28a745', '#fd7e14', '#dc3545', '#11998e'],
};

// Global Store
let wardsData = [];
let currentWardId = 'CCC_Ward_01';
let currentCity = 'Chittagong';
let vulnerabilityData = [];
let currentLags = { lag1: 0, lag2: 0, lag3: 0, lag4: 0 };
let forecastDebounceTimer = null;

// Initialize SPA on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  initSurveillance();
  setupScenarioPresets();
  setupMeteorologicalInputListeners();
  setupTableSearchAndExport();
  setupProtocolSimulatorListeners();
  loadVulnerabilityMatrix();
  loadAnalytics();
  loadTelemetry();
});

// 1. Single-Page Navigation Tabs
function setupNavigation() {
  const navButtons = document.querySelectorAll('.nav-btn');
  const views = document.querySelectorAll('.spa-view');

  function switchTab(targetId) {
    navButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.target === targetId);
    });
    views.forEach(view => {
      view.classList.toggle('active', view.id === targetId);
    });
    window.location.hash = targetId;

    // Resize Plotly charts when tab becomes visible
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 120);
  }

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.target));
  });

  const initialHash = window.location.hash.replace('#', '') || 'surveillance';
  if (document.getElementById(initialHash)) {
    switchTab(initialHash);
  }
}

// 2. 1-Click Meteorological Scenario Presets
function setupScenarioPresets() {
  const presets = {
    monsoon: { rain: 110.0, temp: 31.5, hum: 88.0 },
    premonsoon: { rain: 35.0, temp: 32.0, hum: 75.0 },
    winter: { rain: 0.0, temp: 22.0, hum: 58.0 },
  };

  const chips = document.querySelectorAll('.preset-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      const presetKey = chip.dataset.preset;
      const values = presets[presetKey];
      if (values) {
        document.getElementById('num-rainfall').value = values.rain.toFixed(1);
        document.getElementById('num-temp').value = values.temp.toFixed(1);
        document.getElementById('num-humidity').value = Math.min(values.hum, 100.0).toFixed(1);
        updateRainAccumulationPreview(values.rain);
        runForecast();
      }
    });
  });
}

// 3. Real-Time Meteorological Input Listeners with Humidity Clamping
function setupMeteorologicalInputListeners() {
  const rainInput = document.getElementById('num-rainfall');
  const tempInput = document.getElementById('num-temp');
  const humInput = document.getElementById('num-humidity');
  const dateInput = document.getElementById('input-date');

  function triggerLiveRecalculation() {
    clearTimeout(forecastDebounceTimer);
    forecastDebounceTimer = setTimeout(() => {
      runForecast();
    }, 250);
  }

  // Rainfall listener
  if (rainInput) {
    rainInput.addEventListener('input', () => {
      const val = Math.max(parseFloat(rainInput.value) || 0, 0);
      updateRainAccumulationPreview(val);
      triggerLiveRecalculation();
    });
  }

  // Temperature listener
  if (tempInput) {
    tempInput.addEventListener('input', triggerLiveRecalculation);
  }

  // Humidity listener with strict [0, 100]% clamping
  if (humInput) {
    humInput.addEventListener('input', () => {
      let val = parseFloat(humInput.value);
      if (!isNaN(val)) {
        if (val > 100) {
          humInput.value = 100;
          val = 100;
        } else if (val < 0) {
          humInput.value = 0;
          val = 0;
        }
      }
      triggerLiveRecalculation();
    });
  }

  // Date change listener
  if (dateInput) {
    dateInput.addEventListener('change', () => {
      runForecast();
    });
  }
}

// Update estimated 4-week rainfall accumulation preview in real time
function updateRainAccumulationPreview(currentRain) {
  const previewEl = document.getElementById('rain-accum-preview');
  if (previewEl) {
    // Current week + prior 3 weeks lags (lag1 + lag2 + lag3)
    const estSum = (parseFloat(currentRain) || 0) + (currentLags.lag1 || 0) + (currentLags.lag2 || 0) + (currentLags.lag3 || 0);
    previewEl.textContent = `Estimated 4W Accumulation: ${estSum.toFixed(1)} mm`;
    if (estSum >= 250) {
      previewEl.style.color = '#dc3545';
      previewEl.style.fontWeight = '700';
    } else if (estSum >= 150) {
      previewEl.style.color = '#fd7e14';
      previewEl.style.fontWeight = '700';
    } else {
      previewEl.style.color = 'var(--text)';
      previewEl.style.opacity = '0.65';
      previewEl.style.fontWeight = '400';
    }
  }
}

// 4. Initialize Surveillance Form & Load Wards
async function initSurveillance() {
  try {
    const res = await fetch('/api/wards');
    const data = await res.json();
    wardsData = data.wards || [];

    const citySelect = document.getElementById('select-city');
    const wardSelect = document.getElementById('select-ward');
    const dateInput = document.getElementById('input-date');

    // Populate Cities
    citySelect.innerHTML = '';
    data.cities.forEach(city => {
      const opt = document.createElement('option');
      opt.value = city;
      opt.textContent = city;
      citySelect.appendChild(opt);
    });

    // Populate Wards
    function populateWards(selectedCity) {
      wardSelect.innerHTML = '';
      const filtered = wardsData.filter(w => w.city === selectedCity);
      filtered.forEach(w => {
        const opt = document.createElement('option');
        opt.value = w.ward_id;
        opt.textContent = w.ward_id;
        wardSelect.appendChild(opt);
      });
      if (filtered.length > 0) {
        currentWardId = filtered[0].ward_id;
        wardSelect.value = currentWardId;
        syncWardBaseline(filtered[0]);
      }
    }

    citySelect.addEventListener('change', () => {
      currentCity = citySelect.value;
      populateWards(currentCity);
      runForecast();
    });

    wardSelect.addEventListener('change', () => {
      currentWardId = wardSelect.value;
      const w = wardsData.find(item => item.ward_id === currentWardId);
      if (w) syncWardBaseline(w);
      runForecast();
    });

    dateInput.value = data.latest_date || '2025-12-29';
    populateWards(data.cities[0]);

    document.getElementById('forecast-form').addEventListener('submit', (e) => {
      e.preventDefault();
      runForecast();
    });

    // Run Initial Nowcast
    runForecast();
  } catch (err) {
    console.error('Failed to load wards data:', err);
  }
}

// Sync Form Numerical Inputs with Selected Ward's Baseline
function syncWardBaseline(w) {
  const rain = w.latest_rainfall_mm !== undefined ? w.latest_rainfall_mm : 20.0;
  const temp = w.latest_temp_c !== undefined ? w.latest_temp_c : 28.0;
  const hum = Math.min(Math.max(w.latest_humidity_pct !== undefined ? w.latest_humidity_pct : 75.0, 0), 100.0);

  document.getElementById('num-rainfall').value = rain.toFixed(1);
  document.getElementById('num-temp').value = temp.toFixed(1);
  document.getElementById('num-humidity').value = hum.toFixed(1);

  updateRainAccumulationPreview(rain);
}

// 5. Execute Outbreak Nowcast & Forecast
async function runForecast() {
  const wardId = document.getElementById('select-ward').value;
  const city = document.getElementById('select-city').value;
  const dateVal = document.getElementById('input-date').value;
  const rainfall = Math.max(parseFloat(document.getElementById('num-rainfall').value) || 0, 0);
  const temp = parseFloat(document.getElementById('num-temp').value) || 25;

  // Strict humidity clamp: max 100%
  let humidity = parseFloat(document.getElementById('num-humidity').value);
  if (isNaN(humidity)) humidity = 70;
  if (humidity > 100) {
    humidity = 100;
    document.getElementById('num-humidity').value = '100.0';
  } else if (humidity < 0) {
    humidity = 0;
    document.getElementById('num-humidity').value = '0.0';
  }

  const btn = document.getElementById('btn-run-forecast');
  if (btn) {
    btn.textContent = 'Computing Inference...';
    btn.disabled = true;
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ward_id: wardId,
        city: city,
        date: dateVal,
        rainfall_mm: rainfall,
        avg_temp_c: temp,
        relative_humidity_pct: humidity,
      }),
    });

    const pred = await res.json();
    renderPredictionResults(pred);
    renderLongitudinalTrend(wardId, pred.date);
    renderWardCrossSection();
  } catch (err) {
    console.error('Prediction failed:', err);
  } finally {
    if (btn) {
      btn.textContent = 'Compute Outbreak Nowcast';
      btn.disabled = false;
    }
  }
}

// Render Prediction KPIs & Advisory
function renderPredictionResults(pred) {
  const kpiTier = document.getElementById('kpi-tier');
  const kpiCases = document.getElementById('kpi-cases');
  const kpiProb = document.getElementById('kpi-prob');
  const kpiVuln = document.getElementById('kpi-vuln');
  const statusBadge = document.getElementById('surveillance-status-badge');

  kpiTier.textContent = pred.risk_tier;
  kpiTier.style.color = pred.risk_color;
  document.getElementById('card-tier').style.borderLeftColor = pred.risk_color;

  kpiCases.textContent = pred.expected_cases.toFixed(2);
  kpiProb.textContent = (pred.p_nonzero * 100).toFixed(1) + '%';
  kpiVuln.textContent = pred.vulnerability_index.toFixed(2);

  document.getElementById('kpi-cases-sub').textContent = pred.is_future
    ? 'Status: Future Projection'
    : 'Status: Calibrated Hurdle Inference';

  document.getElementById('kpi-density').textContent =
    'Density: ' + pred.population_density.toLocaleString() + ' /km2';

  statusBadge.textContent = pred.is_future ? 'FUTURE PROJECTION' : 'NOWCAST MONITOR';

  // Update Public Health Action Advisory Box
  const advBox = document.getElementById('advisory-box');
  const advBadge = document.getElementById('advisory-badge');
  const advLead = document.getElementById('advisory-lead');
  const advTitle = document.getElementById('advisory-title');
  const advSummary = document.getElementById('advisory-summary');
  const advList = document.getElementById('advisory-actions');

  advBox.style.borderLeftColor = pred.risk_color;
  advBadge.textContent = pred.advisory.badge;
  advBadge.style.color = pred.risk_color;
  advBadge.style.borderColor = pred.risk_color;
  advLead.textContent = pred.advisory.lead_time;
  advTitle.textContent = pred.advisory.title;
  advSummary.textContent = pred.advisory.summary;

  advList.innerHTML = '';
  pred.advisory.actions.forEach(action => {
    const li = document.createElement('li');
    li.textContent = action;
    advList.appendChild(li);
  });

  // Store lags in global store for live preview
  currentLags = {
    lag1: pred.lag1,
    lag2: pred.lag2,
    lag3: pred.lag3,
    lag4: pred.lag4,
  };

  // Meteorological Lag Attribution Cards
  document.getElementById('lag-1').textContent = pred.lag1.toFixed(1) + ' mm';
  document.getElementById('lag-2').textContent = pred.lag2.toFixed(1) + ' mm';
  document.getElementById('lag-3').textContent = pred.lag3.toFixed(1) + ' mm';
  document.getElementById('lag-4').textContent = pred.lag4.toFixed(1) + ' mm';
  document.getElementById('lag-sum4w').textContent = pred.sum_4w.toFixed(1) + ' mm';

  const rainInput = document.getElementById('num-rainfall');
  if (rainInput) {
    updateRainAccumulationPreview(rainInput.value);
  }
}

// Render Longitudinal Trend Chart
async function renderLongitudinalTrend(wardId, selectedDate) {
  try {
    const res = await fetch(`/api/history?ward_id=${wardId}`);
    const data = await res.json();
    const dates = data.records.map(r => r.date);
    const cases = data.records.map(r => r.dengue_cases);
    const rains = data.records.map(r => r.rainfall_mm);

    const traceRain = {
      x: dates,
      y: rains,
      name: 'Rainfall (mm)',
      type: 'bar',
      yaxis: 'y2',
      marker: { color: 'rgba(223, 255, 82, 0.35)' },
    };

    const traceCases = {
      x: dates,
      y: cases,
      name: 'Actual Cases',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#dc3545', width: 2.5 },
      yaxis: 'y1',
    };

    const layout = JSON.parse(JSON.stringify(PLOTLY_THEME));
    layout.height = 380;
    layout.margin = { l: 45, r: 45, t: 20, b: 35 };
    layout.hovermode = 'x unified';
    layout.legend = { orientation: 'h', y: 1.05, x: 1, xanchor: 'right' };
    layout.yaxis.title = 'Weekly Dengue Cases';
    layout.yaxis2 = {
      title: 'Rainfall (mm)',
      overlaying: 'y',
      side: 'right',
      showgrid: false,
      tickfont: { color: 'rgb(247, 255, 229)' },
    };

    if (selectedDate) {
      layout.shapes = [{
        type: 'line',
        x0: selectedDate,
        x1: selectedDate,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: 'rgb(223, 255, 82)', width: 2, dash: 'dash' },
      }];
    }

    Plotly.newPlot('chart-longitudinal', [traceRain, traceCases], layout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error('Failed to render longitudinal chart:', err);
  }
}

// Render Ward Cross-Section Comparison Bar Chart
async function renderWardCrossSection() {
  try {
    const topWards = wardsData.slice(0, 15).sort((a, b) => a.latest_cases - b.latest_cases);
    const yLabels = topWards.map(w => w.ward_id);
    const xValues = topWards.map(w => w.latest_cases);

    const trace = {
      x: xValues,
      y: yLabels,
      type: 'bar',
      orientation: 'h',
      marker: { color: '#DFFF52' },
    };

    const layout = JSON.parse(JSON.stringify(PLOTLY_THEME));
    layout.height = 380;
    layout.margin = { l: 90, r: 20, t: 20, b: 35 };
    layout.xaxis.title = 'Observed Active Cases';

    Plotly.newPlot('chart-cross-section', [trace], layout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error('Failed to render cross-section chart:', err);
  }
}

// 6. Vulnerability Matrix Tab: Search & CSV Export
function setupTableSearchAndExport() {
  const searchInput = document.getElementById('search-vuln-table');
  const exportBtn = document.getElementById('btn-export-csv');

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      const query = searchInput.value.toLowerCase().trim();
      filterVulnerabilityTable(query);
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      exportTableToCSV();
    });
  }
}

async function loadVulnerabilityMatrix() {
  try {
    const res = await fetch('/api/vulnerability');
    const data = await res.json();
    vulnerabilityData = data.wards || [];

    document.getElementById('vuln-total-wards').textContent = data.total_wards;
    document.getElementById('vuln-high-count').textContent = data.high_vulnerability_count;
    document.getElementById('vuln-avg-drainage').textContent = data.city_avg_waterlog.toFixed(2);
    document.getElementById('vuln-avg-sanitation').textContent = data.city_avg_sanitation.toFixed(2);

    renderVulnerabilityTable(vulnerabilityData);

    // Scatter Chart: Drainage Index vs Population Density
    const xDrain = vulnerabilityData.map(w => w.water_logging_risk_index);
    const yPop = vulnerabilityData.map(w => w.population_density_sqkm);
    const hoverText = vulnerabilityData.map(w => `${w.ward_id} | Score: ${w.vulnerability_score}`);

    const traceScatter = {
      x: xDrain,
      y: yPop,
      mode: 'markers',
      type: 'scatter',
      text: hoverText,
      marker: {
        size: 10,
        color: vulnerabilityData.map(w => w.vulnerability_score),
        colorscale: [[0, '#11998e'], [0.5, '#DFFF52'], [1, '#dc3545']],
        showscale: true,
        colorbar: {
          title: 'Score',
          tickfont: { color: 'rgb(247, 255, 229)' },
          titlefont: { color: 'rgb(247, 255, 229)' },
        },
      },
    };

    const sLayout = JSON.parse(JSON.stringify(PLOTLY_THEME));
    sLayout.height = 400;
    sLayout.margin = { l: 60, r: 30, t: 20, b: 40 };
    sLayout.xaxis.title = 'Water-Logging Risk Index';
    sLayout.yaxis.title = 'Population Density (/km2)';

    Plotly.newPlot('chart-vuln-scatter', [traceScatter], sLayout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error('Failed to load vulnerability matrix:', err);
  }
}

function renderVulnerabilityTable(items) {
  const tbody = document.getElementById('table-vuln-body');
  tbody.innerHTML = '';

  if (items.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; opacity: 0.6; padding: 24px;">No matching wards found.</td></tr>';
    return;
  }

  items.forEach(w => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>#${w.rank}</strong></td>
      <td><strong>${w.ward_id}</strong></td>
      <td>${w.city}</td>
      <td><span style="color: rgb(223, 255, 82); font-weight: 700;">${w.vulnerability_score.toFixed(3)}</span></td>
      <td>${w.water_logging_risk_index.toFixed(2)}</td>
      <td>${w.sanitation_score.toFixed(2)}</td>
      <td>${w.population_density_sqkm.toLocaleString()}</td>
      <td><strong>${w.latest_cases}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterVulnerabilityTable(query) {
  const filtered = vulnerabilityData.filter(w =>
    w.ward_id.toLowerCase().includes(query) ||
    w.city.toLowerCase().includes(query)
  );
  renderVulnerabilityTable(filtered);
}

function exportTableToCSV() {
  const query = (document.getElementById('search-vuln-table')?.value || '').toLowerCase().trim();
  const exportItems = query
    ? vulnerabilityData.filter(w => w.ward_id.toLowerCase().includes(query) || w.city.toLowerCase().includes(query))
    : vulnerabilityData;

  const headers = ['Rank', 'Ward_ID', 'City', 'Vulnerability_Score', 'Water_Logging_Index', 'Sanitation_Score', 'Population_Density_sqkm', 'Active_Cases'];
  const rows = exportItems.map(w => [
    w.rank,
    w.ward_id,
    w.city,
    w.vulnerability_score,
    w.water_logging_risk_index,
    w.sanitation_score,
    w.population_density_sqkm,
    w.latest_cases,
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' +
    [headers.join(','), ...rows.map(e => e.join(','))].join('\n');

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `municipal_dispatch_matrix_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// 7. Epidemiological Analytics Tab
async function loadAnalytics() {
  try {
    const res = await fetch('/api/analytics');
    const data = await res.json();

    const mNames = data.monthly_seasonality.map(m => m.month_name);
    const mCases = data.monthly_seasonality.map(m => m.dengue_cases);
    const mRain = data.monthly_seasonality.map(m => m.rainfall_mm);

    const traceRain = {
      x: mNames,
      y: mRain,
      name: 'Rainfall (mm)',
      type: 'bar',
      yaxis: 'y2',
      marker: { color: 'rgba(223, 255, 82, 0.4)' },
    };

    const traceCases = {
      x: mNames,
      y: mCases,
      name: 'Average Cases / Ward',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#dc3545', width: 3 },
      marker: { size: 8, color: '#dc3545' },
      yaxis: 'y1',
    };

    const mLayout = JSON.parse(JSON.stringify(PLOTLY_THEME));
    mLayout.height = 390;
    mLayout.margin = { l: 45, r: 45, t: 20, b: 35 };
    mLayout.legend = { orientation: 'h', y: 1.05, x: 1, xanchor: 'right' };
    mLayout.yaxis.title = 'Average Dengue Cases';
    mLayout.yaxis2 = {
      title: 'Average Rainfall (mm)',
      overlaying: 'y',
      side: 'right',
      showgrid: false,
      tickfont: { color: 'rgb(247, 255, 229)' },
    };

    Plotly.newPlot('chart-monthly-seasonality', [traceRain, traceCases], mLayout, { responsive: true, displayModeBar: false });

    // Pearson Correlation Bar Chart
    const corrSorted = data.correlations.sort((a, b) => a.correlation - b.correlation);
    const traceCorr = {
      x: corrSorted.map(c => c.correlation),
      y: corrSorted.map(c => c.feature),
      type: 'bar',
      orientation: 'h',
      marker: {
        color: corrSorted.map(c => c.correlation >= 0.6 ? '#dc3545' : (c.correlation >= 0.3 ? '#fd7e14' : '#DFFF52')),
      },
    };

    const cLayout = JSON.parse(JSON.stringify(PLOTLY_THEME));
    cLayout.height = 390;
    cLayout.margin = { l: 140, r: 20, t: 20, b: 35 };
    cLayout.xaxis.title = 'Pearson Correlation with Dengue Cases';

    Plotly.newPlot('chart-correlations', [traceCorr], cLayout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error('Failed to load analytics:', err);
  }
}

// 8. Action Protocols Threshold Simulator with Real-Time Listeners
function setupProtocolSimulatorListeners() {
  const rain4w = document.getElementById('sim-rain-4w');
  const temp = document.getElementById('sim-temp');
  const hum = document.getElementById('sim-hum');

  if (hum) {
    hum.addEventListener('input', () => {
      let val = parseFloat(hum.value);
      if (val > 100) hum.value = 100;
      if (val < 0) hum.value = 0;
      window.runThresholdSimulator();
    });
  }

  if (rain4w) rain4w.addEventListener('input', window.runThresholdSimulator);
  if (temp) temp.addEventListener('input', window.runThresholdSimulator);
}

window.runThresholdSimulator = function () {
  const rain = parseFloat(document.getElementById('sim-rain-4w').value) || 0;
  const temp = parseFloat(document.getElementById('sim-temp').value) || 25;

  let hum = parseFloat(document.getElementById('sim-hum').value);
  if (isNaN(hum)) hum = 70;
  if (hum > 100) {
    hum = 100;
    document.getElementById('sim-hum').value = '100.0';
  } else if (hum < 0) {
    hum = 0;
    document.getElementById('sim-hum').value = '0.0';
  }

  const isHighRain = rain >= 250;
  const isIdealTemp = temp >= 26 && temp <= 33;
  const isHighHum = hum >= 80;

  const resultBox = document.getElementById('sim-result-box');
  const resultTitle = document.getElementById('sim-result-title');
  const resultText = document.getElementById('sim-result-text');

  if (isHighRain && isIdealTemp && isHighHum) {
    resultBox.style.borderLeftColor = '#dc3545';
    resultTitle.textContent = 'CRITICAL OUTBREAK ALERT';
    resultTitle.style.color = '#dc3545';
    resultText.textContent = `TRIPLE-THRESHOLD EXCEEDED: High cumulative rainfall (${rain} mm) with optimal temperature (${temp}°C) and ${hum}% humidity. Severe probability of explosive vector emergence in 14-21 days. Mobilize vehicle-mounted thermal fogging within 48 hours.`;
  } else if ((rain >= 180 && isIdealTemp) || (isHighHum && rain >= 150)) {
    resultBox.style.borderLeftColor = '#fd7e14';
    resultTitle.textContent = 'HIGH RISK ADVISORY';
    resultTitle.style.color = '#fd7e14';
    resultText.textContent = `PRE-OUTBREAK VECTOR AMPLIFICATION: Extensive aquatic breeding reservoirs detected (${rain} mm rain, ${hum}% humidity). Public health teams have a 2 to 3 week lead time to apply biological larvicide (Bti) to all open drains before adults mature.`;
  } else if (rain >= 80 || (hum >= 70 && temp >= 22)) {
    resultBox.style.borderLeftColor = '#ffc107';
    resultTitle.textContent = 'ELEVATED RISK WATCH';
    resultTitle.style.color = '#ffc107';
    resultText.textContent = 'MICROCLIMATE PRIMING: Weather indicators support moderate mosquito reproduction. Inspect vacant construction plots, tire shops, and clear surface drainage.';
  } else {
    resultBox.style.borderLeftColor = '#28a745';
    resultTitle.textContent = 'ROUTINE BASELINE';
    resultTitle.style.color = '#28a745';
    resultText.textContent = 'BASELINE VECTOR SUPPRESSION: Meteorological indicators suppress vector amplification and viral replication. Maintain routine municipal sanitation.';
  }
};

// 9. Model Telemetry Tab
async function loadTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    const data = await res.json();
    if (!data.splits) return;

    const testSplit = data.splits.test;
    const hurdle = testSplit.hurdle_model;
    const base = testSplit.baseline_all_zero;

    document.getElementById('telemetry-samples').textContent = testSplit.sample_count.toLocaleString();
    document.getElementById('telemetry-prevalence').textContent = testSplit.positive_prevalence_pct.toFixed(2) + '%';
    document.getElementById('telemetry-mae').textContent = hurdle.mae.toFixed(4) + ' (vs. ' + base.mae.toFixed(4) + ')';
    document.getElementById('telemetry-recall').textContent = (hurdle.recall_positive * 100).toFixed(1) + '%';
    document.getElementById('telemetry-f1').textContent = hurdle.f1_positive.toFixed(4);
  } catch (err) {
    console.error('Failed to load telemetry:', err);
  }
}
