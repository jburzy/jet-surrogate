/* PRISM (Physics Reinterpretation with Intelligent Surrogate Models). Shared front-end script.
   Every page is driven by the JSON API under /api/. When the API cannot be
   reached (for example when previewing with python3 -m http.server) the
   script falls back to the sample payloads in fixtures/. */

/* ---------- Config ---------- */
window.JS_FIXTURES = window.JS_FIXTURES || false; /* true: always use fixtures/ */
window.JS_API_BASE = window.JS_API_BASE || '/api';
window.JS_POLL_MS = window.JS_POLL_MS || 3000;

(function () {
  'use strict';

  const API = window.JS_API_BASE;
  const FIXTURE_DIR = 'fixtures/';
  const STATUS_TAG = { queued: 'tag--gray', running: 'tag--blue', done: 'tag--green', failed: 'tag--red' };
  const STATUS_TEXT = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed' };
  const EXPERIMENT_TAG = { ATLAS: 'tag--blue', CMS: 'tag--magenta', LHCb: 'tag--purple' };
  const STATUS_LABEL_TAG = { example: 'tag--warm', preserved: 'tag--green' };

  /* ---------- Small helpers ---------- */

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function el(html) {
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function param(name) {
    return new URLSearchParams(window.location.search).get(name);
  }

  function safeUrl(u) {
    /* Only allow http(s), mailto and relative links in API-provided hrefs */
    const s = String(u || '');
    if (/^(https?:|mailto:|\/|\.\/|[a-z0-9_-]+\.html)/i.test(s)) return s;
    return '#';
  }

  const fmt = {
    bytes(n) {
      if (!(n >= 0)) return '';
      const units = ['B', 'kB', 'MB', 'GB', 'TB'];
      let i = 0;
      let v = n;
      while (v >= 1000 && i < units.length - 1) { v /= 1000; i += 1; }
      return (i === 0 ? v : v.toFixed(v < 10 ? 2 : 1)) + ' ' + units[i];
    },
    int(n) {
      return n == null ? '' : Number(n).toLocaleString();
    },
    date(unix) {
      if (!unix) return '';
      try {
        return new Date(unix * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
      } catch (e) {
        return new Date(unix * 1000).toString();
      }
    },
    isoDate(unix) {
      return unix ? new Date(unix * 1000).toISOString() : '';
    },
    duration(a, b) {
      if (!a || !b || b < a) return '';
      const s = Math.round(b - a);
      if (s < 60) return s + ' s';
      const m = Math.floor(s / 60);
      if (m < 60) return m + ' min ' + (s % 60) + ' s';
      return Math.floor(m / 60) + ' h ' + (m % 60) + ' min';
    },
    effDigits(err) {
      if (!(err > 0)) return 4;
      return Math.min(5, Math.max(3, Math.ceil(-Math.log10(err)) + 1));
    },
    eff(v, err) {
      if (v == null) return '';
      const d = fmt.effDigits(err);
      return Number(v).toFixed(d) + (err != null ? ' ± ' + Number(err).toFixed(d) : '');
    },
    pct(v, err) {
      if (v == null) return '';
      const d = Math.max(0, fmt.effDigits(err) - 2);
      return (100 * v).toFixed(d) + (err != null ? ' ± ' + (100 * err).toFixed(d) : '') + ' %';
    },
    prob(v) {
      return v == null ? '' : Number(v).toFixed(3);
    }
  };

  /* ---------- API access with fixture fallback ---------- */

  let fixtureMode = null; /* null: unknown, true/false once probed */
  let probePromise = null;
  const fixtureJobPolls = {};

  function probe() {
    if (window.JS_FIXTURES) { fixtureMode = true; return Promise.resolve(true); }
    if (probePromise) return probePromise;
    probePromise = fetch(API + '/info', { headers: { Accept: 'application/json' } })
      .then(r => {
        const ct = r.headers.get('content-type') || '';
        if (!r.ok || ct.indexOf('json') < 0) throw new Error('no api');
        return r.json();
      })
      .then(info => { fixtureMode = false; window.__apiInfo = info; return false; })
      .catch(() => { fixtureMode = true; return true; });
    return probePromise;
  }

  function fixtureJson(file) {
    return fetch(FIXTURE_DIR + file).then(r => {
      if (!r.ok) throw new Error('Fixture ' + file + ' not found');
      return r.json();
    });
  }

  function fixtureGet(path) {
    const m = path.match(/^\/([a-z]+)(?:\/([^/?]+))?(?:\/([^?]+))?(\?.*)?$/);
    const kind = m && m[1];
    const id = m && m[2];
    if (kind === 'info') {
      return Promise.resolve({ name: 'PRISM', tagline: 'Physics Reinterpretation with Intelligent Surrogate Models', version: 'preview', url: 'https://prism.web.cern.ch', repo_url: 'https://github.com/jburzy/jet-surrogate', max_upload_mb: 2000, n_analyses: 1 });
    }
    if (kind === 'analyses' && !id) return fixtureJson('analyses.json');
    if (kind === 'analyses' && id) {
      return fixtureJson('analysis.json').then(a => {
        if (a.id !== id) throw apiError(404, 'Unknown analysis "' + id + '" (fixture mode only knows ' + a.id + ')');
        return a;
      });
    }
    if (kind === 'jobs' && !id) {
      return fixtureJson('job.json').then(j => {
        const running = Object.assign({}, j, { id: 'running00001', status: 'running', label: 'm_pid = 2 GeV, ctau = 1 mm', progress: '11200/20000 events', result: null, finished: null, created: j.created + 600, started: j.created + 610 });
        const failed = Object.assign({}, j, { id: 'failed000001', status: 'failed', label: '', progress: null, result: null, error: 'could not parse HepMC header', created: j.created - 900, started: j.created - 890, finished: j.created - 880 });
        return [running, j, failed];
      });
    }
    if (kind === 'jobs' && id) {
      return fixtureJson('job.json').then(j => {
        if (id.indexOf('fixture') === 0) {
          /* Simulated job created by the submit page in fixture mode */
          const n = (fixtureJobPolls[id] = (fixtureJobPolls[id] || 0) + 1);
          const base = Object.assign({}, j, { id: id, created: Math.floor(Date.now() / 1000) - 3 * n });
          if (n <= 1) return Object.assign(base, { status: 'queued', progress: null, result: null, started: null, finished: null });
          if (n <= 3) return Object.assign(base, { status: 'running', progress: (n - 1) * 7000 + '/20000 events', result: null, finished: null, started: base.created + 2 });
          return Object.assign(base, { started: base.created + 2, finished: base.created + 3 * n });
        }
        if (id === 'running00001') return Object.assign({}, j, { id: id, status: 'running', progress: '11200/20000 events', result: null, finished: null });
        if (id === 'failed000001') return Object.assign({}, j, { id: id, status: 'failed', progress: null, result: null, error: 'could not parse HepMC header' });
        if (id !== j.id) throw apiError(404, 'Unknown job "' + id + '"');
        return j;
      });
    }
    return Promise.reject(apiError(404, 'No fixture for ' + path));
  }

  function apiError(status, detail) {
    const e = new Error(detail || ('Request failed (' + status + ')'));
    e.status = status;
    return e;
  }

  function apiGet(path) {
    return probe().then(fx => {
      if (fx) return fixtureGet(path);
      return fetch(API + path, { headers: { Accept: 'application/json' } }).then(r => {
        return r.json().catch(() => ({})).then(body => {
          if (!r.ok) throw apiError(r.status, body && body.detail);
          return body;
        });
      });
    });
  }

  /* Multipart upload with progress. Resolves with the parsed JSON body. */
  function apiUpload(path, formData, onProgress) {
    return probe().then(fx => {
      if (fx) {
        return new Promise(resolve => {
          let p = 0;
          const total = 1e9;
          const t = setInterval(() => {
            p = Math.min(1, p + 0.12);
            if (onProgress) onProgress(p * total, total);
            if (p >= 1) { clearInterval(t); resolve({ id: 'fixture' + Date.now().toString(36), status: 'queued' }); }
          }, 250);
        });
      }
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', API + path);
        xhr.setRequestHeader('Accept', 'application/json');
        xhr.upload.addEventListener('progress', e => {
          if (onProgress) onProgress(e.loaded, e.lengthComputable ? e.total : 0);
        });
        xhr.addEventListener('load', () => {
          let body = {};
          try { body = JSON.parse(xhr.responseText || '{}'); } catch (e) { body = {}; }
          if (xhr.status >= 200 && xhr.status < 300) resolve(body);
          else reject(apiError(xhr.status, body.detail || ('Upload failed (HTTP ' + xhr.status + ')')));
        });
        xhr.addEventListener('error', () => reject(apiError(0, 'Network error during upload. Check your connection and try again.')));
        xhr.addEventListener('abort', () => reject(apiError(0, 'Upload cancelled.')));
        xhr.send(formData);
      });
    });
  }

  function jobUrl(id, suffix) {
    return API + '/jobs/' + encodeURIComponent(id) + (suffix || '');
  }

  function figureUrl(analysisId, file) {
    return API + '/analyses/' + encodeURIComponent(analysisId) + '/figures/' + encodeURIComponent(file);
  }

  /* ---------- Rendering helpers ---------- */

  const ICONS = {
    arrow: '<svg class="card__arrow" viewBox="0 0 32 32" aria-hidden="true"><path d="M18 6l-1.43 1.393L24.15 15H4v2h20.15l-7.58 7.573L18 26l10-10L18 6z"/></svg>',
    error: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2zm-1.125 5h2.25v12h-2.25zM16 25a1.5 1.5 0 1 1 1.5-1.5A1.5 1.5 0 0 1 16 25z"/></svg>',
    success: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2zm-2 19.59l-5-5L10.59 15 14 18.41 21.41 11l1.596 1.586z"/></svg>',
    info: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2zm0 6a1.5 1.5 0 1 1-1.5 1.5A1.5 1.5 0 0 1 16 8zm4 16.125h-8v-2.25h2.875v-5.75H13v-2.25h4.125v8H20z"/></svg>',
    warning: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2zm-1.125 5h2.25v12h-2.25zM16 25a1.5 1.5 0 1 1 1.5-1.5A1.5 1.5 0 0 1 16 25z"/></svg>',
    download: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M26 24v4H6v-4H4v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4zm0-10l-1.41-1.41L17 20.17V2h-2v18.17l-7.59-7.58L6 14l10 10 10-10z"/></svg>',
    doc: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M25.7 9.3l-7-7A.9.9 0 0 0 18 2H8a2 2 0 0 0-2 2v24a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V10a.9.9 0 0 0-.3-.7zM18 4.4l5.6 5.6H18zM24 28H8V4h8v6a2 2 0 0 0 2 2h6z"/><path d="M10 22h12v2H10zm0-6h12v2H10z"/></svg>',
    link: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M29.25 6.76a6 6 0 0 0-8.5 0l1.42 1.42a4 4 0 1 1 5.67 5.67l-8 8a4 4 0 1 1-5.67-5.66l1.41-1.42-1.41-1.42-1.42 1.42a6 6 0 0 0 0 8.5A6 6 0 0 0 17 25a6 6 0 0 0 4.27-1.76l8-8a6 6 0 0 0-.02-8.48z"/><path d="M4.19 24.82a4 4 0 0 1 0-5.67l8-8a4 4 0 0 1 5.67 0A3.94 3.94 0 0 1 19 14a4 4 0 0 1-1.17 2.85L15.71 19l1.42 1.42 2.12-2.12a6 6 0 0 0-8.51-8.51l-8 8a6 6 0 0 0 0 8.51A6 6 0 0 0 7 28a6.07 6.07 0 0 0 4.28-1.76l-1.42-1.42a4 4 0 0 1-5.67 0z"/></svg>'
  };

  function tag(text, cls) {
    return '<span class="tag ' + (cls || '') + '">' + esc(text) + '</span>';
  }

  function experimentTag(a) {
    return a.experiment ? tag(a.experiment, EXPERIMENT_TAG[a.experiment] || 'tag--teal') : '';
  }

  function statusTag(a) {
    return a.status ? tag(a.status, STATUS_LABEL_TAG[a.status] || 'tag--gray') : '';
  }

  function jobStatusTag(status) {
    return tag(STATUS_TEXT[status] || status, STATUS_TAG[status] || 'tag--gray');
  }

  function notification(kind, title, text, extraHtml) {
    return el('<div class="notification notification--' + esc(kind) + '" role="' + (kind === 'error' ? 'alert' : 'status') + '">' +
      (ICONS[kind] || ICONS.info) +
      '<div><strong>' + esc(title) + '</strong>' + (text ? ' ' + esc(text) : '') + (extraHtml || '') + '</div></div>');
  }

  function analysisCard(a) {
    const href = 'analysis.html?id=' + encodeURIComponent(a.id);
    return el('<article class="card">' +
      '<div class="tag-row">' + experimentTag(a) + statusTag(a) + '</div>' +
      '<h3><a href="' + esc(href) + '">' + esc(a.title) + '</a></h3>' +
      '<p>' + esc(a.short || '') + '</p>' +
      '<div class="card__meta"><span>Version ' + esc(a.version || '') + (a.updated ? ' · updated ' + esc(a.updated) : '') + '</span>' + ICONS.arrow + '</div>' +
      '</article>');
  }

  function fillCards(container, list, emptyText) {
    container.innerHTML = '';
    if (!list || !list.length) {
      container.appendChild(el('<div class="empty">' + esc(emptyText || 'No analyses are published yet.') + '</div>'));
      return;
    }
    list.forEach(a => container.appendChild(analysisCard(a)));
  }

  function showError(container, err, what) {
    container.innerHTML = '';
    container.appendChild(notification('error', what || 'Something went wrong.', err && err.message ? err.message : String(err)));
  }

  /* ---------- Histogram (inline SVG, Carbon tokens) ---------- */

  function niceStep(max, n) {
    const raw = max / n;
    const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const f = raw / p;
    const m = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
    return m * p;
  }

  function histogramSvg(hist, opts) {
    opts = opts || {};
    if (!hist || !hist.counts || !hist.edges || hist.edges.length !== hist.counts.length + 1) {
      return '<p class="muted small">No histogram available.</p>';
    }
    const W = 480, H = 260, L = 48, R = 12, T = 24, B = 40;
    const counts = hist.counts.map(Number);
    const edges = hist.edges.map(Number);
    const x0 = edges[0], x1 = edges[edges.length - 1];
    const maxC = Math.max(1, Math.max.apply(null, counts));
    const step = niceStep(maxC, 4);
    const yMax = Math.ceil(maxC / step) * step;
    const sx = v => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const sy = c => T + (1 - c / yMax) * (H - T - B);
    const total = counts.reduce((a, b) => a + b, 0);
    let out = '<svg class="hist" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
      esc('Histogram of the per-event signal-region probability for ' + fmt.int(total) + ' events, ' + counts.length + ' bins from ' + x0 + ' to ' + x1) + '">';
    out += '<text class="hist__title" x="' + L + '" y="14">' + esc(opts.title || 'Per-event signal-region probability') + '</text>';
    for (let c = 0; c <= yMax + 1e-9; c += step) {
      const y = sy(c);
      out += '<line class="hist__grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>';
      out += '<text class="hist__text" x="' + (L - 6) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' + fmt.int(c) + '</text>';
    }
    counts.forEach((c, i) => {
      const xa = sx(edges[i]), xb = sx(edges[i + 1]);
      const y = sy(c);
      const w = Math.max(0.5, xb - xa - 1);
      out += '<rect class="hist__bar" x="' + xa.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + w.toFixed(1) + '" height="' + (sy(0) - y).toFixed(1) + '">' +
        '<title>' + esc(edges[i].toFixed(2) + ' to ' + edges[i + 1].toFixed(2) + ': ' + fmt.int(c) + ' events') + '</title></rect>';
    });
    out += '<line class="hist__axis" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(0) + '" y2="' + sy(0) + '"/>';
    out += '<line class="hist__axis" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + sy(0) + '"/>';
    const nx = 5;
    for (let k = 0; k <= nx; k++) {
      const v = x0 + (k / nx) * (x1 - x0);
      out += '<line class="hist__axis" x1="' + sx(v) + '" x2="' + sx(v) + '" y1="' + sy(0) + '" y2="' + (sy(0) + 4) + '"/>';
      out += '<text class="hist__text" x="' + sx(v) + '" y="' + (sy(0) + 16) + '" text-anchor="middle">' + v.toFixed(1) + '</text>';
    }
    out += '<text class="hist__text" x="' + ((L + W - R) / 2) + '" y="' + (H - 6) + '" text-anchor="middle">probability that the event enters the signal region</text>';
    out += '<text class="hist__text" transform="translate(12 ' + ((T + sy(0)) / 2) + ') rotate(-90)" text-anchor="middle">events</text>';
    out += '</svg>';
    return out;
  }

  /* ---------- Job status panel and result card ---------- */

  function statusPanel(job) {
    const permalink = 'jobs.html?id=' + encodeURIComponent(job.id);
    let text = '';
    if (job.status === 'queued') text = 'Waiting for a worker. Jobs run one after another, so this can take a few minutes when the service is busy.';
    else if (job.status === 'running') text = 'Running the surrogate' + (job.progress ? ': ' + job.progress : '') + '. You can leave this page and come back with the permalink below.';
    else if (job.status === 'failed') text = job.error || 'The job failed. Check the log for details.';
    else text = 'Finished' + (job.finished && job.started ? ' in ' + fmt.duration(job.started, job.finished) : '') + '.';
    const busy = job.status === 'queued' || job.status === 'running';
    return el('<section class="status-panel" aria-live="polite">' +
      '<div class="status-panel__head"><h3>Job <span class="job-id">' + esc(job.id) + '</span></h3>' + jobStatusTag(job.status) +
      (job.label ? '<span class="muted small">' + esc(job.label) + '</span>' : '') + '</div>' +
      '<p class="status-panel__text">' + esc(text) + '</p>' +
      (busy ? '<div class="progress progress--indeterminate" role="progressbar" aria-label="Job progress" aria-valuetext="' + esc(job.progress || job.status) + '"><div class="progress__track"><div class="progress__bar"></div></div></div>' : '') +
      '<p class="small" style="margin:0.75rem 0 0"><a href="' + esc(permalink) + '">' + ICONS.link + ' Permalink</a> · <a href="' + esc(jobUrl(job.id, '/log')) + '" target="_blank" rel="noopener">Log</a></p>' +
      '</section>');
  }

  function resultCard(job, analysis) {
    const r = job.result || {};
    const a = analysis || {};
    const permalink = 'jobs.html?id=' + encodeURIComponent(job.id);
    const title = a.title || job.analysis || r.analysis || '';
    const sr = a.signal_region ? esc(a.signal_region) : 'the signal region defined by this analysis';
    const html = '<section class="result" aria-label="Result of job ' + esc(job.id) + '">' +
      '<div class="result__head">' +
      '<div class="tag-row">' + jobStatusTag(job.status) + (a.experiment ? experimentTag(a) : '') + '</div>' +
      '<h3>' + esc(title) + '</h3>' +
      '<p class="muted small" style="margin:0">Job <span class="job-id">' + esc(job.id) + '</span>' +
      (job.label ? ' · ' + esc(job.label) : '') +
      (job.source ? ' · ' + esc(job.source) : '') +
      (job.created ? ' · submitted ' + esc(fmt.date(job.created)) : '') + '</p>' +
      '</div>' +
      '<div class="result__grid">' +
      '<div class="result__cell">' +
      '<div class="stat stat--big"><span class="stat__label">Signal-region efficiency</span>' +
      '<span class="stat__value">' + esc(Number(r.sr_efficiency).toFixed(fmt.effDigits(r.sr_efficiency_err))) +
      '<span class="stat__err">± ' + esc(Number(r.sr_efficiency_err).toFixed(fmt.effDigits(r.sr_efficiency_err))) + '</span></span>' +
      '<span class="stat__pct">' + esc(fmt.pct(r.sr_efficiency, r.sr_efficiency_err)) + ' of your events</span></div>' +
      '<div class="stat-row">' +
      '<div class="stat"><span class="stat__label">Events analysed</span><span class="stat__value">' + esc(fmt.int(r.n_events)) + '</span></div>' +
      '<div class="stat"><span class="stat__label">Truth jets</span><span class="stat__value">' + esc(fmt.int(r.n_truth_jets)) + '</span></div>' +
      '<div class="stat"><span class="stat__label">Mean jet probability</span><span class="stat__value">' + esc(fmt.prob(r.mean_jet_probability)) + '</span></div>' +
      (r.sr_efficiency_threshold05 != null ? '<div class="stat"><span class="stat__label">Efficiency counting jets at p &gt; 0.5</span><span class="stat__value">' + esc(Number(r.sr_efficiency_threshold05).toFixed(fmt.effDigits(r.sr_efficiency_err))) + '</span></div>' : '') +
      '</div>' +
      '<p class="muted small" style="margin:0">Surrogate ' + esc(r.model || a.model && a.model.file || '') + (r.version ? ', version ' + esc(r.version) : '') + '</p>' +
      '</div>' +
      '<div class="result__cell">' + histogramSvg(r.histogram) +
      '<p class="hist__caption">Each event gets a probability of entering the signal region, combined from the per-jet surrogate outputs. The efficiency is the mean of this distribution. The per-event values are in the HDF5 download.</p></div>' +
      '</div>' +
      '<p class="result__explain"><strong>What this number means.</strong> The efficiency is the predicted fraction of your events that would pass the selection of this analysis: ' + sr + ' The uncertainty is statistical (from the number of events you uploaded) and does not include the accuracy of the surrogate itself, which is documented on the analysis page.' +
      (r.sr_efficiency_threshold05 != null ? ' The second efficiency counts a jet as tagged only when its probability is above 0.5, a cross-check that should agree with the main number when the surrogate is confident.' : '') + '</p>' +
      '<div class="result__links">' +
      '<a href="' + esc(jobUrl(job.id, '/result.h5')) + '">' + ICONS.download + ' Per-event probabilities (HDF5)</a>' +
      '<a href="' + esc(jobUrl(job.id, '/log')) + '" target="_blank" rel="noopener">' + ICONS.doc + ' Job log</a>' +
      '<a href="' + esc(permalink) + '">' + ICONS.link + ' Permalink</a>' +
      (a.id ? '<a href="analysis.html?id=' + esc(encodeURIComponent(a.id)) + '">About this analysis</a>' : '') +
      '</div></section>';
    return el(html);
  }

  /* Render a job into a container and keep polling while it is busy.
     Resolves with the final job record. */
  function trackJob(id, container, analysisPromise) {
    let stopped = false;
    let failures = 0;
    return new Promise(resolve => {
      function render(job, analysis) {
        container.innerHTML = '';
        if (job.status === 'done' && job.result) {
          container.appendChild(resultCard(job, analysis));
        } else {
          if (job.status === 'failed') container.appendChild(notification('error', 'Job failed.', job.error || 'See the log for details.'));
          container.appendChild(statusPanel(job));
        }
      }
      function tick() {
        if (stopped) return;
        Promise.all([apiGet('/jobs/' + encodeURIComponent(id)), analysisPromise || Promise.resolve(null)])
          .then(([job, analysis]) => {
            failures = 0;
            render(job, analysis || null);
            if (job.status === 'queued' || job.status === 'running') setTimeout(tick, window.JS_POLL_MS);
            else { stopped = true; resolve(job); }
          })
          .catch(err => {
            failures += 1;
            if (err.status === 404 || failures > 5) {
              container.innerHTML = '';
              container.appendChild(notification('error', 'Could not load job ' + id + '.', err.message));
              stopped = true;
              resolve(null);
            } else {
              setTimeout(tick, window.JS_POLL_MS * 2);
            }
          });
      }
      tick();
    });
  }

  /* ---------- Chrome (footer info, nav) ---------- */

  function initChrome() {
    const nav = $('.site-header__nav');
    const cur = nav && nav.querySelector('[aria-current="page"]');
    if (nav && cur) {
      const offset = cur.offsetLeft - (nav.clientWidth - cur.offsetWidth) / 2;
      if (offset > 0) nav.scrollLeft = offset;
    }
    apiGet('/info').then(info => {
      document.querySelectorAll('[data-info-version]').forEach(n => { n.textContent = info.version ? 'v' + info.version : ''; });
      document.querySelectorAll('[data-info-repo]').forEach(n => { if (info.repo_url) n.href = info.repo_url; });
      document.querySelectorAll('[data-info-name]').forEach(n => { if (info.name) n.textContent = info.name; });
      document.querySelectorAll('[data-info-tagline]').forEach(n => { if (info.tagline) n.textContent = info.tagline; });
      document.querySelectorAll('[data-info-count]').forEach(n => { if (info.n_analyses != null) n.textContent = info.n_analyses === 1 ? '1 preserved analysis' : info.n_analyses + ' preserved analyses'; });
      document.querySelectorAll('[data-info-max-upload]').forEach(n => { if (info.max_upload_mb) n.textContent = fmt.bytes(info.max_upload_mb * 1e6); });
      if (fixtureMode) {
        const f = $('.site-footer__inner');
        if (f) f.appendChild(el('<span class="muted">Preview mode: showing sample data from fixtures/ because the API is unreachable.</span>'));
      }
    }).catch(() => {});
  }

  /* ---------- Pages ---------- */

  const pages = {};

  pages.home = function () {
    const grid = $('#featured');
    if (!grid) return;
    apiGet('/analyses').then(list => fillCards(grid, list)).catch(err => showError(grid, err, 'Could not load the analysis library.'));
  };

  pages.analyses = function () {
    const grid = $('#library');
    const count = $('#library-count');
    apiGet('/analyses').then(list => {
      fillCards(grid, list);
      if (count) count.textContent = list.length === 1 ? '1 analysis' : list.length + ' analyses';
    }).catch(err => showError(grid, err, 'Could not load the analysis library.'));
  };

  pages.analysis = function () {
    const id = param('id');
    const root = $('#analysis');
    if (!id) {
      root.innerHTML = '';
      root.appendChild(notification('warning', 'No analysis selected.', 'Pick one from the library.', ' <a href="analyses.html">Go to the library</a>'));
      return;
    }
    apiGet('/analyses/' + encodeURIComponent(id)).then(a => {
      document.title = a.title + ' · PRISM';
      $('#a-title').textContent = a.title;
      $('#a-short').textContent = a.short || '';
      $('#a-tags').innerHTML = experimentTag(a) + statusTag(a) + (a.tags || []).map(t => tag(t, 'tag--gray')).join('');
      const submit = 'submit.html?analysis=' + encodeURIComponent(a.id);
      $('#a-submit').href = submit;
      $('#a-submit-2').href = submit;
      if (a.repo_url) { $('#a-readme').href = safeUrl(a.repo_url); $('#a-readme-2').href = safeUrl(a.repo_url); }
      else { $('#a-readme').hidden = true; $('#a-readme-2').hidden = true; }
      $('#a-description').innerHTML = a.description_html || '<p class="muted">No description provided.</p>';
      $('#a-signal-region').textContent = a.signal_region || 'Not specified.';
      const inputs = $('#a-inputs');
      inputs.innerHTML = (a.inputs || []).map(s => '<li>' + esc(s) + '</li>').join('') || '<li class="muted">Not specified.</li>';
      $('#a-limits').textContent = 'Up to ' + fmt.int(a.max_events) + ' events per job (default ' + fmt.int(a.default_max_events) + ').';
      const refs = $('#a-references');
      refs.innerHTML = (a.references || []).map(r => '<li><a href="' + esc(safeUrl(r.url)) + '" rel="noopener">' + esc(r.label || r.url) + '</a></li>').join('') || '<li class="muted">No references listed.</li>';

      /* Validation figures and table */
      const figs = $('#a-figures');
      figs.innerHTML = '';
      (a.figures || []).forEach(f => {
        const fig = el('<figure><img src="' + esc(figureUrl(a.id, f.file)) + '" alt="' + esc(f.caption || f.file) + '" loading="lazy"><figcaption>' + esc(f.caption || '') + '</figcaption></figure>');
        const img = fig.querySelector('img');
        img.addEventListener('error', () => {
          const box = el('<div class="figure-missing" role="img" aria-label="' + esc(f.caption || f.file) + '">Figure ' + esc(f.file) + ' is not available.</div>');
          img.replaceWith(box);
        });
        figs.appendChild(fig);
      });
      const tbl = $('#a-validation');
      const rows = a.validation || [];
      if (rows.length) {
        const cols = [];
        rows.forEach(r => Object.keys(r).forEach(k => { if (cols.indexOf(k) < 0) cols.push(k); }));
        const isNum = k => rows.every(r => r[k] == null || typeof r[k] === 'number');
        tbl.innerHTML = '<div class="table-wrap"><table class="data"><thead><tr>' +
          cols.map(k => '<th' + (isNum(k) ? ' class="num"' : '') + ' scope="col">' + esc(k.replace(/_/g, ' ')) + '</th>').join('') +
          '</tr></thead><tbody>' +
          rows.map(r => '<tr>' + cols.map(k => '<td' + (isNum(k) ? ' class="num"' : '') + '>' + esc(typeof r[k] === 'number' ? (Math.abs(r[k]) < 1 ? r[k].toFixed(3) : r[k]) : (r[k] == null ? '' : r[k])) + '</td>').join('') + '</tr>').join('') +
          '</tbody></table></div>';
      } else {
        tbl.innerHTML = '';
      }
      if (!rows.length && !(a.figures || []).length) {
        $('#a-validation-empty').hidden = false;
      }

      /* Model provenance */
      const m = a.model || {};
      $('#a-model').innerHTML =
        '<dl class="kv">' +
        '<dt>Predictor type</dt><dd><code>' + esc(m.type || '') + '</code></dd>' +
        '<dt>Model file</dt><dd><code>' + esc(m.file || '') + '</code></dd>' +
        '<dt>Model version</dt><dd>' + esc(m.version || a.version || '') + '</dd>' +
        '<dt>Record version</dt><dd>' + esc(a.version || '') + '</dd>' +
        '<dt>Updated</dt><dd>' + esc(a.updated || '') + '</dd>' +
        (a.contact ? '<dt>Contact</dt><dd>' + esc(a.contact) + '</dd>' : '') +
        '</dl>' +
        (m.training ? '<h4>Training summary</h4><p>' + esc(m.training) + '</p>' : '');
      root.hidden = false;
      $('#analysis-loading').hidden = true;
    }).catch(err => {
      $('#analysis-loading').innerHTML = '';
      $('#analysis-loading').appendChild(notification('error', 'Could not load analysis "' + id + '".', err.message, ' <a href="analyses.html">Back to the library</a>'));
    });
  };

  pages.submit = function () {
    const form = $('#submit-form');
    const select = $('#analysis');
    const help = $('#analysis-help');
    const srText = $('#signal-region');
    const inputsList = $('#inputs-list');
    const drop = $('#dropzone');
    const fileInput = $('#file-input');
    const fileInfo = $('#file-info');
    const labelInput = $('#label');
    const maxInput = $('#max-events');
    const maxHelp = $('#max-events-help');
    const btn = $('#submit-btn');
    const progress = $('#upload-progress');
    const notes = $('#notifications');
    const jobSlot = $('#job-slot');
    const analysisLink = $('#analysis-link');
    let analyses = [];
    let current = null;
    let file = null;
    let maxUploadBytes = 0;
    let busy = false;

    function note(kind, title, text, extra) {
      notes.innerHTML = '';
      notes.appendChild(notification(kind, title, text, extra));
    }

    function setFile(f) {
      file = f || null;
      fileInfo.innerHTML = '';
      if (!file) { drop.hidden = false; return; }
      const tooBig = maxUploadBytes && file.size > maxUploadBytes;
      fileInfo.appendChild(el('<div class="file-item"><span class="file-item__name">' + esc(file.name) + '</span>' +
        '<span class="file-item__size">' + esc(fmt.bytes(file.size)) + '</span>' +
        '<button type="button" class="file-item__remove" aria-label="Remove file"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M24 9.4L22.6 8 16 14.6 9.4 8 8 9.4l6.6 6.6L8 22.6 9.4 24l6.6-6.6 6.6 6.6 1.4-1.4-6.6-6.6L24 9.4z"/></svg></button></div>'));
      if (tooBig) fileInfo.appendChild(el('<p class="field__help field__help--error">This file is larger than the upload limit of ' + esc(fmt.bytes(maxUploadBytes)) + '. Reduce the number of events or gzip the file.</p>'));
      fileInfo.querySelector('.file-item__remove').addEventListener('click', () => { fileInput.value = ''; setFile(null); });
      drop.hidden = true;
    }

    function selectAnalysis(id) {
      current = analyses.find(a => a.id === id) || null;
      if (!current) { help.textContent = ''; srText.textContent = ''; inputsList.innerHTML = ''; return; }
      help.textContent = current.short || '';
      analysisLink.href = 'analysis.html?id=' + encodeURIComponent(current.id);
      analysisLink.hidden = false;
      maxInput.value = current.default_max_events || '';
      maxInput.max = current.max_events || '';
      maxHelp.textContent = 'Only the first N events of the file are used. The limit for this analysis is ' + fmt.int(current.max_events) + ' events.';
      srText.textContent = 'Loading the signal-region definition.';
      apiGet('/analyses/' + encodeURIComponent(current.id)).then(a => {
        if (!current || a.id !== current.id) return;
        Object.assign(current, a);
        srText.textContent = a.signal_region || 'This analysis does not describe its signal region.';
        inputsList.innerHTML = (a.inputs || []).map(s => '<li>' + esc(s) + '</li>').join('');
      }).catch(() => { srText.textContent = 'The signal-region definition could not be loaded.'; });
    }

    apiGet('/info').then(info => { maxUploadBytes = (info.max_upload_mb || 0) * 1e6; if (file) setFile(file); }).catch(() => {});

    apiGet('/analyses').then(list => {
      analyses = list;
      select.innerHTML = list.map(a => '<option value="' + esc(a.id) + '">' + esc(a.title) + '</option>').join('');
      if (!list.length) {
        select.innerHTML = '<option value="">No analyses available</option>';
        btn.disabled = true;
        return;
      }
      const wanted = param('analysis');
      if (wanted && list.some(a => a.id === wanted)) select.value = wanted;
      else if (wanted) note('warning', 'Unknown analysis "' + wanted + '".', 'Showing the first analysis in the library instead.');
      selectAnalysis(select.value);
    }).catch(err => {
      note('error', 'Could not load the analysis library.', err.message);
      btn.disabled = true;
    });

    select.addEventListener('change', () => selectAnalysis(select.value));

    /* Drop zone: click or keyboard opens the picker, drag and drop sets the file */
    drop.addEventListener('click', () => fileInput.click());
    drop.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });
    ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('is-dragover'); }));
    ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('is-dragover'); }));
    drop.addEventListener('drop', e => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) setFile(f);
    });
    fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

    form.addEventListener('submit', e => {
      e.preventDefault();
      if (busy) return;
      notes.innerHTML = '';
      if (!current) { note('error', 'Pick an analysis first.'); return; }
      if (!file) { note('error', 'Choose a HepMC file to upload.'); drop.focus(); return; }
      if (maxUploadBytes && file.size > maxUploadBytes) { note('error', 'The file is larger than the upload limit of ' + fmt.bytes(maxUploadBytes) + '.'); return; }
      let maxEvents = parseInt(maxInput.value, 10);
      if (!(maxEvents > 0)) maxEvents = current.default_max_events || 1;
      if (current.max_events) maxEvents = Math.min(maxEvents, current.max_events);
      maxInput.value = maxEvents;

      const fd = new FormData();
      fd.append('analysis', current.id);
      fd.append('file', file, file.name);
      if (labelInput.value.trim()) fd.append('label', labelInput.value.trim().slice(0, 200));
      fd.append('max_events', String(maxEvents));

      busy = true;
      btn.disabled = true;
      btn.textContent = 'Uploading';
      jobSlot.innerHTML = '';
      progress.hidden = false;
      progress.classList.remove('progress--indeterminate');
      const bar = progress.querySelector('.progress__bar');
      const lab = progress.querySelector('.progress__value');
      bar.style.width = '0%';
      lab.textContent = '0 %';
      progress.setAttribute('aria-valuenow', '0');

      apiUpload('/jobs', fd, (loaded, total) => {
        if (total) {
          const p = Math.min(100, Math.round(100 * loaded / total));
          bar.style.width = p + '%';
          lab.textContent = fmt.bytes(loaded) + ' of ' + fmt.bytes(total) + ' (' + p + ' %)';
          progress.setAttribute('aria-valuenow', String(p));
          if (p >= 100) { lab.textContent = 'Upload complete, waiting for the server to store the file'; progress.classList.add('progress--indeterminate'); }
        } else {
          lab.textContent = fmt.bytes(loaded) + ' sent';
          progress.classList.add('progress--indeterminate');
        }
      }).then(res => {
        progress.hidden = true;
        note('success', 'Job ' + res.id + ' submitted.', 'The result appears below when the job finishes.', ' <a href="jobs.html?id=' + esc(encodeURIComponent(res.id)) + '">Permalink</a>');
        const a = current;
        return trackJob(res.id, jobSlot, Promise.resolve(a));
      }).catch(err => {
        progress.hidden = true;
        note('error', 'Submission failed.', err.message);
      }).then(() => {
        busy = false;
        btn.disabled = false;
        btn.textContent = 'Submit';
      });
    });
  };

  pages.jobs = function () {
    const id = param('id');
    const single = $('#job-single');
    const listWrap = $('#job-list');
    const analysesP = apiGet('/analyses').catch(() => []);

    if (id) {
      single.hidden = false;
      $('#job-heading').textContent = 'Job ' + id;
      const slot = $('#job-slot');
      slot.innerHTML = '<p class="muted">Loading job.</p>';
      const analysisP = apiGet('/jobs/' + encodeURIComponent(id))
        .then(j => j.analysis ? apiGet('/analyses/' + encodeURIComponent(j.analysis)).catch(() => ({ id: j.analysis, title: j.analysis })) : null)
        .catch(() => null);
      trackJob(id, slot, analysisP);
    }

    Promise.all([apiGet('/jobs?limit=50'), analysesP]).then(([jobs, analyses]) => {
      const titles = {};
      (analyses || []).forEach(a => { titles[a.id] = a.title; });
      if (!jobs.length) {
        listWrap.innerHTML = '<div class="empty">No jobs yet. <a href="submit.html">Submit the first one.</a></div>';
        return;
      }
      listWrap.innerHTML = '<div class="table-wrap"><table class="data"><thead><tr>' +
        '<th scope="col">Job</th><th scope="col">Analysis</th><th scope="col">Label</th><th scope="col">Status</th><th scope="col">Submitted</th><th scope="col" class="num">Efficiency</th><th scope="col"><span class="visually-hidden">Details</span></th>' +
        '</tr></thead><tbody>' +
        jobs.map(j => {
          const r = j.result;
          const link = 'jobs.html?id=' + encodeURIComponent(j.id);
          return '<tr>' +
            '<td class="nowrap"><a class="job-id" href="' + esc(link) + '">' + esc(j.id) + '</a></td>' +
            '<td><a href="analysis.html?id=' + esc(encodeURIComponent(j.analysis || '')) + '">' + esc(titles[j.analysis] || j.analysis || '') + '</a></td>' +
            '<td>' + esc(j.label || '') + '</td>' +
            '<td class="nowrap">' + jobStatusTag(j.status) + (j.status === 'running' && j.progress ? '<span class="muted small"> ' + esc(j.progress) + '</span>' : '') + '</td>' +
            '<td class="nowrap"><time datetime="' + esc(fmt.isoDate(j.created)) + '">' + esc(fmt.date(j.created)) + '</time></td>' +
            '<td class="num nowrap">' + (r && r.sr_efficiency != null ? esc(fmt.eff(r.sr_efficiency, r.sr_efficiency_err)) : '<span class="muted">' + (j.status === 'failed' ? 'failed' : '') + '</span>') + '</td>' +
            '<td class="nowrap"><a href="' + esc(link) + '">View</a></td>' +
            '</tr>';
        }).join('') + '</tbody></table></div>';
    }).catch(err => showError(listWrap, err, 'Could not load the job list.'));
  };

  /* ---------- Boot ---------- */

  document.addEventListener('DOMContentLoaded', () => {
    initChrome();
    const page = document.body.getAttribute('data-page');
    if (page && pages[page]) pages[page]();
  });

  /* Expose a few helpers for debugging and for pages with extra scripts */
  window.SR = { apiGet, apiUpload, fmt, histogramSvg, resultCard, statusPanel, trackJob, notification, esc };
})();
