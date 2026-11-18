const COLORS = { BT: '#2a78d6', Debris: '#eb6834', ER: '#1baf7a', P: '#eda100', Ti: '#e87ba4', Other: '#6250d6' };
const LAYOUT_COLORS = { film_roi: '#0b84a5', weld_info: '#c0392b' };

Vue.createApp({
  template: '#app-tpl',
  data: () => ({
    models: null, modelsError: '', file: null, imageUrl: '', result: null, busy: false, error: '',
    detector: '', floor: 0.1, threshold: 0.1, showLayout: true, showLabels: true, hover: null, dragOver: false, imgW: 0,
  }),
  computed: {
    visibleDefects() {
      if (!this.result) return [];
      return this.result.defects.map((d, idx) => ({ ...d, idx })).filter(d => d.conf >= this.threshold);
    },
    counts() {
      const c = {};
      for (const d of this.visibleDefects) c[d.cls] = (c[d.cls] || 0) + 1;
      return c;
    },
    layoutBoxes() {
      const out = [];
      for (const name of ['film_roi', 'weld_info'])
        this.result[name].forEach((r, i) => out.push({ ...r, key: name + i, color: LAYOUT_COLORS[name], label: `${name} ${r.conf_pct}` }));
      return out;
    },
    // kích thước nét/chữ theo hệ toạ độ ảnh gốc, để trông giống nhau trên mọi cỡ ảnh
    scale() { return this.result && this.imgW ? this.result.width / this.imgW : 1; },
    stroke() { return 2 * this.scale; },
    fontSize() { return 13 * this.scale; },
  },
  async mounted() {
    try {
      const r = await fetch('/api/models');
      if (!r.ok) throw new Error(r.status);
      this.models = await r.json();
      this.detector = this.models.default_detector;
      this.floor = this.threshold = this.models.conf_min;
    } catch (e) { this.modelsError = String(e); }
    window.addEventListener('resize', this.measure);
  },
  methods: {
    color: c => COLORS[c] || '#e11',
    setFile(f) {
      if (!f) return;
      this.file = f; this.result = null; this.error = '';
      if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
      this.imageUrl = URL.createObjectURL(f);
    },
    onPick(e) { this.setFile(e.target.files[0]); },
    onDrop(e) { this.dragOver = false; this.setFile(e.dataTransfer.files[0]); },
    onImgLoad() { this.measure(); },
    measure() { const img = document.querySelector('.stage img'); if (img) this.imgW = img.clientWidth; },
    async run() {
      this.busy = true; this.error = '';
      const fd = new FormData();
      fd.append('file', this.file); fd.append('detector', this.detector); fd.append('conf_min', this.floor);
      try {
        const r = await fetch('/api/predict', { method: 'POST', body: fd });
        const body = await r.json();
        if (!r.ok) throw new Error(body.detail || r.status);
        this.result = body;
        this.$nextTick(this.measure);
      } catch (e) { this.error = 'Lỗi: ' + e.message; }
      finally { this.busy = false; }
    },
    labelY(d) { const h = this.fontSize * 1.3; return d.bbox[1] - h >= 0 ? d.bbox[1] - h : d.bbox[3]; },
    labelW(d) { return (d.cls.length + d.conf_pct.length + 1) * this.fontSize * 0.62 + this.fontSize * 0.5; },
    probsTitle(d) {
      return d.resnet_probs ? Object.entries(d.resnet_probs).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join('\n') : '';
    },
    download(name, href) { const a = document.createElement('a'); a.href = href; a.download = name; a.click(); },
    downloadJson() {
      const { annotated_jpeg_base64, ...data } = this.result;
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
      this.download(this.result.image.replace(/\.[^.]+$/, '') + '_result.json', url);
    },
    downloadImage() {
      this.download(this.result.image.replace(/\.[^.]+$/, '') + '_annotated.jpg', 'data:image/jpeg;base64,' + this.result.annotated_jpeg_base64);
    },
    copy(t) { navigator.clipboard && navigator.clipboard.writeText(t); },
  },
}).mount('#app');
