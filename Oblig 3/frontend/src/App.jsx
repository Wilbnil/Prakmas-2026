import { useEffect, useState, useCallback } from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ReferenceLine, ReferenceDot, BarChart, Bar, CartesianGrid,
} from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'
const api = async (path, opts) => {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' }, ...opts,
  })
  const body = await res.json()
  if (!res.ok) throw new Error(body.error || res.statusText)
  return body
}

const LOCATIONS = ['Adelaide','Albany','Albury','AliceSprings','BadgerysCreek','Ballarat','Bendigo','Brisbane','Cairns','Canberra','Cobar','CoffsHarbour','Dartmoor','Darwin','GoldCoast','Hobart','Katherine','Launceston','Melbourne','MelbourneAirport','Mildura','Moree','MountGambier','MountGinini','Newcastle','Nhil','NorahHead','NorfolkIsland','Nuriootpa','PearceRAAF','Penrith','Perth','PerthAirport','Portland','Richmond','Sale','SalmonGums','Sydney','SydneyAirport','Townsville','Tuggeranong','Uluru','WaggaWagga','Walpole','Watsonia','Williamtown','Witchcliffe','Wollongong','Woomera']
const DIRS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
const num = (name, label, unit) => ({ name, label: unit ? `${label} (${unit})` : label, type: 'number' })
const sel = (name, label, options) => ({ name, label, type: 'select', options })

const SECTIONS = [
  { title: 'When and where', fields: [
    { name: 'Date', label: 'Date', type: 'date' },
    sel('Location', 'Location', LOCATIONS),
    sel('RainToday', 'Rain today', ['No', 'Yes']),
  ]},
  { title: 'Temperature, rain and sun', fields: [
    num('MinTemp','Min temp','°C'), num('MaxTemp','Max temp','°C'),
    num('Temp9am','Temp 9am','°C'), num('Temp3pm','Temp 3pm','°C'),
    num('Rainfall','Rainfall','mm'), num('Evaporation','Evaporation','mm'), num('Sunshine','Sunshine','hours'),
  ]},
  { title: 'Wind', fields: [
    sel('WindGustDir','Gust direction',DIRS), num('WindGustSpeed','Gust speed','km/h'),
    sel('WindDir9am','Direction 9am',DIRS), sel('WindDir3pm','Direction 3pm',DIRS),
    num('WindSpeed9am','Speed 9am','km/h'), num('WindSpeed3pm','Speed 3pm','km/h'),
  ]},
  { title: 'Humidity, pressure and cloud', fields: [
    num('Humidity9am','Humidity 9am','%'), num('Humidity3pm','Humidity 3pm','%'),
    num('Pressure9am','Pressure 9am','hPa'), num('Pressure3pm','Pressure 3pm','hPa'),
    num('Cloud9am','Cloud 9am','oktas'), num('Cloud3pm','Cloud 3pm','oktas'),
  ]},
]

const EXAMPLE = { Date:'2017-06-20', Location:'Sydney', RainToday:'Yes', MinTemp:12, MaxTemp:19, Temp9am:14, Temp3pm:17,
  Rainfall:6.4, Evaporation:3, Sunshine:2, WindGustDir:'SSW', WindGustSpeed:48, WindDir9am:'S', WindDir3pm:'SSE',
  WindSpeed9am:20, WindSpeed3pm:28, Humidity9am:88, Humidity3pm:78, Pressure9am:1014, Pressure3pm:1012, Cloud9am:7, Cloud3pm:7 }

const pct = (v) => (v * 100).toFixed(1) + '%'

export default function App() {
  const [models, setModels] = useState([])
  const [model, setModel] = useState('rf')
  const [threshold, setThreshold] = useState(0.5)
  const [tab, setTab] = useState('predict')
  const [error, setError] = useState('')
  const [version, setVersion] = useState(0) // bumps when threshold is saved

  useEffect(() => {
    api('/models').then(setModels).catch((e) => setError(`Cannot reach backend: ${e.message}`))
  }, [])

  const current = models.find((m) => m.id === model)
  useEffect(() => { if (current) setThreshold(current.threshold) }, [model, models.length]) // eslint-disable-line

  const saveThreshold = useCallback(async (value) => {
    try {
      const res = await api(`/models/${model}/threshold`, { method: 'PUT', body: JSON.stringify({ threshold: value }) })
      setThreshold(res.threshold)
      setModels((ms) => ms.map((m) => (m.id === model ? { ...m, threshold: res.threshold } : m)))
      setVersion((v) => v + 1)
      setError('')
    } catch (e) { setError(e.message) }
  }, [model])

  return (
    <>
      <header>
        <h1>Will it rain tomorrow?</h1>
        <div className="controls">
          <label>Model
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </label>
          <label className="slider">
            <span>Decision threshold: <b>{threshold.toFixed(2)}</b> (rain is predicted at or above this probability)</span>
            <input type="range" min="0" max="1" step="0.01" value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              onPointerUp={(e) => saveThreshold(+e.target.value)}
              onKeyUp={(e) => saveThreshold(+e.target.value)} />
          </label>
          {current && (
            <button className="btn ghost" style={{ color: '#fff', borderColor: '#6fc3cc' }}
              onClick={() => saveThreshold(current.default_threshold)}>
              Reset to tuned value ({current.default_threshold.toFixed(2)})
            </button>
          )}
        </div>
      </header>
      <nav role="tablist">
        {[['predict','Predict'],['performance','Test results'],['importance','Feature importance']].map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>
        ))}
      </nav>
      <main>
        {error && <p className="err">{error}</p>}
        {tab === 'predict' && <Predict model={model} version={version} />}
        {tab === 'performance' && <Performance model={model} version={version} />}
        {tab === 'importance' && <Importance model={model} />}
      </main>
    </>
  )
}

function Predict({ model, version }) {
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => { setResult(null) }, [model, version])

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError('')
    try { setResult(await api(`/models/${model}/predict`, { method: 'POST', body: JSON.stringify(values) })) }
    catch (err) { setError(err.message) }
    setBusy(false)
  }

  return (
    <form onSubmit={submit}>
      {SECTIONS.map((s) => (
        <fieldset key={s.title}>
          <legend>{s.title}</legend>
          <div className="grid">
            {s.fields.map((f) => (
              <label className="field" key={f.name}>{f.label}
                {f.type === 'select' ? (
                  <select value={values[f.name] ?? ''} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}>
                    <option value="">Unknown</option>
                    {f.options.map((o) => <option key={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type={f.type} step="any" value={values[f.name] ?? ''}
                    onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
                )}
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      <p className="muted">Leave a field empty if the value is unknown. The model fills it in the same way it did during training.</p>
      <div className="actions">
        <button className="btn" disabled={busy}>{busy ? 'Predicting…' : 'Predict rain'}</button>
        <button type="button" className="btn ghost" onClick={() => setValues(EXAMPLE)}>Fill example</button>
        <button type="button" className="btn ghost" onClick={() => { setValues({}); setResult(null) }}>Clear form</button>
      </div>
      {error && <p className="err">{error}</p>}
      {result && (
        <div className={'result ' + (result.label ? 'rain' : '')} aria-live="polite">
          <h3>{result.label ? 'Rain expected tomorrow' : 'No rain expected tomorrow'}</h3>
          <div>Probability of rain: <b>{pct(result.proba)}</b></div>
          <div className="bar">
            <i style={{ width: pct(result.proba) }} />
            <u style={{ left: pct(result.threshold) }} title="Threshold" />
          </div>
          <div className="muted">Orange marker: threshold at {result.threshold.toFixed(2)}</div>
        </div>
      )}
    </form>
  )
}

function useFetch(path, deps) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    setError('');
    api(path)
      .then((result) => {
        if (active) setData(result);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, deps);
  return [data, error];
}

const METRIC_LABELS = [['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1'],['f2','F2'],['auc','ROC AUC']]
const axis = { stroke: '#5d6f74', fontSize: 12 }

function Curve({ title, data, xl, yl, point, diagonal }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ left: 0, right: 12, top: 8, bottom: 16 }}>
          <CartesianGrid stroke="#e3eae9" />
          <XAxis dataKey="x" type="number" domain={[0, 1]} label={{ value: xl, position: 'insideBottom', offset: -8 }} {...axis} />
          <YAxis type="number" domain={[0, 1]} label={{ value: yl, angle: -90, position: 'insideLeft' }} {...axis} />
          <Tooltip formatter={(v) => v.toFixed(3)} labelFormatter={(v) => `${xl}: ${(+v).toFixed(3)}`} />
          {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#9aa9ac" strokeDasharray="4 4" />}
          <Line dataKey="y" dot={false} stroke="#2f5d9e" strokeWidth={2} isAnimationActive={false} name={yl} />
          {point && <ReferenceDot x={point.x} y={point.y} r={6} fill="#c4572a" stroke="#fff" />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function Performance({ model, version }) {
  const [d, error] = useFetch(`/models/${model}/metrics`, [model, version])
  if (error) return <p className="err">{error}</p>
  if (!d) return <p className="muted">Loading test results…</p>
  const c = d.confusion
  return (
    <>
      <p className="muted">Results on the fold-1 test set, at threshold {d.threshold.toFixed(2)}.</p>
      <div className="tiles">
        {METRIC_LABELS.map(([k, l]) => <div className="tile" key={k}><span>{l}</span><b>{d.metrics[k].toFixed(3)}</b></div>)}
      </div>
      <div className="charts">
        <div className="card">
          <h2>Confusion matrix</h2>
          <div className="cm">
            <div className="h" /><div className="h">Predicted dry</div><div className="h">Predicted rain</div>
            <div className="h">Actually dry</div><div className="good"><b>{c.tn}</b>correct</div><div className="bad"><b>{c.fp}</b>false alarm</div>
            <div className="h">Actually rain</div><div className="bad"><b>{c.fn}</b>missed rain</div><div className="good"><b>{c.tp}</b>correct</div>
          </div>
        </div>
        <Curve title="ROC curve" data={d.roc} xl="False positive rate" yl="True positive rate" diagonal
          point={{ x: d.operating_point.fpr, y: d.operating_point.tpr }} />
        <Curve title="Precision–recall curve" data={d.pr} xl="Recall" yl="Precision"
          point={{ x: d.metrics.recall, y: d.metrics.precision }} />
        <div className="card">
          <h2>Metrics across thresholds</h2>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={d.sweep} margin={{ left: 0, right: 12, top: 8, bottom: 16 }}>
              <CartesianGrid stroke="#e3eae9" />
              <XAxis dataKey="threshold" type="number" domain={[0, 1]} label={{ value: 'Threshold', position: 'insideBottom', offset: -8 }} {...axis} />
              <YAxis domain={[0, 1]} {...axis} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Legend verticalAlign="top" />
              <ReferenceLine x={d.threshold} stroke="#c4572a" />
              <Line dataKey="precision" stroke="#2f5d9e" dot={false} isAnimationActive={false} />
              <Line dataKey="recall" stroke="#1f6f78" dot={false} isAnimationActive={false} />
              <Line dataKey="f2" stroke="#c4572a" dot={false} isAnimationActive={false} name="F2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  )
}

function Importance({ model }) {
  const [d, error] = useFetch(`/models/${model}/importance`, [model])
  if (error) return <p className="err">{error}</p>
  if (!d) return <p className="muted">Loading feature importance…</p>
  if (!d.features.length) return <p className="err">No importance data found. Run the export code in analysis.py first.</p>
  const rows = d.features.map((f) => ({ ...f, mean: +f.mean.toFixed(4) }))
  return (
    <div className="card">
      <h2>Which features matter most</h2>
      <p className="muted">{d.method}. Longer bars mean the model relies more on that feature.</p>
      <ResponsiveContainer width="100%" height={Math.max(300, rows.length * 28)}>
        <BarChart data={rows} layout="vertical" margin={{ left: 30, right: 20 }}>
          <CartesianGrid stroke="#e3eae9" horizontal={false} />
          <XAxis type="number" {...axis} />
          <YAxis type="category" dataKey="feature" width={110} {...axis} />
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <Bar dataKey="mean" fill="#1f6f78" name="AUC drop" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

