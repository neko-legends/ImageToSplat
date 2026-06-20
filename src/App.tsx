import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  Box,
  CheckCircle2,
  Download,
  FolderOpen,
  Image as ImageIcon,
  Play,
  RefreshCw,
  Settings,
  TerminalSquare,
  Wand2,
  XCircle,
} from 'lucide-react'

const DEFAULT_PORT = 17340
const SAMPLE_IMAGES = [
  'static/example_inputs/creature_butterfly.webp',
  'static/example_inputs/building_stone_house.webp',
  'static/example_inputs/vehicle_pirate_ship.webp',
  'static/example_inputs/plant_water_lily.webp',
]

type AgentStatus = {
  ok: boolean
  enabled: boolean
  port: number
  url: string
  busy: boolean
  activeJob?: AgentJob | null
  lastJob?: AgentJob | null
  outputRoot?: string
  models?: ModelStatus
}

type AgentJob = {
  id: string
  action: string
  status: 'running' | 'completed' | 'failed'
  message?: string
  progress?: number
  error?: string
  result?: GenerationResult | SetupResult
}

type ModelStatus = {
  ready: boolean
  ckptRoot: string
  missing: Array<{ label: string; relativePath: string; path: string }>
}

type SetupResult = {
  ok: boolean
  ckptRoot: string
  missing: string[]
  message: string
}

type GenerationResult = {
  ok: boolean
  outputDir: string
  gaussianCount: number
  generationSeconds: number
  paths: Record<string, string>
  settings: {
    seed: number
    steps: number
    guidanceScale: number
    numGaussians: number
  }
}

type LaunchStatus = {
  url: string
  port: number
  started: boolean
  message: string
}

function isTauriRuntime() {
  return typeof window !== 'undefined' && Boolean((window as any).__TAURI_INTERNALS__)
}

function formatError(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function fileName(path: string) {
  return path.replace(/\\/g, '/').split('/').pop() || path
}

function isGenerationResult(result: AgentJob['result']): result is GenerationResult {
  return Boolean(result && 'paths' in result)
}

export default function App() {
  const [port, setPort] = useState(DEFAULT_PORT)
  const [apiUrl, setApiUrl] = useState(`http://127.0.0.1:${DEFAULT_PORT}`)
  const [status, setStatus] = useState<AgentStatus | null>(null)
  const [selectedImage, setSelectedImage] = useState('')
  const [outputDir, setOutputDir] = useState('')
  const [outputName, setOutputName] = useState('')
  const [seed, setSeed] = useState(42)
  const [steps, setSteps] = useState(20)
  const [guidanceScale, setGuidanceScale] = useState(3)
  const [numGaussians, setNumGaussians] = useState(262144)
  const [message, setMessage] = useState('Starting local service')
  const [polling, setPolling] = useState(false)
  const [viewerSrc, setViewerSrc] = useState('')
  const [preparedSrc, setPreparedSrc] = useState('')
  const hasBooted = useRef(false)
  const tauri = isTauriRuntime()

  const activeJob = status?.activeJob
  const lastJob = status?.lastJob
  const generation = isGenerationResult(lastJob?.result) ? lastJob.result : null
  const modelReady = Boolean(status?.models?.ready)
  const busy = Boolean(status?.busy)
  const progress = Math.round(((activeJob?.progress ?? (busy ? 0 : 0)) || 0) * 100)

  const statusTone = useMemo(() => {
    if (!status) return 'idle'
    if (status.busy) return 'busy'
    if (status.models?.ready) return 'ready'
    return 'setup'
  }, [status])

  const refreshStatus = useCallback(async () => {
    const response = await fetch(`${apiUrl}/status`)
    if (!response.ok) {
      throw new Error(`Service returned ${response.status}`)
    }
    const next = (await response.json()) as AgentStatus
    setStatus(next)
    setMessage(next.busy ? next.activeJob?.message || 'Running' : 'Ready')
    setPolling(next.busy)
    return next
  }, [apiUrl])

  async function startService() {
    setMessage('Starting local service')
    try {
      if (tauri) {
        const { invoke } = await import('@tauri-apps/api/core')
        const launch = await invoke<LaunchStatus>('start_agent_api', { port })
        setApiUrl(launch.url)
        setMessage(launch.message)
      } else {
        setApiUrl(`http://127.0.0.1:${port}`)
        setMessage('Checking local service')
      }
    } catch (error) {
      setMessage(`Service start failed: ${formatError(error)}`)
    }
  }

  async function chooseImage() {
    if (!tauri) {
      setMessage('Desktop file picker is available in the Tauri app')
      return
    }
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const selected = await open({
        multiple: false,
        directory: false,
        filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp'] }],
      })
      if (typeof selected === 'string') {
        setSelectedImage(selected)
        if (!outputName.trim()) {
          setOutputName(fileName(selected).replace(/\.[^.]+$/, ''))
        }
      }
    } catch (error) {
      setMessage(`Could not pick image: ${formatError(error)}`)
    }
  }

  async function chooseOutputDir() {
    if (!tauri) {
      setMessage('Desktop folder picker is available in the Tauri app')
      return
    }
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const selected = await open({ multiple: false, directory: true })
      if (typeof selected === 'string') setOutputDir(selected)
    } catch (error) {
      setMessage(`Could not pick output folder: ${formatError(error)}`)
    }
  }

  async function postJob(path: '/setup' | '/generate', body: Record<string, unknown>) {
    const response = await fetch(`${apiUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const json = await response.json()
    if (!response.ok || json.ok === false) {
      throw new Error(json.error || `Request failed with ${response.status}`)
    }
    setPolling(true)
    setMessage('Queued')
  }

  async function runSetup() {
    try {
      await postJob('/setup', {})
    } catch (error) {
      setMessage(`Setup failed: ${formatError(error)}`)
    }
  }

  async function runGeneration() {
    if (!selectedImage.trim()) {
      setMessage('Choose an input image first')
      return
    }
    try {
      await postJob('/generate', {
        imagePath: selectedImage.trim(),
        outputDir: outputDir.trim() || undefined,
        outputName: outputName.trim() || undefined,
        seed,
        steps,
        guidanceScale,
        numGaussians,
      })
    } catch (error) {
      setMessage(`Generation failed: ${formatError(error)}`)
    }
  }

  async function openOutput(path: string) {
    if (!path) return
    try {
      if (tauri) {
        const { openPath } = await import('@tauri-apps/plugin-opener')
        await openPath(path)
      } else {
        setMessage(path)
      }
    } catch (error) {
      setMessage(`Open failed: ${formatError(error)}`)
    }
  }

  useEffect(() => {
    if (hasBooted.current) return
    hasBooted.current = true
    startService()
  }, [])

  useEffect(() => {
    refreshStatus().catch((error) => {
      setStatus(null)
      setMessage(`Service offline: ${formatError(error)}`)
    })
  }, [apiUrl, refreshStatus])

  useEffect(() => {
    if (!polling) return
    const timer = window.setInterval(() => {
      refreshStatus().catch((error) => {
        setPolling(false)
        setMessage(`Status failed: ${formatError(error)}`)
      })
    }, 1200)
    return () => window.clearInterval(timer)
  }, [polling, refreshStatus])

  useEffect(() => {
    async function updatePreview() {
      if (!tauri || !generation?.paths) {
        setViewerSrc('')
        setPreparedSrc('')
        return
      }
      const { convertFileSrc } = await import('@tauri-apps/api/core')
      const prepared = generation.paths.prepared
      const ply = generation.paths.ply
      setPreparedSrc(prepared ? convertFileSrc(prepared) : '')
      setViewerSrc(ply ? `/viewer/viewer.html?ply=${encodeURIComponent(convertFileSrc(ply))}&ts=${Date.now()}` : '')
    }
    updatePreview().catch((error) => setMessage(`Preview failed: ${formatError(error)}`))
  }, [generation, tauri])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header className="brand">
          <img src="/app-icon.png" alt="" />
          <div>
            <h1>Neko Splat Forge</h1>
            <p>ImageToSplat</p>
          </div>
        </header>

        <section className="section">
          <div className="section-title">
            <Activity size={14} />
            Service
          </div>
          <div className={`status-pill ${statusTone}`}>
            {busy ? 'Running' : modelReady ? 'Ready' : status ? 'Setup needed' : 'Offline'}
          </div>
          <div className="status-copy">{message}</div>
          <label className="field compact">
            <span>Port</span>
            <input type="number" value={port} min={1} max={65535} onChange={(event) => setPort(Number(event.target.value))} />
          </label>
          <div className="button-row">
            <button type="button" onClick={startService}>
              <TerminalSquare size={15} />
              Start
            </button>
            <button type="button" onClick={() => refreshStatus().catch((error) => setMessage(formatError(error)))}>
              <RefreshCw size={15} />
              Refresh
            </button>
          </div>
          <button type="button" className="primary" disabled={busy || modelReady} onClick={runSetup}>
            <Download size={15} />
            Setup models
          </button>
        </section>

        <section className="section">
          <div className="section-title">
            <ImageIcon size={14} />
            Input
          </div>
          <button type="button" onClick={chooseImage}>
            <FolderOpen size={15} />
            Choose image
          </button>
          <input
            className="path-input"
            value={selectedImage}
            placeholder="Image path"
            onChange={(event) => setSelectedImage(event.target.value)}
          />
          <div className="sample-grid">
            {SAMPLE_IMAGES.map((sample) => (
              <button key={sample} type="button" onClick={() => setSelectedImage(sample)}>
                {fileName(sample).replace(/\.[^.]+$/, '').replace(/_/g, ' ')}
              </button>
            ))}
          </div>
        </section>

        <section className="section">
          <div className="section-title">
            <Settings size={14} />
            Generation
          </div>
          <label className="field">
            <span>Gaussians</span>
            <select value={numGaussians} onChange={(event) => setNumGaussians(Number(event.target.value))}>
              <option value={32768}>32,768</option>
              <option value={65536}>65,536</option>
              <option value={131072}>131,072</option>
              <option value={262144}>262,144</option>
            </select>
          </label>
          <label className="field">
            <span>Steps</span>
            <input type="range" min={1} max={50} value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
            <b>{steps}</b>
          </label>
          <label className="field">
            <span>Guidance</span>
            <input
              type="range"
              min={1}
              max={10}
              step={0.5}
              value={guidanceScale}
              onChange={(event) => setGuidanceScale(Number(event.target.value))}
            />
            <b>{guidanceScale.toFixed(1)}</b>
          </label>
          <label className="field compact">
            <span>Seed</span>
            <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
          </label>
          <label className="field compact">
            <span>Name</span>
            <input value={outputName} placeholder="Auto" onChange={(event) => setOutputName(event.target.value)} />
          </label>
          <button type="button" onClick={chooseOutputDir}>
            <FolderOpen size={15} />
            Output folder
          </button>
          <input
            className="path-input"
            value={outputDir}
            placeholder={status?.outputRoot || 'Default output folder'}
            onChange={(event) => setOutputDir(event.target.value)}
          />
          <button type="button" className="primary action" disabled={busy || !status} onClick={runGeneration}>
            <Wand2 size={16} />
            Generate splat
          </button>
        </section>
      </aside>

      <main className="workspace">
        <section className="hero-band">
          <div>
            <h2>Single image to Gaussian splat</h2>
            <p>{selectedImage ? fileName(selectedImage) : 'No image selected'}</p>
          </div>
          <div className="job-meter">
            <span>{activeJob?.action || lastJob?.action || 'idle'}</span>
            <strong>{busy ? `${progress}%` : lastJob?.status || 'standby'}</strong>
          </div>
        </section>

        {busy && (
          <div className="progress-track">
            <div style={{ width: `${progress}%` }} />
          </div>
        )}

        <section className="preview-layout">
          <div className="preview-stage">
            {viewerSrc ? (
              <iframe src={viewerSrc} title="Gaussian splat preview" />
            ) : (
              <div className="empty-preview">
                <Box size={36} />
                <span>Preview appears after a PLY export</span>
              </div>
            )}
          </div>

          <div className="details-pane">
            <div className="result-state">
              {lastJob?.status === 'failed' ? <XCircle size={18} /> : <CheckCircle2 size={18} />}
              <span>{lastJob ? `${lastJob.action}: ${lastJob.status}` : 'No completed job yet'}</span>
            </div>
            {lastJob?.error && <div className="error-box">{lastJob.error}</div>}
            {preparedSrc && <img className="prepared" src={preparedSrc} alt="Prepared input" />}
            {generation && (
              <>
                <dl className="metrics">
                  <div>
                    <dt>Gaussians</dt>
                    <dd>{generation.gaussianCount.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Seconds</dt>
                    <dd>{generation.generationSeconds.toFixed(1)}</dd>
                  </div>
                  <div>
                    <dt>Output</dt>
                    <dd>{fileName(generation.outputDir)}</dd>
                  </div>
                </dl>
                <div className="artifact-grid">
                  {Object.entries(generation.paths)
                    .filter(([key]) => key !== 'prepared')
                    .map(([key, path]) => (
                      <button key={key} type="button" onClick={() => openOutput(path)}>
                        <Play size={13} />
                        {key.toUpperCase()}
                      </button>
                    ))}
                </div>
              </>
            )}
            {!modelReady && status?.models && (
              <div className="missing-list">
                <strong>Missing model files</strong>
                {status.models.missing.map((item) => (
                  <span key={item.relativePath}>{item.relativePath}</span>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  )
}
