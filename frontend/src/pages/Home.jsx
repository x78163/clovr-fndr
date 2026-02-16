import { useState } from 'react'
import PhotoUpload from '../components/PhotoUpload'
import DetectionCard from '../components/DetectionCard'
import useDetection from '../hooks/useDetection'

export default function Home() {
  const { detections, isLoading, error, processingTime, imageSize, detectImage, clearDetections } =
    useDetection()
  const [mode, setMode] = useState('upload') // 'upload' | 'camera'

  const handleUpload = async (file) => {
    clearDetections()
    await detectImage(file)
  }

  // Show only the top detection, and only if confidence >= 60%
  const topDetection = detections.length > 0
    ? detections.reduce((best, d) => d.confidence > best.confidence ? d : best)
    : null
  const filtered = topDetection && topDetection.confidence >= 0.6 ? [topDetection] : []
  const fourLeafCount = filtered.filter((d) => d.class === 'four-leaf').length

  return (
    <div className="p-4 max-w-lg mx-auto space-y-4">
      {/* Mode Toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setMode('upload')}
          className={mode === 'upload' ? 'btn-primary flex-1' : 'btn-secondary flex-1'}
        >
          Upload Photo
        </button>
        <button
          onClick={() => setMode('camera')}
          className={mode === 'camera' ? 'btn-primary flex-1' : 'btn-secondary flex-1'}
        >
          Live Camera
        </button>
      </div>

      {/* Detection Area */}
      {mode === 'upload' ? (
        <PhotoUpload
          onUpload={handleUpload}
          isLoading={isLoading}
          detections={filtered}
          imageSize={imageSize}
        />
      ) : (
        <div className="card text-center py-12">
          <p className="text-slate-400">Live camera detection coming soon</p>
          <p className="text-sm text-slate-500 mt-2">
            Use photo upload for now
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card border-red-500/50 bg-red-900/20">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {filtered.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Best Match
            </h2>
            {processingTime && (
              <span className="text-sm text-slate-400">
                {processingTime.toFixed(0)}ms
              </span>
            )}
          </div>

          {fourLeafCount > 0 && (
            <div className="card border-clover-500/50 bg-clover-900/20 text-center">
              <p className="text-2xl font-bold text-clover-400">
                {fourLeafCount} Four-Leaf Clover{fourLeafCount !== 1 ? 's' : ''} Found!
              </p>
            </div>
          )}

          {filtered.map((det, i) => (
            <DetectionCard key={i} detection={det} />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && filtered.length === 0 && !error && (
        <div className="text-center py-8">
          <p className="text-6xl mb-4">🍀</p>
          <p className="text-slate-400">Upload a photo of a clover patch</p>
          <p className="text-sm text-slate-500 mt-1">
            AI will scan for four-leaf clovers
          </p>
        </div>
      )}
    </div>
  )
}
