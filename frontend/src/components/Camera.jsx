import { useEffect } from 'react'
import useCamera from '../hooks/useCamera'

export default function Camera({ onCapture, active = false, fps = 5 }) {
  const { videoRef, canvasRef, isActive, error, startCamera, stopCamera, captureFrame } =
    useCamera()

  useEffect(() => {
    if (active && !isActive) {
      startCamera('environment')
    } else if (!active && isActive) {
      stopCamera()
    }
  }, [active, isActive, startCamera, stopCamera])

  // Auto-capture loop
  useEffect(() => {
    if (!isActive || !onCapture || !active) return

    const interval = setInterval(async () => {
      const blob = await captureFrame()
      if (blob) onCapture(blob)
    }, 1000 / fps)

    return () => clearInterval(interval)
  }, [isActive, active, onCapture, captureFrame, fps])

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 card">
        <div className="text-center">
          <p className="text-red-400 mb-2">Camera Error</p>
          <p className="text-sm text-slate-400">{error}</p>
          <button onClick={() => startCamera()} className="btn-primary mt-4">
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
