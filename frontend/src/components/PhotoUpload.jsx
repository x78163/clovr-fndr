import { useState, useRef, useCallback } from 'react'
import DetectionOverlay from './DetectionOverlay'

export default function PhotoUpload({ onUpload, isLoading, detections = [], imageSize }) {
  const [preview, setPreview] = useState(null)
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })
  const inputRef = useRef(null)
  const imgRef = useRef(null)

  const updateContainerSize = useCallback(() => {
    if (imgRef.current) {
      setContainerSize({
        width: imgRef.current.clientWidth,
        height: imgRef.current.clientHeight,
      })
    }
  }, [])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setPreview(URL.createObjectURL(file))
    if (onUpload) onUpload(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (!file) return

    setPreview(URL.createObjectURL(file))
    if (onUpload) onUpload(file)
  }

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="card cursor-pointer text-center hover:border-clover-500 transition-colors"
      >
        {preview ? (
          <div className="relative">
            <img
              ref={imgRef}
              src={preview}
              alt="Upload preview"
              className="w-full rounded-lg max-h-96 object-contain"
              onLoad={updateContainerSize}
            />
            {detections.length > 0 && imageSize && containerSize.width > 0 && (
              <DetectionOverlay
                detections={detections}
                imageWidth={imageSize[0]}
                imageHeight={imageSize[1]}
                containerWidth={containerSize.width}
                containerHeight={containerSize.height}
              />
            )}
            {isLoading && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-lg">
                <div className="animate-spin w-8 h-8 border-2 border-clover-400 border-t-transparent rounded-full" />
              </div>
            )}
          </div>
        ) : (
          <div className="py-12">
            <p className="text-4xl mb-3">📷</p>
            <p className="text-slate-300">Tap to upload a photo</p>
            <p className="text-sm text-slate-500 mt-1">or drag and drop</p>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  )
}
