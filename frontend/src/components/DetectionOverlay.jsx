import { useRef, useEffect } from 'react'

const CLASS_COLORS = {
  'three-leaf': '#94a3b8',
  'four-leaf': '#a855f7',
  'five-leaf': '#f59e0b',
  'six-plus-leaf': '#a855f7',
}

export default function DetectionOverlay({
  detections = [],
  imageWidth,
  imageHeight,
  containerWidth,
  containerHeight,
}) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    canvas.width = containerWidth || 640
    canvas.height = containerHeight || 480

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!detections.length || !imageWidth || !imageHeight) return

    // Account for object-contain: image is scaled to fit with aspect ratio preserved
    const imgAspect = imageWidth / imageHeight
    const contAspect = containerWidth / containerHeight

    let renderW, renderH, offsetX, offsetY
    if (imgAspect > contAspect) {
      renderW = containerWidth
      renderH = containerWidth / imgAspect
      offsetX = 0
      offsetY = (containerHeight - renderH) / 2
    } else {
      renderH = containerHeight
      renderW = containerHeight * imgAspect
      offsetX = (containerWidth - renderW) / 2
      offsetY = 0
    }

    const scaleX = renderW / imageWidth
    const scaleY = renderH / imageHeight

    detections.forEach((det) => {
      const color = CLASS_COLORS[det.class] || '#22c55e'

      if (det.mask && det.mask.length > 2) {
        // Draw filled mask polygon
        ctx.beginPath()
        const [firstX, firstY] = det.mask[0]
        ctx.moveTo(firstX * scaleX + offsetX, firstY * scaleY + offsetY)
        for (let i = 1; i < det.mask.length; i++) {
          const [px, py] = det.mask[i]
          ctx.lineTo(px * scaleX + offsetX, py * scaleY + offsetY)
        }
        ctx.closePath()

        // Semi-transparent fill
        ctx.fillStyle = color + '40'
        ctx.fill()

        // Solid outline
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.stroke()
      } else {
        // Fallback: draw bounding box
        const [x1, y1, x2, y2] = det.bbox
        const sx1 = x1 * scaleX + offsetX
        const sy1 = y1 * scaleY + offsetY
        const sw = (x2 - x1) * scaleX
        const sh = (y2 - y1) * scaleY

        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(sx1, sy1, sw, sh)
      }

      // Draw label
      const [x1, y1] = det.bbox
      const sx1 = x1 * scaleX + offsetX
      const sy1 = y1 * scaleY + offsetY

      const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`
      ctx.font = 'bold 12px sans-serif'
      const textWidth = ctx.measureText(label).width
      ctx.fillStyle = color
      ctx.fillRect(sx1, sy1 - 18, textWidth + 8, 18)

      ctx.fillStyle = '#000'
      ctx.fillText(label, sx1 + 4, sy1 - 5)
    })
  }, [detections, imageWidth, imageHeight, containerWidth, containerHeight])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ width: '100%', height: '100%' }}
    />
  )
}
