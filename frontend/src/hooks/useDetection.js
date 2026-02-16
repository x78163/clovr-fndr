import { useState, useCallback } from 'react'
import { detection } from '../services/api'

export default function useDetection() {
  const [detections, setDetections] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [processingTime, setProcessingTime] = useState(null)
  const [imageSize, setImageSize] = useState(null)

  const detectImage = useCallback(async (imageBlob, confidence = 0.25) => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await detection.detect(imageBlob, confidence)
      setDetections(response.data.detections)
      setProcessingTime(response.data.processing_time_ms)
      setImageSize(response.data.image_size)
      return response.data
    } catch (err) {
      const message = err.response?.data?.detail || err.message || 'Detection failed'
      setError(message)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const clearDetections = useCallback(() => {
    setDetections([])
    setError(null)
    setProcessingTime(null)
    setImageSize(null)
  }, [])

  return {
    detections,
    isLoading,
    error,
    processingTime,
    imageSize,
    detectImage,
    clearDetections,
  }
}
