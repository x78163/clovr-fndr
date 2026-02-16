import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
})

export const detection = {
  detect: (imageFile, confidence = 0.25) => {
    const formData = new FormData()
    formData.append('image', imageFile)
    formData.append('confidence', confidence)
    return apiClient.post('/detect/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  health: () => apiClient.get('/detect/health/'),
}

export const discoveries = {
  list: (params) => apiClient.get('/discoveries/', { params }),
  get: (id) => apiClient.get(`/discoveries/${id}/`),
  create: (data) => {
    const formData = new FormData()
    Object.entries(data).forEach(([key, val]) => {
      if (val !== null && val !== undefined) formData.append(key, val)
    })
    return apiClient.post('/discoveries/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  delete: (id) => apiClient.delete(`/discoveries/${id}/`),
}

export default apiClient
