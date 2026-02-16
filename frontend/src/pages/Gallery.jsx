import { useState, useEffect } from 'react'
import { discoveries } from '../services/api'

export default function Gallery() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchDiscoveries = async () => {
      try {
        const response = await discoveries.list()
        setItems(response.data.results || response.data || [])
      } catch (err) {
        setError('Could not load discoveries. Is the backend running?')
      } finally {
        setLoading(false)
      }
    }
    fetchDiscoveries()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-2 border-clover-400 border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className="p-4 max-w-lg mx-auto">
      <h2 className="text-xl font-bold mb-4">My Discoveries</h2>

      {error && (
        <div className="card border-yellow-500/50 bg-yellow-900/20 mb-4">
          <p className="text-yellow-400 text-sm">{error}</p>
        </div>
      )}

      {items.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-5xl mb-4">🔍</p>
          <p className="text-slate-400">No discoveries yet</p>
          <p className="text-sm text-slate-500 mt-1">
            Find and save your first four-leaf clover!
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          {items.map((item) => (
            <div key={item.id} className="card p-2">
              {item.image && (
                <img
                  src={item.image}
                  alt={item.clover_type}
                  className="w-full aspect-square object-cover rounded-lg mb-2"
                />
              )}
              <p className="text-sm font-semibold capitalize">
                {item.clover_type?.replace('-', ' ')}
              </p>
              <p className="text-xs text-slate-400">
                {item.confidence ? `${(item.confidence * 100).toFixed(0)}%` : ''}{' '}
                {item.location_name || ''}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
