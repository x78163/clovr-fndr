const TYPE_STYLES = {
  'three-leaf': { bg: 'bg-slate-600', label: '3-Leaf' },
  'four-leaf': { bg: 'bg-clover-600', label: '4-Leaf' },
  'five-leaf': { bg: 'bg-amber-600', label: '5-Leaf' },
  'six-plus-leaf': { bg: 'bg-purple-600', label: '6+ Leaf' },
}

export default function DetectionCard({ detection }) {
  const style = TYPE_STYLES[detection.class] || TYPE_STYLES['four-leaf']
  const confidence = (detection.confidence * 100).toFixed(1)

  return (
    <div className="card flex items-center gap-3">
      <span className={`${style.bg} text-white text-sm font-bold px-3 py-1 rounded-full`}>
        {style.label}
      </span>
      <div className="flex-1">
        <p className="text-sm text-slate-300">Confidence</p>
        <div className="w-full bg-slate-700 rounded-full h-2 mt-1">
          <div
            className="bg-clover-400 h-2 rounded-full transition-all"
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>
      <span className="text-lg font-bold text-clover-400">{confidence}%</span>
    </div>
  )
}
