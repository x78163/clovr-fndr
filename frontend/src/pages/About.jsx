export default function About() {
  return (
    <div className="p-4 max-w-lg mx-auto space-y-4">
      <h2 className="text-xl font-bold">About Clovr Fndr</h2>

      <div className="card space-y-3">
        <p className="text-slate-300">
          Clovr Fndr uses a custom-trained YOLO11 AI model to detect four-leaf
          clovers in photos and live camera feeds.
        </p>
        <p className="text-slate-300">
          Point your camera at a clover patch, and the AI will scan for rare
          four-leaf (and even five-leaf!) clovers in real time.
        </p>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Fun Facts</h3>
        <ul className="space-y-2 text-sm text-slate-300">
          <li>4-leaf clovers occur in about 1 in 5,000 clovers</li>
          <li>5-leaf clovers are about 1 in 20,000</li>
          <li>The record is a 56-leaf clover found in 2009 in Japan!</li>
          <li>Each leaf represents: Hope, Faith, Love, and Luck</li>
        </ul>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Detection Classes</h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-slate-400" />
            <span className="text-slate-300">3-Leaf Clover (common)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-clover-500" />
            <span className="text-slate-300">4-Leaf Clover (rare, 1 in 5,000)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-slate-300">5-Leaf Clover (very rare)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-purple-500" />
            <span className="text-slate-300">6+ Leaf Clover (ultra-rare!)</span>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Technology</h3>
        <ul className="space-y-1 text-sm text-slate-400">
          <li>Model: YOLO11 (Ultralytics)</li>
          <li>Backend: Django + DRF</li>
          <li>Frontend: React + Tailwind</li>
          <li>Training: Custom clover dataset</li>
        </ul>
      </div>

      <p className="text-center text-sm text-slate-500 pt-4">
        Clovr Fndr v0.1.0
      </p>
    </div>
  )
}
